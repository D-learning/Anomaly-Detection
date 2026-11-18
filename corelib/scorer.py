from typing import List, Union

import faiss
import numpy as np

import pickle
import os

##  3-4-14 CLASS NN(FaissNN)
class FaissNN:

    def __init__(self, on_gpu: bool = False, num_workers: int = 4, device=0) -> None:
        """
        FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

        self.device = device

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                self.device,
                index,
                self._gpu_cloner_options(),
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = self.device
            return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, cfg)
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class _BaseMerger:

    def __init__(self):
        """Merges feature embedding by name."""

    ## YTODO 3-4-12 fn Merger.merge()
    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class ConcatMerger(_BaseMerger):

    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class NearestNeighbourScorer:

    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """
        Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    ## YTODO 3-4-10 fn scorer.predict()
    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        ## YTODO 3-4-11 use feature_merger.merge(), go to 3-4-12
        query_features = self.feature_merger.merge(
            query_features,
        )

        ## YTODO 3-4-13 use NN(), go to 3-4-14
        query_distances, query_nns = self.imagelevel_nn(query_features)

        anomaly_scores = np.mean(query_distances, axis=-1)

        ## go to 3-4-9
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> str:

        save_path = self._index_file(save_folder, prepend)
        self.nn_method.save(save_path)
        if save_features_separately:
            save_path = self._detection_file(save_folder, prepend)
            self._save(save_path, self.detection_features)

        return save_path

    def save_and_reset(self, save_folder: str) -> str:
        save_path = self.save(save_folder)
        self.nn_method.reset_index()

        return save_path

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
