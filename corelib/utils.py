import math
from typing import Optional

import GPUtil
import numpy as np
import psutil


import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import cv2
import tqdm
import logging
import csv
import torch.nn as nn

LOGGER = logging.getLogger(__name__)



def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    file_name = 'results.csv'
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, file_name)
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics




def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        # image = PIL.Image.open(image_path).convert("RGB")
        # # image = image_transform(image)
        # image = transform_img(image)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(256,256))
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros([3,256,256])

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        hoi = heatmap_on_image(cv2heatmap(segmentation*255),image)
        cv2.imwrite(savename.replace('.png','_org.png'),image)
        cv2.imwrite(savename.replace('.png','_mask.png'),mask.transpose(1, 2, 0)*255)
        cv2.imwrite(savename.replace('.png','_segmentation.png'),segmentation*255)
        cv2.imwrite(savename.replace('.png','_hoi.png'),hoi)
        # f, axes = plt.subplots(1, 2 + int(masks_provided))
        # axes[0].imshow(image.transpose(1, 2, 0))
        # axes[1].imshow(mask.transpose(1, 2, 0))
        # axes[2].imshow(segmentation)
        # f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        # f.tight_layout()
        # f.savefig(savename)
        # plt.close()


def set_torch_device(gpu_ids):
    """Returns correct torch.device.
    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")

# Image handling classes.
class PatchMaker(nn.Module):
    def __init__(self, patchsize, stride=None):
        super(PatchMaker, self).__init__()
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4, metric: str = 'L2') -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
            metric: Distance metric to use. One of 'L2', 'cosine', 'L1'.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.metric = metric
        self.search_index = None
        self.gpu_resources = faiss.StandardGpuResources() if on_gpu else None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                self.gpu_resources, 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.metric == 'L1':
            # L1 metric is not supported on GPU in faiss
            return faiss.IndexFlat(dimension, faiss.METRIC_L1)
        # L2 and cosine (cosine uses L2 on normalized features)
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                self.gpu_resources, dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    ## YTODO 2-3-1 fn NN.fit()
    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        if self.metric == 'cosine':
            features = features.copy()
            faiss.normalize_L2(features)
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
        if self.metric == 'cosine':
            query_features = query_features.copy()
            faiss.normalize_L2(query_features)

        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        if self.metric == 'cosine':
            index_features = index_features.copy()
            faiss.normalize_L2(index_features)
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


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):

    ## YTODO 2-2-1 fn merger.merge()
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class FeatureAggregator(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device):
        super(FeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NearestNeighbourScorer(object):
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
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        ## YTODO 2-2 Merger.merge (this is just reshaping to N x CHW)
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        ## YTODO 2-3 use NN.fit
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
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
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


def get_memory_size(device_type, memory_type="free", device_index=None):

    if device_type == "cpu":
        memories = psutil.virtual_memory()
        return getattr(memories, memory_type)
    elif device_type in ["gpu", "cuda"]:
        assert memory_type in ["free", "available"]
        free_gpu_memory = GPUtil.getGPUs()[device_index].memoryFree * 1024**2
        return free_gpu_memory
    else:
        raise ValueError("Invalid device type")


def get_max_len_feas(
    fea_dim,
    device="cpu",
    max_usage=0.25,
    memory_type="free",
    device_index=None,
    byte_per_element=4,
):
    usage_one_fea = fea_dim * byte_per_element

    memory = get_memory_size(device, memory_type=memory_type, device_index=device_index)
    max_used_memory = memory * max_usage

    max_num_feas = int(max_used_memory / usage_one_fea)

    return max_num_feas


def get_max_num_feas(
    fea_dim,
    device_type="cpu",
    max_usage=0.25,
    memory_type="available",
    device_index=None,
    byte_per_element=4,
):
    usage_one_fea = fea_dim * byte_per_element

    memory = get_memory_size(
        device_type, memory_type=memory_type, device_index=device_index
    )
    max_used_memory = memory * max_usage

    max_len_feas = int(max_used_memory / usage_one_fea)

    return max_len_feas


def get_max_num_feas_for_gram(
    fea_dim,
    device_type="cpu",
    max_usage=0.75,
    memory_type="available",
    device_index=None,
    byte_per_element: int = 4,
    num_operations: int = 1,
):

    available_memory = (
        get_memory_size(device_type, memory_type=memory_type, device_index=device_index)
        * max_usage
    )

    d = fea_dim
    a = available_memory / num_operations
    b = byte_per_element
    # b*2*d*x + b*x^2 \leq a; x = num_feas
    x = np.sqrt(d**2 + a / b) - d
    return int(x)


def get_max_num_feas_for_pairwise_dist_matrix(
    fea_dim,
    device_type="cpu",
    max_usage=0.75,
    memory_type="available",
    device_index=None,
    byte_per_element=4,
    num_support_features: Optional[int] = None,
):
    return get_max_num_feas_for_gram(
        fea_dim=fea_dim,
        device_type=device_type,
        max_usage=max_usage,
        memory_type=memory_type,
        device_index=device_index,
        byte_per_element=byte_per_element,
        num_operations=3,
    )


def auto_get_max_query_size_pdist(
    fea_dim,
    n_support=None,
    max_usage=0.5,
    device_type="cpu",
    device_index=None,
    memory_type="available",
    byte_per_element=4,
):

    available_memory = get_memory_size(
        device_type=device_type,
        device_index=device_index,
        memory_type=memory_type,
    )
    available_memory *= max_usage

    return _get_max_query_size_pdist(
        available_memory=available_memory,
        fea_dim=fea_dim,
        n_support=n_support,
        byte_per_element=byte_per_element,
    )


def _get_max_query_size_pdist(
    available_memory, fea_dim, n_support=None, byte_per_element=4
):

    pdist_metric = "l2"

    if n_support is None:
        return _get_max_sample_size_pdist_l2_to_itself(
            available_memory=available_memory,
            fea_dim=fea_dim,
            byte_per_element=byte_per_element,
        )

    return _get_max_query_size_pdist_l2(
        available_memory=available_memory,
        fea_dim=fea_dim,
        n_support=n_support,
        byte_per_element=byte_per_element,
    )


def _get_max_query_size_pdist_l2(
    available_memory, fea_dim, n_support, byte_per_element=4
):
    # byte_per_element = 4 corresponds to float32

    # X: m x d
    # Y: n x d
    # b: memory per element
    # required memory for l2 pdist: b(2md + 2nd + mn)
    # max m given available memory: b(2md + 2nd + mn) \leq a
    # that is, m \leq (a/b - 2nd) / (2d + n)

    a = available_memory
    n = n_support
    d = fea_dim
    b = byte_per_element

    m = (a / b - 2 * n * d) / (2 * d + n)
    m = int(m)
    return m


def _get_max_sample_size_pdist_l2_to_itself(
    available_memory, fea_dim, byte_per_element=4
):
    # X: m x d
    # Y: m x d
    # b: memory per element
    # required memory for l2 pdist: b(2md + 2md + mm)
    # max m given available memory: b(2md + 2md + mm) \leq a
    # that is, m \leq (a/b - 2nd) / (2d + n)
    a = available_memory
    d = fea_dim
    b = byte_per_element

    m = -2 * d + math.sqrt(a / b + 4 * d**2)
    m = int(m)

    return m

def chunk_indices(lst_length, n_chunks, shuffle=False):
    indices = np.arange(lst_length)
    
    if shuffle:
        np.random.shuffle(indices)
    
    chunk_size = lst_length // n_chunks
    remainder = lst_length % n_chunks
    
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(indices[start:end].tolist())
        start = end
    
    return chunks

