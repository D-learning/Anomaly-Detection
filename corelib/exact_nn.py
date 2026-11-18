"""Exact squared-L2 nearest neighbours on torch, without reduced precision."""
import numpy as np
import torch


class TorchExactNN:
    def __init__(self, device, metric='L2', query_chunk=512):
        if metric not in ('L2', 'cosine'):
            raise ValueError('Exact torch backend supports L2 or cosine')
        self.device = torch.device(device)
        self.metric = metric
        self.query_chunk = query_chunk
        self.search_index = None
        # Match the float32 arithmetic used by IndexFlatL2; do not use TF32.
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False

    def _tensor(self, array):
        value = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=self.device)
        if self.metric == 'cosine':
            value = torch.nn.functional.normalize(value, dim=1)
        return value

    def fit(self, features):
        self.search_index = self._tensor(features)

    @torch.no_grad()
    def run(self, n_nearest_neighbours, query_features, index_features=None):
        bank = self.search_index if index_features is None else self._tensor(index_features)
        if bank is None or not 1 <= n_nearest_neighbours <= len(bank):
            raise ValueError('Fit a nonempty bank and request a valid number of neighbours')
        queries = self._tensor(query_features)
        if not len(queries):
            return np.empty((0, n_nearest_neighbours), dtype=np.float32), np.empty((0, n_nearest_neighbours), dtype=np.int64)
        bank_norm = (bank*bank).sum(1)
        distances, indices = [], []
        for query in queries.split(self.query_chunk):
            squared = (query*query).sum(1, keepdim=True) + bank_norm[None, :] - 2*(query @ bank.T)
            squared.clamp_min_(0)
            values, selected = squared.topk(n_nearest_neighbours, dim=1, largest=False, sorted=True)
            distances.append(values.cpu().numpy())
            indices.append(selected.cpu().numpy())
        return np.concatenate(distances), np.concatenate(indices)

    def reset_index(self):
        self.search_index = None

    def save(self, filename):
        with open(filename, 'wb') as file:
            np.save(file, self.search_index.cpu().numpy(), allow_pickle=False)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.fit(np.load(file, allow_pickle=False))
