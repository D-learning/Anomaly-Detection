import math
from typing import Optional

import GPUtil
import numpy as np
import psutil


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
