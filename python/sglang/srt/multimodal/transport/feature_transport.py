"""How preprocessed multimodal features reach the scheduler."""

import torch

from sglang.srt.multimodal.transport.cuda_ipc import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    MmItemMemoryPool,
    get_mm_feature_pool_size_per_worker,
)
from sglang.srt.multimodal.transport.memory_pool import get_mm_feature_consumer_count
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils import logger


def _create_cuda_ipc_pool(base_gpu_id: int) -> MmItemMemoryPool:
    # SGLANG_MM_FEATURE_CACHE_MB is the total pool budget across all
    # tokenizer workers. Each worker gets an equal share so that adding
    # workers doesn't multiply the GPU-side footprint.
    worker_num = get_serving().tokenizer_worker_num
    per_worker_pool_size = get_mm_feature_pool_size_per_worker(
        MM_FEATURE_CACHE_SIZE, worker_num
    )
    total_pool_size = per_worker_pool_size * worker_num
    logger.info(
        "CUDA IPC multimodal feature pools reserve %.0f MiB total on "
        "GPU %d (%.0f MiB per tokenizer worker × %d; configured "
        "budget %.0f MiB).",
        total_pool_size / (1024 * 1024),
        base_gpu_id,
        per_worker_pool_size / (1024 * 1024),
        worker_num,
        MM_FEATURE_CACHE_SIZE / (1024 * 1024),
    )
    cudaipc_mmfeature_pool = MmItemMemoryPool(
        per_worker_pool_size,
        MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
        base_gpu_id,
        get_mm_feature_consumer_count(),
    )
    return cudaipc_mmfeature_pool


def move_feature_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, list):
        return [move_feature_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move_feature_to_cpu(v) for v in value)
    return value
