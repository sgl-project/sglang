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
from sglang.srt.utils import envs, logger


def keeps_mm_features_on_device(mm_feature_transport: str) -> bool:
    """Whether a feature transport expects processor outputs to stay on GPU."""
    return mm_feature_transport in ("cuda_ipc", "cuda_vmm")


class MultimodalFeatureTransport:
    """Prepares a processor's features for the scheduler: on CPU or in GPU memory.

    `cuda_ipc` holds a CUDA IPC pool for the processor's lifetime.
    """

    def __init__(
        self, mm_feature_transport: str, *, base_gpu_id: int, skip_mm_pool: bool
    ):
        self.mm_feature_transport = (
            mm_feature_transport
            if mm_feature_transport in ("cpu", "cuda_ipc", "cuda_vmm")
            else "cpu"
        )
        self.use_cuda_ipc = self.mm_feature_transport == "cuda_ipc"
        self.keep_mm_features_on_device = keeps_mm_features_on_device(
            self.mm_feature_transport
        )
        self.use_ipc_pool_handle_cache = (
            self.use_cuda_ipc and envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()
        )
        self.cudaipc_mmfeature_pool = (
            _create_cuda_ipc_pool(base_gpu_id)
            if self.use_cuda_ipc and not skip_mm_pool
            else None
        )


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
