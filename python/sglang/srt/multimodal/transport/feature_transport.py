"""How preprocessed multimodal features reach the scheduler."""

from typing import List

import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.multimodal.transport.cuda_ipc import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
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

    def precompute_hashes_before_cpu_transfer(
        self, mm_items: List[MultimodalDataItem]
    ) -> None:
        """Hash features in place, then move them to the CPU unless kept on GPU."""
        for item in mm_items:
            item.set_pad_value()
            if not self.keep_mm_features_on_device:
                item.feature = move_feature_to_cpu(item.feature)
                item.precomputed_embeddings = move_feature_to_cpu(
                    item.precomputed_embeddings
                )

    def prepare_mm_items_for_transport(
        self, mm_items: List[MultimodalDataItem]
    ) -> List[MultimodalDataItem]:
        """Wrap final GPU features, rolling back every lease if one wrap fails."""
        if not self.use_cuda_ipc:
            return mm_items

        # Pool misses fall back to plain CPU tensors. The scheduler copies out
        # and releases each successful pool slice.
        updates = []
        try:
            for item in mm_items:
                fields = (
                    ("feature", item.feature),
                    ("precomputed_embeddings", item.precomputed_embeddings),
                )
                for field, tensor in fields:
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    wrapped = self._wrap_tensor_for_cuda_ipc(tensor)
                    setattr(item, field, wrapped)
                    updates.append((item, field, tensor, wrapped))
        except BaseException as error:
            rollback_errors = []
            for item, field, tensor, wrapped in reversed(updates):
                try:
                    if isinstance(wrapped, CudaIpcTensorTransportProxy):
                        self.cudaipc_mmfeature_pool.cancel_proxy(wrapped)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
                finally:
                    setattr(item, field, tensor)
            if rollback_errors:
                error.add_note(
                    f"{len(rollback_errors)} CUDA IPC rollback operation(s) also failed"
                )
                raise error from rollback_errors[0]
            raise
        return mm_items

    def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):
        """helper function to turn a tensor into a cuda-ipc tensor"""
        if not tensor.is_cuda:
            return tensor

        proxy = self.cudaipc_mmfeature_pool.wrap_tensor(
            tensor,
            use_pool_handle_cache=self.use_ipc_pool_handle_cache,
        )
        return proxy if proxy is not None else tensor.cpu()


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
