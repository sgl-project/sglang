"""Send CUDA override embeds to the scheduler through a CUDA IPC pool, not by value."""

import logging
from typing import Iterable, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.multimodal.transport.cuda_ipc import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    get_mm_feature_pool_size_per_worker,
)
from sglang.srt.multimodal.transport.memory_pool import get_mm_feature_consumer_count
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


class PositionalEmbedsTransport:
    def __init__(self, base_gpu_id: int, mm_processor) -> None:
        parallel = get_parallel()
        # By value otherwise: IPC needs one node, every pipeline stage would ack the
        # slot, and the msgpack IPC encoder cannot encode the proxy.
        self.enabled = (
            is_cuda()
            and parallel.nnodes == 1
            and parallel.pp_size == 1
            and envs.SGLANG_USE_PICKLE_IPC.get()
        )
        self._base_gpu_id = base_gpu_id
        # A multimodal model with --mm-feature-transport=cuda_ipc already has a pool.
        self._pool: Optional[MmItemMemoryPool] = getattr(
            mm_processor, "cudaipc_mmfeature_pool", None
        )
        self._use_pool_handle_cache = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()

    def _get_pool(self) -> MmItemMemoryPool:
        if self._pool is None:
            pool_size = get_mm_feature_pool_size_per_worker(
                MM_FEATURE_CACHE_SIZE, get_serving().tokenizer_worker_num
            )
            logger.info(
                "Creating a %.0f MiB CUDA IPC pool on GPU %d for GPU-resident "
                "embedding overrides (SGLANG_MM_FEATURE_CACHE_MB).",
                pool_size / (1024 * 1024),
                self._base_gpu_id,
            )
            self._pool = MmItemMemoryPool(
                pool_size,
                MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
                self._base_gpu_id,
                get_mm_feature_consumer_count(),
            )
        return self._pool

    def prepare_for_dispatch(
        self, tokenized_objs: Iterable
    ) -> List[CudaIpcTensorTransportProxy]:
        """Returns the proxies to cancel if dispatch fails; a full pool means by value."""
        if not self.enabled:
            return []
        proxies = []
        try:
            for tokenized_obj in tokenized_objs:
                overrides = tokenized_obj.positional_embed_overrides
                if overrides is None or not (
                    isinstance(overrides.embeds, torch.Tensor)
                    and overrides.embeds.is_cuda
                    and overrides.embeds.numel() > 0
                ):
                    continue
                proxy = self._get_pool().wrap_tensor(
                    overrides.embeds, use_pool_handle_cache=self._use_pool_handle_cache
                )
                if proxy is None:
                    continue
                proxies.append(proxy)
                # A new object: requests may share the caller's PositionalEmbeds.
                tokenized_obj.positional_embed_overrides = PositionalEmbeds(
                    embeds=proxy, positions=overrides.positions
                )
        except BaseException:
            self.cancel_for_dispatch(proxies)
            raise
        return proxies

    def cancel_for_dispatch(self, proxies: List[CudaIpcTensorTransportProxy]) -> None:
        for proxy in proxies:
            self._pool.cancel_proxy(proxy)
