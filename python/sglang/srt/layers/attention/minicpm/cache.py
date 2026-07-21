from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.configs.mamba_utils import BaseLinearStateParams
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator


class MiniCPMCompressedCache:
    def __init__(self, pool, kernel_size: int, kernel_stride: int):
        self.pool = pool
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.allocated_lens = [
            [0] * pool._alloc_size,
            [0] * pool._alloc_size,
        ]
        self.allocator = None

    def _sparse_len(self, length: int, scale: int) -> int:
        kernel_size = self.kernel_size * scale
        if length < kernel_size:
            return 0
        return (length - kernel_size) // (self.kernel_stride * scale) + 1

    def _allocate_to_lengths(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        req_indices = req_pool_indices_cpu.tolist()
        seq_lens = seq_lens_cpu.tolist()
        tables = (
            self.pool.req_to_sparse_k1_token,
            self.pool.req_to_sparse_k2_token,
        )
        plans = []
        for level, (table, scale) in enumerate(zip(tables, (1, 4))):
            targets = {
                req_idx: self._sparse_len(seq_len, scale)
                for req_idx, seq_len in zip(req_indices, seq_lens)
            }
            rows = [
                (req_idx, self.allocated_lens[level][req_idx], target)
                for req_idx, target in targets.items()
                if target > self.allocated_lens[level][req_idx]
            ]
            plans.append((table, rows, sum(end - start for _, start, end in rows)))

        allocator = tree_cache.token_to_kv_pool_allocator
        if self.allocator is not None and self.allocator is not allocator:
            raise RuntimeError("MiniCPM compressed cache allocator changed")

        allocated = []
        try:
            for _, _, size in plans:
                allocated.append(
                    alloc_token_slots(tree_cache, size) if size > 0 else None
                )

            for (table, rows, _), locs in zip(plans, allocated):
                if locs is None:
                    continue
                offset = 0
                for req_idx, start, end in rows:
                    count = end - start
                    table[req_idx, start:end] = locs[offset : offset + count].to(
                        torch.int32
                    )
                    offset += count
            for level, (_, rows, _) in enumerate(plans):
                for req_idx, _, end in rows:
                    self.allocated_lens[level][req_idx] = end
        except Exception:
            for locs in allocated:
                if locs is not None:
                    allocator.free(locs)
            raise

        if any(locs is not None for locs in allocated):
            self.allocator = allocator

    def alloc_for_extend(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        self._allocate_to_lengths(tree_cache, req_pool_indices_cpu, seq_lens_cpu)

    def alloc_for_decode(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        token_per_req: int,
    ) -> None:
        self._allocate_to_lengths(
            tree_cache,
            req_pool_indices_cpu,
            seq_lens_cpu + token_per_req,
        )

    def free(self, req_pool_idx: int) -> None:
        allocated = []
        for table, lengths in zip(
            (
                self.pool.req_to_sparse_k1_token,
                self.pool.req_to_sparse_k2_token,
            ),
            self.allocated_lens,
        ):
            length = lengths[req_pool_idx]
            if length > 0:
                allocated.append(table[req_pool_idx, :length].clone())
                table[req_pool_idx, :length].zero_()
                lengths[req_pool_idx] = 0

        if allocated:
            assert self.allocator is not None
            self.allocator.free(torch.cat(allocated))

    def clear(self) -> None:
        self.pool.req_to_sparse_k1_token.zero_()
        self.pool.req_to_sparse_k2_token.zero_()
        for lengths in self.allocated_lens:
            lengths[:] = [0] * len(lengths)
        self.allocator = None


class _MiniCPMSparsePoolMixin:
    def _init_compressed_cache(
        self,
        *,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        kernel_size: int,
        kernel_stride: int,
    ) -> None:
        saver = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        with saver.region(GPU_MEMORY_TYPE_KV_CACHE):
            k1_size = (max_context_len - kernel_size) // kernel_stride + 1
            k2_size = (max_context_len - kernel_size * 4) // (kernel_stride * 4) + 1
            self.req_to_sparse_k1_token = torch.zeros(
                (self._alloc_size, k1_size), dtype=torch.int32, device=device
            )
            self.req_to_sparse_k2_token = torch.zeros(
                (self._alloc_size, k2_size), dtype=torch.int32, device=device
            )
        self.compressed_cache = MiniCPMCompressedCache(self, kernel_size, kernel_stride)

    def alloc_aux_for_extend(
        self,
        *,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        self.compressed_cache.alloc_for_extend(
            tree_cache, req_pool_indices_cpu, seq_lens_cpu
        )

    def alloc_aux_for_decode(
        self,
        *,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        token_per_req: int,
    ) -> None:
        self.compressed_cache.alloc_for_decode(
            tree_cache,
            req_pool_indices_cpu,
            seq_lens_cpu,
            token_per_req,
        )

    def free(self, req) -> None:
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.compressed_cache.free(req.req_pool_idx)
        super().free(req)

    def clear(self) -> None:
        super().clear()
        self.compressed_cache.clear()


class MiniCPMReqToTokenPool(_MiniCPMSparsePoolMixin, ReqToTokenPool):
    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        kernel_size: int,
        kernel_stride: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self._init_compressed_cache(
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
        )


class MiniCPMHybridReqToTokenPool(_MiniCPMSparsePoolMixin, HybridReqToTokenPool):
    def __init__(
        self,
        *,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        kernel_size: int,
        kernel_stride: int,
        cache_params: Optional[BaseLinearStateParams] = None,
        mamba_size: int = None,
        mamba_spec_state_size: int = None,
        enable_mamba_extra_buffer: bool = False,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        speculative_eagle_topk: Optional[int] = None,
        mamba_layer_ids: List[int] = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        mamba_envelope_layout: bool = False,
    ):
        if mamba_layer_ids is None and cache_params is not None:
            mamba_layer_ids = cache_params.layers
        super().__init__(
            size=size,
            mamba_size=mamba_size if mamba_size is not None else size,
            mamba_spec_state_size=(
                mamba_spec_state_size if mamba_spec_state_size is not None else 0
            ),
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            cache_params=cache_params,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            enable_mamba_extra_buffer_lazy=enable_mamba_extra_buffer_lazy,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_eagle_topk=speculative_eagle_topk,
            mamba_layer_ids=mamba_layer_ids or [],
            enable_overlap_schedule=enable_overlap_schedule,
            start_layer=start_layer,
            enable_linear_replayssm=enable_linear_replayssm,
            linear_replayssm_cache_len=linear_replayssm_cache_len,
            mamba_envelope_layout=mamba_envelope_layout,
        )
        self._init_compressed_cache(
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
        )


def create_req_to_token_pool(
    *,
    configurator: KVCacheConfigurator,
    size: int,
    max_context_len: int,
    enable_memory_saver: bool,
):
    config = configurator.model_config.hf_config
    server_args = configurator.server_args
    common = dict(
        size=size,
        max_context_len=max_context_len,
        device=configurator.device,
        enable_memory_saver=enable_memory_saver,
    )
    sparse = config.has_minicpm_sparse_attention and not server_args.minicpm_force_dense
    cache_params = config.mamba2_cache_params

    if cache_params is None:
        if not sparse:
            return ReqToTokenPool(**common)
        return MiniCPMReqToTokenPool(
            **common,
            kernel_size=config.sparse_kernel_size,
            kernel_stride=config.sparse_kernel_stride,
        )

    hybrid = dict(
        **common,
        cache_params=cache_params,
        mamba_size=server_args.max_mamba_cache_size,
        mamba_spec_state_size=size,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer(),
        enable_mamba_extra_buffer_lazy=server_args.enable_mamba_extra_buffer_lazy(),
        speculative_num_draft_tokens=server_args.max_speculative_num_draft_tokens,
        speculative_eagle_topk=server_args.speculative_eagle_topk,
        enable_linear_replayssm=server_args.enable_linear_replayssm,
        linear_replayssm_cache_len=server_args.linear_replayssm_cache_len,
        mamba_envelope_layout=server_args.enable_page_major_kv_layout,
        mamba_layer_ids=[
            layer_id
            for layer_id in cache_params.layers
            if configurator.layer_info.start_layer
            <= layer_id
            < configurator.layer_info.end_layer
        ],
        enable_overlap_schedule=not server_args.disable_overlap_schedule,
        start_layer=configurator.layer_info.start_layer,
    )
    if sparse:
        return MiniCPMHybridReqToTokenPool(
            **hybrid,
            kernel_size=config.sparse_kernel_size,
            kernel_stride=config.sparse_kernel_stride,
        )
    return HybridReqToTokenPool(**hybrid)
