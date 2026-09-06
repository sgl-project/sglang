"""Deduplicate MLA/DSA host cache across attention-TP ranks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# These backends tolerate buffer-less host pools on non-source ranks.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def mla_dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    parallel = get_parallel()
    if is_dp_attention_enabled():
        return parallel.attn_tp_rank, parallel.attn_tp_size
    return parallel.tp_rank, parallel.tp_size


def mla_host_dedup_eligible(kv_cache, storage_backend: Optional[str]) -> bool:
    """Rank-independent gate. CUDA only; FP4 excluded (its per-rank scale
    buffer is not covered by the broadcast)."""
    return (
        isinstance(kv_cache, MLATokenToKVPool)
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


class MLAHostDedupBroadcaster:
    """Layerwise MLA/DSA broadcast over a dedicated NCCL group."""

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        group: torch.distributed.ProcessGroup,
        src_global_rank: int,
    ):
        self.device_pool = device_pool
        self.group = group
        self.src_global_rank = src_global_rank
        self.is_src = mla_dedup_rank_and_size()[0] == 0
        self.layer_num = device_pool.layer_num
        self.device = device_pool.device
        self.chunk_tokens = envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.get()
        if self.chunk_tokens <= 0:
            raise ValueError(
                "SGLANG_MLA_DEDUP_CHUNK_TOKENS must be positive, "
                f"got {self.chunk_tokens}."
            )
        self.kv_staging = torch.empty(
            self.layer_num * self.chunk_tokens * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype,
            device=self.device,
        )
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.chunk_tokens * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )
        logger.info(
            "MLA host-dedup broadcast chunk configured: base_tokens=%d, "
            "effective_layer_tokens=%d",
            self.chunk_tokens,
            self.layer_num * self.chunk_tokens,
        )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> MLAHostDedupBroadcaster:
        """Build and initialize the NCCL group before host-pool allocation."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        broadcaster = cls(device_pool, group, src_global_rank=group_ranks[0])
        broadcaster._warmup_group()
        return broadcaster

    def _warmup_group(self) -> None:
        """Initialize the NCCL communicator before serving."""
        warmup = self.kv_staging[:1]
        if self.is_src:
            warmup.zero_()
        torch.distributed.broadcast(warmup, src=self.src_global_rank, group=self.group)
        torch.cuda.synchronize(self.device)
        logger.info("MLA host-dedup NCCL broadcast group warmup completed")

    def prepare_broadcast(
        self, device_indices: torch.Tensor, load_stream
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare reusable KV/indexer indices for one layerwise load."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)

        page_idx = None
        if self.idx_bufs is not None:
            page_size = self.device_pool.page_size
            if page_size > 1:
                if indices.numel() % page_size != 0:
                    raise ValueError(
                        "DSA dedup broadcast expects page-aligned device indices: "
                        f"got {indices.numel()} indices for page_size={page_size}."
                    )
                # Preserve logical page order across rank-local allocations.
                page_idx = indices[::page_size] // page_size
            else:
                page_idx = indices
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
        return indices, page_idx

    def broadcast_loaded_layer(
        self,
        layer_id: int,
        prepared: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> None:
        """Broadcast one loaded KV layer and its optional DSA indexer layer."""
        indices, page_idx = prepared
        self._bcast_layer(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
            layer_id,
        )
        if self.idx_bufs is not None:
            assert page_idx is not None
            self._bcast_layer(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                layer_id,
            )

    def _bcast_layer(
        self,
        buf_list,
        staging,
        target,
        elem,
        layer_id: int,
    ) -> None:
        """Broadcast one layer in chunks using the shared staging buffer."""
        n = target.shape[0]
        rows_per_chunk = staging.numel() // elem
        assert rows_per_chunk > 0
        layer_buf = buf_list[layer_id]
        row_shape = layer_buf.shape[1:]

        for start in range(0, n, rows_per_chunk):
            cur = min(rows_per_chunk, n - start)
            idx = target[start : start + cur]
            chunk = staging[: cur * elem]
            chunk_rows = chunk.view(cur, *row_shape)
            if self.is_src:
                torch.index_select(layer_buf, 0, idx, out=chunk_rows)
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                layer_buf.index_copy_(0, idx, chunk_rows)

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


@dataclass
class MLAHostDedupContext:
    """All state owned by the optional MLA host-dedup path."""

    broadcaster: MLAHostDedupBroadcaster
    prefetch_hits_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    prefetch_completion_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    producer_stream: Optional[object] = None
    last_write_finish_event: Optional[object] = None

    @property
    def is_src(self) -> bool:
        return self.broadcaster.is_src

    @property
    def is_dummy_rank(self) -> bool:
        return not self.is_src

    def destroy(self) -> None:
        self.broadcaster.destroy()
        groups = (self.prefetch_hits_sync_groups or []) + (
            self.prefetch_completion_sync_groups or []
        )
        for group in groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_hits_sync_groups = None
        self.prefetch_completion_sync_groups = None


def maybe_create_mla_host_dedup_context(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = False,
) -> Optional[MLAHostDedupContext]:
    """Create dedup state before host allocation, or preserve the original path."""
    if not enabled:
        return None
    if not mla_host_dedup_eligible(kv_cache, storage_backend):
        return None
    if mla_dedup_rank_and_size()[1] <= 1:
        return None

    broadcaster = MLAHostDedupBroadcaster.build(kv_cache, tp_group, attn_tp_group)
    prefetch_hits_sync_groups = None
    prefetch_completion_sync_groups = None
    if storage_backend is not None:
        prefetch_hits_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
        prefetch_completion_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return MLAHostDedupContext(
        broadcaster,
        prefetch_hits_sync_groups,
        prefetch_completion_sync_groups,
    )


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Prebuild one set of HiCache storage synchronization groups."""
    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
        )
    return groups
