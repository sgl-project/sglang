"""Host-memory dedup for MLA/DSA HiCache across attention-TP ranks.

MLA KV is identical on every attn-TP rank, so only the src rank (attn-TP
rank 0) keeps a real host pool; the other ranks run allocator-only "dummy"
pools and receive loaded pages via an NCCL broadcast on the load stream.

Single source of truth for the dedup gating and the broadcast machinery —
every dedup decision elsewhere must derive from these helpers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# Backends that never touch the host KV buffer directly, so they tolerate
# the buffer-less dummy pools. RDMA/registered backends (mooncake/eic/simm/
# hf3fs/nixl/aibrix) pin or register the buffer — dedup stays off for them.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def mla_dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    if is_dp_attention_enabled():
        return get_attention_tp_rank(), get_attention_tp_size()
    return (
        get_tensor_model_parallel_rank(),
        get_tensor_model_parallel_world_size(),
    )


def mla_host_dedup_eligible(kv_cache, storage_backend: Optional[str]) -> bool:
    """Rank-independent gate. CUDA only; FP4 excluded (its per-rank scale
    buffer is not covered by the broadcast)."""
    return (
        isinstance(kv_cache, MLATokenToKVPool)
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


def is_mla_dedup_dummy_rank(kv_cache, storage_backend: Optional[str]) -> bool:
    """Whether this rank must construct an allocator-only (dummy) host pool."""
    if not mla_host_dedup_eligible(kv_cache, storage_backend):
        return False
    rank, size = mla_dedup_rank_and_size()
    return size > 1 and rank != 0


class MLAHostDedupBroadcaster:
    """Broadcasts host-loaded MLA KV (and DSA indexer) device pages from the
    src rank to its attn-TP peers, over a dedicated NCCL group, coalescing
    all layers of one token chunk into a reused staging buffer."""

    # Tokens (or DSA indexer pages) staged per broadcast chunk.
    CHUNK_TOKENS = 512

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
        self.kv_staging = torch.empty(
            self.layer_num * self.CHUNK_TOKENS * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype,
            device=self.device,
        )
        # DSA keeps a per-page indexer buffer that must be broadcast too.
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.CHUNK_TOKENS * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> MLAHostDedupBroadcaster:
        """Build the NCCL group (a world collective — all dedup participants
        must call in lockstep) and the staging buffers."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        return cls(device_pool, group, src_global_rank=group_ranks[0])

    def broadcast_loaded(self, device_indices: torch.Tensor, load_stream) -> None:
        """Broadcast loaded pages from the src rank. Must run with
        ``load_stream`` as the current stream."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)
        self._bcast_rows(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
        )
        if self.idx_bufs is not None:
            page_size = self.device_pool.page_size
            page_idx = (
                torch.unique(torch.div(indices, page_size, rounding_mode="floor"))
                if page_size > 1
                else indices
            )
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
            self._bcast_rows(self.idx_bufs, self.idx_staging, page_idx, self.idx_elem)

    def _bcast_rows(self, buf_list, staging, target, elem) -> None:
        """Chunked broadcast of one per-layer buffer set; ``target`` indexes
        dim 0 (token indices for KV, page indices for the DSA indexer)."""
        n = target.shape[0]
        for start in range(0, n, self.CHUNK_TOKENS):
            cur = min(self.CHUNK_TOKENS, n - start)
            idx = target[start : start + cur]
            chunk = staging[: self.layer_num * cur * elem]
            if self.is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    chunk[o : o + cur * elem].copy_(buf_list[layer_id][idx].reshape(-1))
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                for layer_id in range(self.layer_num):
                    o = layer_id * cur * elem
                    buf_list[layer_id][idx] = chunk[o : o + cur * elem].view(
                        buf_list[layer_id][idx].shape
                    )

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


def maybe_build_mla_broadcaster(
    device_pool,
    tp_group: torch.distributed.ProcessGroup,
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
) -> Optional[MLAHostDedupBroadcaster]:
    """None when dedup does not engage (gate fails or single attn-TP rank)."""
    if not mla_host_dedup_eligible(device_pool, storage_backend):
        return None
    if mla_dedup_rank_and_size()[1] <= 1:
        return None
    return MLAHostDedupBroadcaster.build(device_pool, tp_group, attn_tp_group)


@dataclass
class MLAHostDedupPrebuild:
    """Groups/buffers rendezvoused ahead of the slow host KV allocation."""

    broadcaster: MLAHostDedupBroadcaster
    # None without a storage backend, so a later runtime attach still builds
    # its gloo groups inline.
    prefetch_sync_groups: Optional[List[torch.distributed.ProcessGroup]]


def maybe_prebuild_mla_host_dedup(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
) -> Optional[MLAHostDedupPrebuild]:
    """Issue the controller's init-time world collectives BEFORE the host KV
    pool is allocated.

    The src rank can spend many minutes pinning host KV while the dummy
    ranks race ahead into create_custom_parallel_group (NCCL bcast group +
    gloo prefetch groups) and trip the 600s NCCL watchdog; prebuilding
    completes the rendezvouses in lockstep first. Returns None when dedup
    does not engage — same gating as the controller, so groups are never
    built on ranks that would ignore them.
    """
    broadcaster = maybe_build_mla_broadcaster(
        kv_cache, tp_group, attn_tp_group, storage_backend
    )
    if broadcaster is None:
        return None
    prefetch_sync_groups = None
    if storage_backend is not None:
        prefetch_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return MLAHostDedupPrebuild(broadcaster, prefetch_sync_groups)


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Same construction as HiCacheController._create_prefetch_sync_groups."""
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
