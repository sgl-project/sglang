"""Deduplicate MLA/DSA host cache across attention-TP ranks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


# These backends tolerate buffer-less host pools on non-source ranks.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def _dedup_group_timeout() -> Optional[timedelta]:
    seconds = get_server_args().dist_timeout
    return None if seconds is None else timedelta(seconds=seconds)


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
    """Broadcast restored MLA/DSA layers over a dedicated NCCL group."""

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
        # Layers reuse one staging buffer. A chunk cannot contain more rows
        # than a layer's physical pool; indexer rows are pages, not tokens.
        chunk_rows = self.layer_num * self.chunk_tokens
        kv_rows = min(chunk_rows, max(buf.shape[0] for buf in device_pool.kv_buffer))
        kv_dtype = device_pool.kv_buffer[0].dtype
        kv_elements = kv_rows * device_pool.kv_cache_dim
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        self.packed_staging = None
        self._packed_plan_rows = -1
        self._packed_plan = ()
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            idx_rows = min(chunk_rows, max(buf.shape[0] for buf in self.idx_bufs))
            kv_bytes = kv_elements * device_pool.kv_buffer[0].element_size()
            idx_item_bytes = self.idx_bufs[0].element_size()
            idx_offset = math.ceil(kv_bytes / idx_item_bytes) * idx_item_bytes
            # Typed views share one allocation. A restore can pack its KV and
            # indexer directly into it without concatenation or an extra copy.
            self.packed_staging = torch.empty(
                idx_offset + idx_rows * self.idx_elem * idx_item_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            self.kv_staging = self.packed_staging[:kv_bytes].view(kv_dtype)
            self.idx_staging = self.packed_staging[idx_offset:].view(
                self.idx_bufs[0].dtype
            )
        else:
            self.kv_staging = torch.empty(
                kv_elements, dtype=kv_dtype, device=self.device
            )
        logger.info(
            "MLA host-dedup broadcast chunk configured: base_tokens=%d, "
            "kv_staging_rows=%d",
            self.chunk_tokens,
            kv_rows,
        )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
    ) -> MLAHostDedupBroadcaster:
        """Build and initialize the NCCL group before host-pool allocation."""
        from sglang.srt.distributed.parallel_state import (
            create_custom_parallel_group,
            get_world_group,
        )

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks),
            backend="nccl",
            metadata_group=get_world_group().cpu_group,
            timeout=_dedup_group_timeout(),
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
        """Prepare reusable KV/indexer indices for one host restore."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        # Broadcast reads/writes raw buffers, just like physical L2 copies.
        translate = getattr(self.device_pool, "host_transfer_translate", None)
        if translate is not None:
            indices = translate(indices)
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
        has_indexer = self.idx_bufs is not None and self.idx_bufs[layer_id].shape[0] > 0
        if has_indexer and getattr(self, "packed_staging", None) is not None:
            assert page_idx is not None
            if self._bcast_packed_dsa_layer(layer_id, indices, page_idx):
                return
        self._bcast_layer(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
            layer_id,
        )
        # Shared-topk layers keep only a zero-row indexer placeholder.
        if has_indexer:
            assert page_idx is not None
            self._bcast_layer(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                layer_id,
            )

    def _bcast_packed_dsa_layer(self, layer_id, indices, page_idx) -> bool:
        """Send KV and indexer pages in one collective per aligned chunk."""
        pool = self.device_pool
        page_size = pool.page_size
        rows_per_chunk = min(
            self.kv_staging.numel() // pool.kv_cache_dim,
            self.idx_staging.numel() // self.idx_elem * page_size,
        )
        rows_per_chunk = rows_per_chunk // page_size * page_size
        if rows_per_chunk == 0:
            # A small configured chunk may not hold an entire indexer page.
            return False
        n = indices.numel()
        if self._packed_plan_rows != n:
            kv_buf = pool.kv_buffer[layer_id]
            idx_buf = self.idx_bufs[layer_id]
            kv_row_bytes = pool.kv_cache_dim * kv_buf.element_size()
            idx_item_bytes = idx_buf.element_size()
            plan = []
            for start in range(0, n, rows_per_chunk):
                cur = min(rows_per_chunk, n - start)
                pages = cur // page_size
                kv_bytes = cur * kv_row_bytes
                idx_offset = math.ceil(kv_bytes / idx_item_bytes) * idx_item_bytes
                payload = self.packed_staging[
                    : idx_offset + pages * self.idx_elem * idx_item_bytes
                ]
                kv_chunk = (
                    payload[:kv_bytes].view(kv_buf.dtype).view(cur, *kv_buf.shape[1:])
                )
                idx_chunk = (
                    payload[idx_offset:]
                    .view(idx_buf.dtype)
                    .view(pages, *idx_buf.shape[1:])
                )
                plan.append((start, cur, payload, kv_chunk, idx_chunk))
            # Only retain the current shape; varied request sizes must not
            # grow a cache of tensor views indefinitely.
            self._packed_plan = tuple(plan)
            self._packed_plan_rows = n
        kv_buf = pool.kv_buffer[layer_id]
        idx_buf = self.idx_bufs[layer_id]
        for start, cur, payload, kv_chunk, idx_chunk in self._packed_plan:
            kv_indices = indices[start : start + cur]
            idx_indices = page_idx[start // page_size : (start + cur) // page_size]
            if self.is_src:
                torch.index_select(kv_buf, 0, kv_indices, out=kv_chunk)
                torch.index_select(idx_buf, 0, idx_indices, out=idx_chunk)
            torch.distributed.broadcast(
                payload, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                kv_buf.index_copy_(0, kv_indices, kv_chunk)
                idx_buf.index_copy_(0, idx_indices, idx_chunk)
        return True

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
    *,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
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
            tp_group, attn_cp_group, attn_tp_group, pp_group
        )
        prefetch_completion_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group, pp_group
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
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[torch.distributed.ProcessGroup]:
    """Prebuild one set of HiCache storage synchronization groups."""
    from sglang.srt.distributed.parallel_state import (
        create_custom_parallel_group,
        get_world_group,
    )

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    if pp_group is not None:
        base_groups.append(pp_group)
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(
                group_ranks=list(ranks),
                backend="gloo",
                metadata_group=get_world_group().cpu_group,
                timeout=_dedup_group_timeout(),
            )
        )
    return groups
