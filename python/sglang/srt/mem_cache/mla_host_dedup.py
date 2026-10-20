"""Deduplicate MLA/DSA host cache across attention-TP ranks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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
    """Rank-independent CUDA gate; classic MLA FP4 has unhandled scale buffers.

    V4 uses explicit side-pool transfers, including its FP4 indexer scales.
    """
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    return (
        isinstance(kv_cache, (MLATokenToKVPool, DeepSeekV4TokenToKVPool))
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


@dataclass
class _PendingLayerwiseLoad:
    device_indices: torch.Tensor
    finish_event: object
    next_layer: int = 0
    prepared: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    ready_event: Optional[object] = None
    ready_stream: Optional[object] = None


class MLAHostDedupBroadcaster:
    """Layerwise MLA/DSA broadcast over a dedicated NCCL group."""

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        group: torch.distributed.ProcessGroup,
        src_global_rank: int,
        *,
        layerwise_group_size: int = 4,
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
        if layerwise_group_size <= 0:
            raise ValueError("MLA layer-wise broadcast group size must be positive")
        self.layerwise_group_size = 1
        self._group_staging = None
        self._group_plan_key = None
        self._group_plan = ()
        self._group_last_use_event = None
        self._group_last_use_stream = None
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
            if kv_rows >= device_pool.page_size:
                self.layerwise_group_size = layerwise_group_size
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
        """Prepare reusable KV/indexer indices for one layerwise load."""
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

    def prepare_layerwise_broadcast(self) -> None:
        """Allocate grouped staging before any asynchronous H2D submission."""
        if self.layerwise_group_size > 1 and self._group_staging is None:
            self._group_staging = torch.empty(
                self.packed_staging.numel() * self.layerwise_group_size,
                dtype=torch.uint8,
                device=self.device,
            )

    def broadcast_loaded_layers(self, first: int, stop: int, prepared) -> None:
        """Send a bounded group of ready DSA layers in one payload per chunk.

        The caller waits for the last layer's H2D event before entering. This
        path is used only by optional layer-wise DSA restores; ordinary dedup
        keeps its existing single-layer calls.
        """
        indices, page_idx = prepared
        assert self.idx_bufs is not None and page_idx is not None
        assert 0 <= first < stop <= self.layer_num
        assert stop - first <= self.layerwise_group_size
        pool = self.device_pool
        page_size = pool.page_size
        rows_per_chunk = min(
            self.kv_staging.numel() // pool.kv_cache_dim,
            self.idx_staging.numel() // self.idx_elem * page_size,
        )
        rows_per_chunk = rows_per_chunk // page_size * page_size
        assert rows_per_chunk > 0
        self.prepare_layerwise_broadcast()
        current_stream = None
        if torch.device(self.device).type == "cuda":
            current_stream = torch.cuda.current_stream()
            if (
                self._group_last_use_event is not None
                and current_stream != self._group_last_use_stream
            ):
                # The scratch allocation is shared by all load slots. A later
                # forward stream must not overwrite a still-broadcast payload.
                current_stream.wait_event(self._group_last_use_event)
        has_indexer = tuple(
            self.idx_bufs[layer].shape[0] > 0 for layer in range(first, stop)
        )
        key = (indices.numel(), has_indexer)
        if self._group_plan_key != key:
            plan = []
            kv_buf = pool.kv_buffer[first]
            idx_buf = self.idx_bufs[first]
            for start in range(0, indices.numel(), rows_per_chunk):
                cur = min(rows_per_chunk, indices.numel() - start)
                pages = cur // page_size
                kv_bytes = cur * pool.kv_cache_dim * kv_buf.element_size()
                idx_bytes = pages * self.idx_elem * idx_buf.element_size()
                offset = 0
                views = []
                for with_indexer in has_indexer:
                    kv_view = (
                        self._group_staging[offset : offset + kv_bytes]
                        .view(kv_buf.dtype)
                        .view(cur, *kv_buf.shape[1:])
                    )
                    offset += kv_bytes
                    idx_view = None
                    if with_indexer:
                        idx_view = (
                            self._group_staging[offset : offset + idx_bytes]
                            .view(idx_buf.dtype)
                            .view(pages, *idx_buf.shape[1:])
                        )
                        offset += idx_bytes
                    views.append((kv_view, idx_view))
                plan.append((start, cur, self._group_staging[:offset], tuple(views)))
            self._group_plan = tuple(plan)
            self._group_plan_key = key
        for start, cur, payload, views in self._group_plan:
            kv_indices = indices[start : start + cur]
            idx_indices = page_idx[start // page_size : (start + cur) // page_size]
            if self.is_src:
                for layer, (kv_view, idx_view) in zip(range(first, stop), views):
                    torch.index_select(
                        pool.kv_buffer[layer], 0, kv_indices, out=kv_view
                    )
                    if idx_view is not None:
                        torch.index_select(
                            self.idx_bufs[layer], 0, idx_indices, out=idx_view
                        )
            torch.distributed.broadcast(
                payload, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                for layer, (kv_view, idx_view) in zip(range(first, stop), views):
                    pool.kv_buffer[layer].index_copy_(0, kv_indices, kv_view)
                    if idx_view is not None:
                        self.idx_bufs[layer].index_copy_(0, idx_indices, idx_view)
        if current_stream is not None:
            self._group_last_use_event = torch.cuda.Event()
            self._group_last_use_event.record(current_stream)
            self._group_last_use_stream = current_stream

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


class V4HostDedupBroadcaster(MLAHostDedupBroadcaster):
    """Broadcast V4 side pools using their prepared physical transfer indices."""

    def __init__(self, device_pool, group, src_global_rank):
        self.device_pool = device_pool
        self.group = group
        self.src_global_rank = src_global_rank
        self.is_src = mla_dedup_rank_and_size()[0] == 0
        self.device = device_pool.device
        self.layer_num = device_pool.layer_num
        self.chunk_tokens = envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.get()
        if self.chunk_tokens <= 0:
            raise ValueError("SGLANG_MLA_DEDUP_CHUNK_TOKENS must be positive")
        self.kv_staging = torch.empty(1, dtype=torch.uint8, device=self.device)
        self._page_staging = {}

    def broadcast_transfer(self, transfer, layer_id):
        from sglang.srt.mem_cache.memory_pool_host import (
            DeepSeekV4PagedHostPool,
            DeepSeekV4StateHostPool,
        )

        pool = transfer.host_pool
        if not isinstance(pool, (DeepSeekV4PagedHostPool, DeepSeekV4StateHostPool)):
            return
        local_layer = (
            transfer.layer_mapper(layer_id)
            if transfer.layer_mapper is not None
            else layer_id
        )
        indices = transfer.device_indices
        if local_layer is None or indices is None or indices.numel() == 0:
            return
        if local_layer >= pool.layer_num:
            return
        is_state = isinstance(pool, DeepSeekV4StateHostPool)
        buffers = pool.device_page_views if is_state else pool.device_buffers
        page_size = pool.swa_page_size if is_state else pool.slot_page_size
        item_bytes = pool.state_page_bytes if is_state else pool.item_bytes
        buffer = buffers[local_layer].view(torch.uint8).reshape(-1, item_bytes)
        rows_per_chunk = min(
            max(1, math.ceil(self.chunk_tokens / page_size)), buffer.shape[0]
        )
        key = (rows_per_chunk, item_bytes)
        if key not in self._page_staging:
            self._page_staging[key] = torch.empty(
                key, dtype=torch.uint8, device=self.device
            )
        staging = self._page_staging[key]
        indices = indices.to(device=self.device, dtype=torch.int64)
        if not is_state and indices.numel() % page_size:
            if pool.page_aligned_only:
                raise ValueError("V4 dedup requires whole pages for this pool")
            # Reuse the C4 token codec; a full-page scatter here would overwrite
            # unrelated live tokens sharing the destination page.
            if page_size != 64 or item_bytes != 37440:
                raise ValueError("Unsupported partial V4 C4 page format")
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla

            buffer_ptr = torch.tensor(
                [buffer.data_ptr()], dtype=torch.uint64, device=self.device
            )
            staging_ptr = torch.tensor(
                [staging.data_ptr()], dtype=torch.uint64, device=self.device
            )
            chunk_tokens = rows_per_chunk * page_size
            for start in range(0, indices.numel(), chunk_tokens):
                selected = indices[start : start + chunk_tokens]
                packed = torch.arange(
                    selected.numel(), dtype=torch.int64, device=self.device
                )
                if self.is_src:
                    staging.zero_()
                    transfer_cache_dsv4_mla(buffer_ptr, staging_ptr, selected, packed)
                chunk = staging[: math.ceil(selected.numel() / page_size)]
                torch.distributed.broadcast(
                    chunk, src=self.src_global_rank, group=self.group
                )
                if not self.is_src:
                    transfer_cache_dsv4_mla(staging_ptr, buffer_ptr, packed, selected)
            return
        rows = pool._to_page_indices(indices)
        for start in range(0, rows.numel(), rows_per_chunk):
            selected = rows[start : start + rows_per_chunk]
            chunk = staging[: selected.numel()]
            if self.is_src:
                torch.index_select(buffer, 0, selected, out=chunk)
            torch.distributed.broadcast(
                chunk, src=self.src_global_rank, group=self.group
            )
            if not self.is_src:
                buffer.index_copy_(0, selected, chunk)


@dataclass
class MLAHostDedupContext:
    """All state owned by the optional MLA host-dedup path."""

    broadcaster: MLAHostDedupBroadcaster
    prefetch_hits_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    prefetch_completion_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    producer_stream: Optional[object] = None
    last_write_finish_event: Optional[object] = None
    layerwise_overlap: bool = False
    pending_loads: dict[int, _PendingLayerwiseLoad] = field(default_factory=dict)
    layerwise_finish_events: dict[int, object] = field(default_factory=dict)
    layerwise_finish_streams: dict[int, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prepare = getattr(self.broadcaster, "prepare_layerwise_broadcast", None)
        if self.layerwise_overlap and callable(prepare):
            prepare()

    @property
    def is_src(self) -> bool:
        return self.broadcaster.is_src

    @property
    def is_dummy_rank(self) -> bool:
        return not self.is_src

    def begin_layerwise_load(
        self, index: int, device_indices: torch.Tensor, finish_event: object
    ) -> None:
        if index in self.pending_loads:
            raise RuntimeError("MLA load slot reused before its broadcast completed")
        if index in self.layerwise_finish_events:
            # Even a completed event still belongs to an outstanding ACK.
            # Replacing it could make that ACK observe a later load's event.
            raise RuntimeError("MLA load slot reused before its acknowledgement")
        self.pending_loads[index] = _PendingLayerwiseLoad(
            device_indices=device_indices, finish_event=finish_event
        )
        self.layerwise_finish_events[index] = finish_event

    def has_pending_layerwise_load(self, index: int) -> bool:
        return index in self.pending_loads

    def is_layerwise_load_done(self, index: int) -> bool:
        if index in self.pending_loads:
            return False
        event = self.layerwise_finish_events.get(index)
        return event is None or event.query()

    def release_layerwise_load(self, index: int) -> None:
        if not self.is_layerwise_load_done(index):
            raise RuntimeError("Cannot release an unfinished MLA layer-wise load")
        self.layerwise_finish_events.pop(index, None)
        self.layerwise_finish_streams.pop(index, None)

    def reset_layerwise_loads(self) -> None:
        if self.pending_loads:
            raise RuntimeError("Cannot reset MLA dedup before its forward completes")
        for event in self.layerwise_finish_events.values():
            event.synchronize()
        self.layerwise_finish_events.clear()
        self.layerwise_finish_streams.clear()

    def broadcast_ready_layers(self, index: int, threshold: int) -> None:
        """Run broadcasts on the forward stream, after its H2D layer wait.

        The forward and dedup collectives then have one stream order on every
        TP rank. Only the layers reached by the model have to finish before
        their computation; later H2D copies may continue on the load stream.
        """
        load = self.pending_loads.get(index)
        if load is None:
            stream = self.layerwise_finish_streams.get(index)
            if stream is not None and torch.cuda.current_stream() != stream:
                torch.cuda.current_stream().wait_event(
                    self.layerwise_finish_events[index]
                )
            return
        broadcaster = self.broadcaster
        group_size = getattr(broadcaster, "layerwise_group_size", 1)
        grouped = isinstance(group_size, int) and group_size > 1
        current_stream = torch.cuda.current_stream() if grouped else None
        if (
            grouped
            and load.ready_event is not None
            and current_stream != load.ready_stream
        ):
            current_stream.wait_event(load.ready_event)
        if threshold < load.next_layer:
            return
        if load.prepared is None:
            from sglang.srt.utils import get_device_module

            load.prepared = broadcaster.prepare_broadcast(
                load.device_indices, get_device_module().current_stream()
            )
        elif grouped and current_stream != load.ready_stream:
            # Waiting for the previous group orders data access, but does not
            # keep its indices alive on a new stream after the load is removed.
            for tensor in load.prepared:
                if tensor is not None and tensor.is_cuda:
                    tensor.record_stream(current_stream)
        stop = min(threshold + 1, broadcaster.layer_num)
        if grouped:
            stop = min(
                ((threshold // group_size) + 1) * group_size, broadcaster.layer_num
            )
            # Copies are submitted on the load stream. Looking ahead must wait
            # for every layer included in the grouped payload, on each rank.
            broadcaster.device_pool.layer_transfer_counter.events[index].wait(stop - 1)
            for first in range(load.next_layer, stop, group_size):
                broadcaster.broadcast_loaded_layers(
                    first, min(first + group_size, stop), load.prepared
                )
            load.ready_event = getattr(broadcaster, "_group_last_use_event", None)
            if load.ready_event is None:
                load.ready_event = torch.cuda.Event()
                load.ready_event.record(current_stream)
            load.ready_stream = current_stream
        else:
            for layer_id in range(load.next_layer, stop):
                broadcaster.broadcast_loaded_layer(layer_id, load.prepared)
        load.next_layer = max(load.next_layer, stop)
        if load.next_layer >= broadcaster.layer_num:
            load.finish_event.record()
            if grouped:
                self.layerwise_finish_streams[index] = current_stream
            del self.pending_loads[index]

    def destroy(self) -> None:
        self.reset_layerwise_loads()
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
    layerwise_overlap: bool = False,
) -> Optional[MLAHostDedupContext]:
    """Create dedup state before host allocation, or preserve the original path."""
    if not enabled:
        return None
    if layerwise_overlap and not isinstance(kv_cache, MLATokenToKVPool):
        raise ValueError(
            "MLA layer-wise overlap currently supports flat MLA/DSA pools only."
        )

    if not mla_host_dedup_eligible(kv_cache, storage_backend):
        return None
    if mla_dedup_rank_and_size()[1] <= 1:
        return None

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    broadcaster_cls = (
        V4HostDedupBroadcaster
        if isinstance(kv_cache, DeepSeekV4TokenToKVPool)
        else MLAHostDedupBroadcaster
    )
    broadcaster = broadcaster_cls.build(kv_cache, tp_group, attn_tp_group)
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
        layerwise_overlap=layerwise_overlap,
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
