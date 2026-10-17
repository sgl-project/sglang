"""Direct linker that keeps pages in pinned DRAM on the source rank.

TODO: spill to local SSD; an SSD hit needs a DRAM prefetch before admission.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import NamedTuple, Optional

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.host_memory import available_host_memory_bytes
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.pool_host.base import HICACHE_HOST_MEMORY_RESERVE_BYTES
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    _cuda_host_unregister,
    alloc_with_host_register,
    make_kernel_ptr_table,
)
from sglang.srt.mem_cache.storage.local_store.local_page_index import (
    LocalPageIndex,
    PageEntry,
    Reservation,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import LinkerLoadBroadcast
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
    apply_hit_policy,
)
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import get_device_module, is_cuda, is_hip

if is_cuda() or is_hip():
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_per_layer_mla_pf_lf,
    )

logger = logging.getLogger(__name__)
device_module = get_device_module()


class StreamLayerLoadCounter:
    """Layer wait hook backed by per-layer CUDA events of the load stream."""

    def __init__(self, num_layers: int, on_layer_ready=None):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        # index -> (per-layer events or None, landed)
        self.batches: dict[int, tuple[Optional[list], bool]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        return self.producer_index

    def start(self, index: int, events: Optional[list], landed: bool) -> None:
        self.batches[index] = (events, landed)

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def wait_until(self, threshold: int) -> None:
        batch = self.batches.get(self.consumer_index)
        if batch is None:
            return
        events, _ = batch
        if events is not None:
            device_module.current_stream().wait_event(events[threshold])
        if self.on_layer_ready is not None:
            self.on_layer_ready(self.consumer_index, threshold)

    def finish(self, index: int) -> bool:
        events, landed = self.batches.pop(index, (None, True))
        if events is not None:
            events[-1].synchronize()
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.batches.clear()


class _PoolArena:
    """Pinned host copy of one device pool: [slot, buffer, page bytes]."""

    def __init__(self, entry: DevicePoolEntry):
        self.entry = entry
        self.buffers = [
            buffer for component in entry.components for buffer in component
        ]
        sizes = {size for component in entry.buffer_meta for _, _, size in component}
        if len(sizes) != 1:
            raise ValueError(
                f"Local linker pool {entry.name} needs one page size per buffer, "
                f"got {sorted(sizes)}."
            )
        self.item_size = sizes.pop()
        row_strides = {
            row_stride
            for component in entry.buffer_meta
            for _, row_stride, _ in component
        }
        if len(row_strides) != 1 or self.item_size % next(iter(row_strides)):
            raise ValueError(
                f"Local linker pool {entry.name} needs packed rows of one stride."
            )
        # Device rows per page: 1 for page rows, page_size for token rows.
        self.row_span = self.item_size // row_strides.pop()
        self.page_bytes = self.item_size * len(self.buffers)
        buffers_per_component = len(entry.components[0])
        self.layer_buffers: dict[int, list[int]] = {}
        for layer, mapped in entry.layer_mapping.items():
            indices = [mapped] if isinstance(mapped, int) else list(mapped)
            self.layer_buffers[layer] = [
                component * buffers_per_component + index
                for component in range(len(entry.components))
                for index in indices
            ]
        self.buffer_table = make_kernel_ptr_table(
            self.buffers, self.buffers[0].device, host_memory_registered=False
        )
        self.host = None

    def allocate(self, num_slots: int) -> None:
        self.host = alloc_with_host_register(
            (num_slots, self.page_bytes),
            torch.uint8,
            "cpu",
            True,
            HostTensorAllocator(),
        )

    def release(self) -> None:
        if self.host is not None:
            _cuda_host_unregister(self.host)
            self.host = None

    def device_pages(self, indices: torch.Tensor) -> list[int]:
        return [row // self.row_span for row in self.entry.prepare_locations(indices)]


class _PoolCopy(NamedTuple):
    arena: _PoolArena
    slots: torch.Tensor
    pages: torch.Tensor


class _Offload(NamedTuple):
    done: Optional[object]
    reservation: Optional[Reservation]
    success: bool
    keepalive: tuple


class LocalDirectLinker(UnifiedCacheLinker):
    """Peers receive loads by broadcast and answer from the transfers alone, so
    the source's verdicts reach them through the tree's MIN reductions."""

    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.num_layers = self.pool_group.num_layers

        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        tp_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_size = torch.distributed.get_world_size(group=tp_group)
        self.load_broadcast = None
        if tp_size > 1:
            if not self.pool_group.rank_replicated:
                raise ValueError(
                    "The local linker stores on one rank and needs rank-replicated "
                    "pools when the attention TP size is above 1."
                )
            self.load_broadcast = LinkerLoadBroadcast.build(
                self.pool_group,
                params.tp_cache_group,
                params.attn_tp_cache_group,
                tp_group,
            )
        self.is_source = self.load_broadcast is None or self.load_broadcast.is_src
        self.layer_done_counter = StreamLayerLoadCounter(
            self.num_layers,
            self.load_broadcast.broadcast_until if self.load_broadcast else None,
        )

        self.pending_loads: dict[str, tuple[list[PoolTransfer], list[PageEntry]]] = {}
        self.inflight_loads: dict[int, tuple[list[PageEntry], tuple]] = {}
        self.offloads: deque[_Offload] = deque()
        self.index = None
        self.arenas: dict[PoolName, _PoolArena] = {}
        if self.is_source:
            self._allocate_store()

    def _allocate_store(self) -> None:
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        capacity_gb = extra_config.get("capacity_gb")
        if capacity_gb is None or float(capacity_gb) <= 0:
            raise ValueError(
                "The local linker needs a positive capacity_gb in "
                "--hicache-storage-backend-extra-config."
            )
        capacity = int(float(capacity_gb) * (1024**3))
        arenas = {entry.name: _PoolArena(entry) for entry in self.pool_group.entries}
        page_bytes = sum(arena.page_bytes for arena in arenas.values())
        num_slots = capacity // page_bytes
        if num_slots == 0:
            raise ValueError(
                f"capacity_gb={capacity_gb} holds no page of {page_bytes} bytes."
            )
        available = available_host_memory_bytes() - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if num_slots * page_bytes > available:
            raise ValueError(
                f"The local linker needs {num_slots * page_bytes / 1e9:.2f} GB of "
                f"host memory but only {available / 1e9:.2f} GB is free."
            )
        try:
            for arena in arenas.values():
                arena.allocate(num_slots)
        except BaseException:
            for arena in arenas.values():
                arena.release()
            raise
        self.arenas = arenas
        self.index = LocalPageIndex(num_slots)
        self.load_stream = device_module.Stream()
        self.offload_stream = device_module.Stream()
        logger.info(
            "Local linker store: %d pages of %d bytes (%.2f GB) across pools %s",
            num_slots,
            page_bytes,
            num_slots * page_bytes / 1e9,
            [name.value for name in arenas],
        )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        valid_pages = list(range(1, len(page_keys) + 1))
        if not self.is_source:
            return valid_pages
        for transfer in expanded:
            page_exists = self.index.page_exists(
                page_keys[: valid_pages[-1]], transfer.name
            )
            valid_pages = apply_hit_policy(valid_pages, page_exists, transfer)
            if not valid_pages:
                break
        self.index.pin_lookup(rid, page_keys[: valid_pages[-1]] if valid_pages else [])
        return valid_pages

    def release_lookup(self, rid: str) -> None:
        if self.index is not None:
            self.index.release_lookup(rid)

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Local linker load for rid={rid} is already queued.")
        pins = self.index.take_lookup_pins(rid) if self.index is not None else []
        self.pending_loads[rid] = (expanded, pins)
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        pending = self.pending_loads.pop(rid, None)
        if pending is None:
            return False
        if self.index is not None:
            self.index.unpin(pending[1])
        return True

    def start_layer_wise_loading(self) -> int:
        transfers = {rid: expanded for rid, (expanded, _) in self.pending_loads.items()}
        broadcast = self.load_broadcast
        plans_agree = broadcast is None or broadcast.plans_agree(transfers)
        if not self.pending_loads:
            return -1
        pins = [
            entry for _, entries in self.pending_loads.values() for entry in entries
        ]
        self.pending_loads = {}
        counter_index = self.layer_done_counter.update_producer()
        if not plans_agree:
            logger.error("Local linker load plans differ across ranks.")
            self._start_batch(counter_index, None, False, pins, ())
            return counter_index
        if broadcast is not None:
            broadcast.begin(counter_index, transfers)
        if not self.is_source:
            self._start_batch(counter_index, None, True, pins, ())
            return counter_index

        copies = self._plan_copies(
            [transfer for expanded in transfers.values() for transfer in expanded],
            self.index.slot_of,
        )
        if copies is None:
            logger.error("Local linker lost a pinned page; failing the load batch.")
            self._start_batch(counter_index, None, False, pins, ())
            return counter_index
        events = self._copy_layers_in(copies)
        self._start_batch(counter_index, events, True, pins, tuple(copies))
        return counter_index

    def _start_batch(self, counter_index, events, landed, pins, keepalive) -> None:
        self.layer_done_counter.start(counter_index, events, landed)
        self.inflight_loads[counter_index] = (pins, keepalive)

    def _plan_copies(self, transfers: list[PoolTransfer], slot_of) -> Optional[list]:
        by_pool: dict[PoolName, tuple[list[int], list[int]]] = {}
        for transfer in transfers:
            arena = self.arenas[transfer.name]
            slots, pages = by_pool.setdefault(transfer.name, ([], []))
            for key in transfer.keys:
                slot = slot_of(key, transfer.name)
                if slot is None:
                    return None
                slots.append(slot)
            pages.extend(arena.device_pages(transfer.host_indices))
        return [
            self._pool_copy(self.arenas[name], slots, pages)
            for name, (slots, pages) in by_pool.items()
        ]

    def _pool_copy(self, arena: _PoolArena, slots, pages) -> _PoolCopy:
        def to_device(values):
            tensor = torch.tensor(values, dtype=torch.int64, pin_memory=True)
            return tensor.to(arena.buffer_table.device, non_blocking=True)

        return _PoolCopy(arena, to_device(slots), to_device(pages))

    def _copy_layers_in(self, copies: list[_PoolCopy]) -> list:
        ready_event = device_module.Event()
        ready_event.record()
        self.load_stream.wait_event(ready_event)
        events = []
        with device_module.stream(self.load_stream):
            for layer in range(self.num_layers):
                for arena, slots, pages in copies:
                    for buffer_index in arena.layer_buffers.get(layer, ()):
                        transfer_kv_per_layer_mla_pf_lf(
                            arena.host,
                            arena.buffers[buffer_index],
                            slots,
                            pages,
                            buffer_index,
                            arena.item_size,
                            arena.page_bytes,
                        )
                event = device_module.Event()
                event.record()
                events.append(event)
        return events

    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        landed = self.layer_done_counter.finish(counter_index)
        pins, _ = self.inflight_loads.pop(counter_index, ([], ()))
        if self.index is not None:
            self.index.unpin(pins)
        return landed

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        if not self.is_source:
            self.offloads.append(_Offload(None, None, True, ()))
            return True
        reservation = self.index.reserve(
            [(transfer.name, transfer.keys) for transfer in expanded]
        )
        if reservation is None:
            # A full store fails through the completion, which ranks reduce.
            self.offloads.append(_Offload(None, None, False, ()))
            return True
        done, keepalive = self._copy_out(expanded, reservation)
        self.offloads.append(_Offload(done, reservation, True, keepalive))
        return True

    def _copy_out(self, expanded: list[PoolTransfer], reservation: Reservation):
        copies = []
        for transfer, (_, positions) in zip(expanded, reservation):
            if not positions:
                continue
            arena = self.arenas[transfer.name]
            pages = arena.device_pages(transfer.host_indices)
            copies.append(
                self._pool_copy(
                    arena,
                    [entry.slot for _, entry in positions],
                    [pages[position] for position, _ in positions],
                )
            )
        if not copies:
            return None, ()
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_stream.wait_event(ready_event)
        with device_module.stream(self.offload_stream):
            for arena, slots, pages in copies:
                transfer_kv_all_layer_mla_lf_pf(
                    arena.buffer_table,
                    arena.host,
                    pages,
                    slots,
                    arena.item_size,
                    arena.page_bytes,
                    len(arena.buffers),
                )
            done = device_module.Event()
            done.record()
        return done, tuple(copies)

    def num_completed_offloads(self) -> int:
        count = 0
        for offload in self.offloads:
            if offload.done is not None and not offload.done.query():
                break
            count += 1
        return count

    def pop_completed_offload(self) -> bool:
        offload = self.offloads.popleft()
        if offload.reservation is not None:
            self.index.publish(offload.reservation)
        return offload.success

    def reset(self) -> None:
        if self.is_source:
            self.load_stream.synchronize()
            self.offload_stream.synchronize()
            self.index.clear_pins()
        self.pending_loads.clear()
        self.inflight_loads.clear()
        self.offloads.clear()
        self.layer_done_counter.reset()
        if self.load_broadcast is not None:
            self.load_broadcast.reset()

    def close(self) -> None:
        self.reset()
        for arena in self.arenas.values():
            arena.release()
        self.arenas = {}
        self.index = None
        if self.load_broadcast is not None:
            self.load_broadcast.destroy()
