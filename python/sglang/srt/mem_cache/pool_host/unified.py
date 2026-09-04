from __future__ import annotations

import threading
from collections.abc import Sequence
from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_direct, transfer_kv_per_layer_mla


class _SharedPageEnvelopeHostBacking:
    class _LayoutLeaseState(threading.local):
        def __init__(self):
            self.depth = 0

    class _Side:
        def __init__(
            self,
            *,
            page_size: int,
            page_bytes: int,
            page_num: int,
            grow_direction: str,
        ):
            if grow_direction not in {"up", "down"}:
                raise ValueError(
                    f"Unsupported unified host-pool grow direction: {grow_direction}"
                )
            self.page_size = page_size
            self.page_bytes = page_bytes
            self.page_num = page_num
            self.grow_direction = grow_direction

            # The cache tree owns logical page ids. Physical page ids are private
            # to the backing and may change whenever the two sides are compacted.
            self.free_logical_extents = [(0, page_num)]
            self.logical_to_physical: dict[int, int] = {}
            self.physical_to_logical: dict[int, int] = {}
            self.free_physical_pages: set[int] = set()
            self.watermark = 0 if grow_direction == "up" else page_num - 1

        @property
        def live_page_count(self) -> int:
            return len(self.logical_to_physical)

    def __init__(
        self,
        *,
        device_pools: Sequence,
        pool_names: Sequence[str],
        host_to_device_ratio: float,
        host_size: float,
        pin_memory: bool,
        device: str,
        allocator_type: str,
    ):
        if len(device_pools) != len(pool_names) or not device_pools:
            raise ValueError("Unified host pool devices and names must be non-empty.")
        if len(device_pools) > 2:
            raise ValueError("Unified host backing supports at most two pools.")
        if len(set(pool_names)) != len(pool_names):
            raise ValueError(f"Unified host pool names must be unique: {pool_names}.")

        device_buffers = [pool.get_page_envelope_buffer() for pool in device_pools]
        registrations = [pool.get_contiguous_buf_infos() for pool in device_pools]
        for ptrs, lens, item_lens in registrations:
            if len(ptrs) != len(lens) or len(ptrs) != len(item_lens) or len(ptrs) != 1:
                raise ValueError(
                    "Unified page-envelope pools must expose one memory region."
                )
        shared_regions = {(ptrs[0], lens[0]) for ptrs, lens, _ in registrations}
        if len(shared_regions) != 1:
            raise ValueError("Unified full and SWA pools must share one device region.")

        page_bytes = [int(buffer.shape[1]) for buffer in device_buffers]
        if host_size > 0:
            token_capacities = [
                int(host_size * 1e9 * pool.page_size // item_bytes)
                for pool, item_bytes in zip(device_pools, page_bytes, strict=True)
            ]
        else:
            token_capacities = [
                int(pool.size * host_to_device_ratio) for pool in device_pools
            ]

        nominal_view_bytes = [
            (capacity // pool.page_size + 1) * item_bytes
            for pool, item_bytes, capacity in zip(
                device_pools, page_bytes, token_capacities, strict=True
            )
        ]

        if host_size > 0:
            total_bytes = max(nominal_view_bytes)
        else:
            device_total_bytes = next(iter(shared_regions))[1]
            total_bytes = max(
                int(device_total_bytes * host_to_device_ratio) + max(page_bytes),
                *nominal_view_bytes,
            )
        if total_bytes <= 0:
            raise ValueError(
                f"Unified host backing size must be positive: {total_bytes}"
            )

        available_bytes = host_memory_budget_bytes()
        if total_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory for unified page-envelope backing. "
                f"Requesting {total_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        grow_directions = [pool.grow_direction for pool in device_pools]
        if len(grow_directions) == 2 and set(grow_directions) != {"up", "down"}:
            raise ValueError(
                "Unified full and SWA host pools must grow from opposite ends; "
                f"got {grow_directions}."
            )
        self.sides = {}
        for pool, name, item_bytes, grow_direction in zip(
            device_pools,
            pool_names,
            page_bytes,
            grow_directions,
            strict=True,
        ):
            self.sides[name] = self._Side(
                page_size=pool.page_size,
                page_bytes=item_bytes,
                page_num=total_bytes // item_bytes,
                grow_direction=grow_direction,
            )
        if any(side.page_num == 0 for side in self.sides.values()):
            raise ValueError(
                "Unified host backing must fit at least one page from every pool."
            )

        down_sides = [
            side for side in self.sides.values() if side.grow_direction == "down"
        ]
        self.capacity_bytes = total_bytes
        if down_sides:
            # Match L1's page-indexed view: bytes after the grow-down side's final
            # complete page are tail padding, not shared allocatable capacity.
            self._allocatable_bytes = down_sides[0].page_num * down_sides[0].page_bytes
        else:
            only_side = next(iter(self.sides.values()))
            self._allocatable_bytes = only_side.page_num * only_side.page_bytes

        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.lock = threading.RLock()
        alloc_func = ALLOC_MEMORY_FUNCS[device_buffers[0].device]
        self.raw = alloc_func(
            (total_bytes,),
            dtype=torch.uint8,
            device=device,
            pin_memory=pin_memory,
            allocator=self.allocator,
        )
        self.fd = getattr(self.allocator, "fd", None)
        self.registration_owner = pool_names[0]
        self._users: set[str] = set()
        self._destroyed = False

        # A layout lease pins logical->physical mappings. CUDA transfers hand
        # the lease off to their finish event; synchronous L3 I/O holds it until
        # the backend returns. Compaction blocks new leases and drains both.
        self._layout_condition = threading.Condition(self.lock)
        self._layout_users = 0
        self._layout_lease_state = self._LayoutLeaseState()
        self._compacting = False
        self._pending_transfer_events = {}

    def retain(self, name: str) -> None:
        if name in self._users:
            raise RuntimeError(f"Unified host pool {name!r} is already attached.")
        self._users.add(name)

    def release(self, name: str) -> None:
        with self.lock:
            self._users.discard(name)
            self.destroy_if_unused()

    def destroy_if_unused(self) -> None:
        with self.lock:
            if self._users or self._destroyed:
                return
            self._destroyed = True
            events = self._begin_compaction()
            try:
                for event in events:
                    event.synchronize()
                if self.pin_memory and (_is_cuda or _is_hip):
                    _cuda_host_unregister(self.raw)
                self.raw = None
            finally:
                self._finish_compaction()

    def page_buffer(self, name: str) -> torch.Tensor:
        side = self.sides[name]
        return self.raw[: side.page_num * side.page_bytes].view(
            side.page_num, side.page_bytes
        )

    @staticmethod
    def _reserve_logical_pages(
        extents: list[tuple[int, int]], page_count: int
    ) -> list[int] | None:
        if page_count == 0:
            return []
        remaining = page_count
        pages = []
        new_extents = []
        for start, end in extents:
            capacity = end - start
            take = min(remaining, capacity)
            pages.extend(range(start, start + take))
            if start + take < end:
                new_extents.append((start + take, end))
            remaining -= take
        if remaining == 0:
            extents[:] = new_extents
            return pages

        return None

    def _request_page_counts(
        self, requests: Sequence[tuple[str, int]]
    ) -> tuple[list[int], dict[str, int]]:
        request_page_counts = []
        page_counts = {name: 0 for name in self.sides}
        for name, need_size in requests:
            if name not in self.sides:
                raise ValueError(f"Unknown unified host pool {name!r}.")
            side = self.sides[name]
            if need_size % side.page_size != 0:
                raise AssertionError(
                    "The requested size should be a multiple of the page size."
                )
            page_count = need_size // side.page_size
            request_page_counts.append(page_count)
            page_counts[name] += page_count
        return request_page_counts, page_counts

    @staticmethod
    def _merge_extents(extents: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged = []
        for start, end in sorted(extents):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _logical_free_page_count(side: _Side) -> int:
        return sum(end - start for start, end in side.free_logical_extents)

    def _current_gap_bytes(self) -> int:
        up_sides = [side for side in self.sides.values() if side.grow_direction == "up"]
        down_sides = [
            side for side in self.sides.values() if side.grow_direction == "down"
        ]
        low = up_sides[0].watermark * up_sides[0].page_bytes if up_sides else 0
        high = (
            (down_sides[0].watermark + 1) * down_sides[0].page_bytes
            if down_sides
            else self._allocatable_bytes
        )
        return max(0, high - low)

    def _can_reserve_without_compaction(self, page_counts: dict[str, int]) -> bool:
        extension_bytes = 0
        for name, page_count in page_counts.items():
            side = self.sides[name]
            if page_count > self._logical_free_page_count(side):
                return False
            extension_pages = max(0, page_count - len(side.free_physical_pages))
            extension_bytes += extension_pages * side.page_bytes
        return extension_bytes <= self._current_gap_bytes()

    def _can_fit_packed(self, page_counts: dict[str, int]) -> bool:
        used_bytes = 0
        for name, page_count in page_counts.items():
            side = self.sides[name]
            if page_count > self._logical_free_page_count(side):
                return False
            used_bytes += (side.live_page_count + page_count) * side.page_bytes
        return used_bytes <= self._allocatable_bytes

    def _begin_compaction(self) -> list:
        if self._layout_lease_state.depth:
            raise RuntimeError(
                "Cannot compact a unified host pool while the current thread "
                "holds a layout lease."
            )
        self._compacting = True
        while self._layout_users:
            self._layout_condition.wait()
        events = list(self._pending_transfer_events.values())
        self._pending_transfer_events = {}
        return events

    def _finish_compaction(self) -> None:
        self._compacting = False
        self._layout_condition.notify_all()

    def acquire_layout(self) -> None:
        with self._layout_condition:
            if self._layout_lease_state.depth:
                self._layout_lease_state.depth += 1
                return
            self._prune_completed_transfer_events_locked()
            while self._compacting:
                self._layout_condition.wait()
            self._layout_users += 1
            self._layout_lease_state.depth = 1

    def _prune_completed_transfer_events_locked(self) -> None:
        completed = []
        for key, event in self._pending_transfer_events.items():
            try:
                if event.query():
                    completed.append(key)
            except Exception:
                # A failed query is not proof of completion. Keep the event so
                # compaction will synchronize it before moving its pages.
                continue
        for key in completed:
            del self._pending_transfer_events[key]

    def release_layout(self, finish_event=None, transfer_key=None) -> None:
        with self._layout_condition:
            if self._layout_lease_state.depth <= 0:
                raise RuntimeError(
                    "Unified host layout lease released without acquire."
                )
            self._prune_completed_transfer_events_locked()
            if finish_event is not None:
                if transfer_key is None:
                    transfer_key = id(finish_event)
                self._pending_transfer_events[transfer_key] = finish_event
            self._layout_lease_state.depth -= 1
            if self._layout_lease_state.depth:
                return
            self._layout_users -= 1
            if self._layout_users == 0:
                self._layout_condition.notify_all()

    @contextmanager
    def layout_lease(self):
        self.acquire_layout()
        try:
            yield
        finally:
            self.release_layout()

    def _compact_side(self, side: _Side) -> None:
        reverse = side.grow_direction == "down"
        live = sorted(side.physical_to_logical.items(), reverse=reverse)
        if reverse:
            targets = range(side.page_num - 1, side.page_num - 1 - len(live), -1)
        else:
            targets = range(len(live))

        new_logical_to_physical = {}
        new_physical_to_logical = {}
        for (source, logical), target in zip(live, targets, strict=True):
            if source != target:
                source_offset = source * side.page_bytes
                target_offset = target * side.page_bytes
                self.raw[target_offset : target_offset + side.page_bytes].copy_(
                    self.raw[source_offset : source_offset + side.page_bytes].clone()
                )
            new_logical_to_physical[logical] = target
            new_physical_to_logical[target] = logical

        side.logical_to_physical = new_logical_to_physical
        side.physical_to_logical = new_physical_to_logical
        side.free_physical_pages.clear()
        side.watermark = (
            len(live) if side.grow_direction == "up" else side.page_num - 1 - len(live)
        )

    def _compact(self) -> None:
        events = self._begin_compaction()
        try:
            for event in events:
                event.synchronize()
            for side in self.sides.values():
                self._compact_side(side)
        finally:
            self._finish_compaction()

    def _take_physical_pages(self, side: _Side, page_count: int) -> list[int]:
        reverse = side.grow_direction == "down"
        holes = sorted(side.free_physical_pages, reverse=reverse)
        pages = holes[:page_count]
        side.free_physical_pages.difference_update(pages)
        remaining = page_count - len(pages)
        if remaining == 0:
            return pages

        if side.grow_direction == "up":
            pages.extend(range(side.watermark, side.watermark + remaining))
            side.watermark += remaining
        else:
            pages.extend(range(side.watermark, side.watermark - remaining, -1))
            side.watermark -= remaining
        return pages

    @staticmethod
    def _release_center_holes(side: _Side) -> None:
        if side.grow_direction == "up":
            while side.watermark > 0 and side.watermark - 1 in side.free_physical_pages:
                side.watermark -= 1
                side.free_physical_pages.remove(side.watermark)
        else:
            while (
                side.watermark + 1 < side.page_num
                and side.watermark + 1 in side.free_physical_pages
            ):
                side.watermark += 1
                side.free_physical_pages.remove(side.watermark)

    @staticmethod
    def _page_ids(indices: torch.Tensor, page_size: int) -> list[int]:
        indices = indices.to(dtype=torch.int64, device="cpu").flatten()
        if indices.numel() % page_size != 0:
            raise AssertionError("Host indices must contain whole pages.")
        groups = indices.view(-1, page_size)
        offsets = torch.arange(page_size, dtype=torch.int64)
        if groups.numel() and (
            bool(torch.any(groups[:, 0] % page_size != 0))
            or not torch.equal(groups, groups[:, :1] + offsets)
        ):
            raise AssertionError("Host indices must contain contiguous pages.")
        return (groups[:, 0] // page_size).tolist()

    def available_size(self, name: str) -> int:
        with self.lock:
            side = self.sides[name]
            page_count = min(
                self._logical_free_page_count(side),
                self.free_bytes() // side.page_bytes,
            )
            return page_count * side.page_size

    def free_bytes(self) -> int:
        with self.lock:
            used_bytes = sum(
                side.live_page_count * side.page_bytes for side in self.sides.values()
            )
            return self._allocatable_bytes - used_bytes

    @staticmethod
    def _expand_pages(pages: Sequence[int], page_size: int) -> torch.Tensor:
        if not pages:
            return torch.empty(0, dtype=torch.int64)
        page_ids = torch.tensor(pages, dtype=torch.int64)
        offsets = torch.arange(page_size, dtype=torch.int64)
        return (page_ids[:, None] * page_size + offsets).reshape(-1)

    def alloc(self, name: str, need_size: int) -> torch.Tensor | None:
        result = self.alloc_many(((name, need_size),))
        return None if result is None else result[0]

    def alloc_many(
        self, requests: Sequence[tuple[str, int]]
    ) -> list[torch.Tensor] | None:
        with self.lock:
            request_page_counts, page_counts = self._request_page_counts(requests)
            if not self._can_reserve_without_compaction(page_counts):
                if not self._can_fit_packed(page_counts):
                    return None
                self._compact()
            if not self._can_reserve_without_compaction(page_counts):
                raise RuntimeError(
                    "Unified host-pool compaction did not create the planned capacity."
                )

            logical_extents = {
                name: list(side.free_logical_extents)
                for name, side in self.sides.items()
            }
            logical_pages = []
            for (name, _), page_count in zip(
                requests, request_page_counts, strict=True
            ):
                pages = self._reserve_logical_pages(logical_extents[name], page_count)
                if pages is None:
                    return None
                logical_pages.append(pages)

            physical_pages_by_name = {
                name: self._take_physical_pages(self.sides[name], page_count)
                for name, page_count in page_counts.items()
            }
            physical_offsets = {name: 0 for name in self.sides}
            results = []
            for (name, _), request_logical_pages in zip(
                requests, logical_pages, strict=True
            ):
                side = self.sides[name]
                start = physical_offsets[name]
                end = start + len(request_logical_pages)
                request_physical_pages = physical_pages_by_name[name][start:end]
                physical_offsets[name] = end
                for logical_page, physical_page in zip(
                    request_logical_pages, request_physical_pages, strict=True
                ):
                    side.logical_to_physical[logical_page] = physical_page
                    side.physical_to_logical[physical_page] = logical_page
                results.append(
                    self._expand_pages(request_logical_pages, side.page_size)
                )

            for name, extents in logical_extents.items():
                self.sides[name].free_logical_extents = extents
            return results

    def free(self, name: str, indices: torch.Tensor) -> int:
        with self.lock:
            side = self.sides[name]
            logical_pages = self._page_ids(indices, side.page_size)
            if len(logical_pages) != len(set(logical_pages)):
                raise AssertionError("Host pages cannot be freed more than once.")
            if any(page not in side.logical_to_physical for page in logical_pages):
                raise AssertionError(f"Host pages are not allocated in {name!r}.")
            for logical_page in logical_pages:
                physical_page = side.logical_to_physical.pop(logical_page)
                del side.physical_to_logical[physical_page]
                side.free_physical_pages.add(physical_page)
            side.free_logical_extents = self._merge_extents(
                [
                    *side.free_logical_extents,
                    *((page, page + 1) for page in logical_pages),
                ]
            )
            self._release_center_holes(side)
            return len(indices)

    def clear(self, name: str) -> None:
        with self.lock:
            events = self._begin_compaction()
            try:
                for event in events:
                    event.synchronize()
                side = self.sides[name]
                side.free_logical_extents = [(0, side.page_num)]
                side.logical_to_physical.clear()
                side.physical_to_logical.clear()
                side.free_physical_pages.clear()
                side.watermark = 0 if side.grow_direction == "up" else side.page_num - 1
            finally:
                self._finish_compaction()

    def translate_indices(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        with self.lock:
            side = self.sides[name]
            logical_pages = self._page_ids(indices, side.page_size)
            try:
                physical_pages = [
                    side.logical_to_physical[page] for page in logical_pages
                ]
            except KeyError as error:
                raise AssertionError(
                    f"Host page {error.args[0]} is not allocated in {name!r}."
                ) from error
            translated = self._expand_pages(physical_pages, side.page_size)
            return translated.to(device=indices.device)

    def translate_index(self, name: str, index: int) -> int:
        with self.lock:
            side = self.sides[name]
            logical_page, offset = divmod(index, side.page_size)
            try:
                physical_page = side.logical_to_physical[logical_page]
            except KeyError as error:
                raise AssertionError(
                    f"Host page {logical_page} is not allocated in {name!r}."
                ) from error
            return physical_page * side.page_size + offset


class UnifiedPageEnvelopeHostPool(HostKVCache):
    """Host mirror that transfers complete unified-memory page envelopes."""

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
        _shared_backing: _SharedPageEnvelopeHostBacking | None = None,
    ):
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Unified page-envelope host pools do not support packed MTP draft pools."
            )

        device_page_buffer = device_pool.get_page_envelope_buffer()
        if device_page_buffer.ndim != 2:
            raise ValueError(
                "Unified page-envelope buffers must have shape [pages, bytes]; "
                f"got {tuple(device_page_buffer.shape)}."
            )
        page_bytes = int(device_page_buffer.shape[1])
        if page_bytes % page_size != 0:
            raise ValueError(
                "Unified page-envelope bytes must be divisible by page_size; "
                f"got page_bytes={page_bytes}, page_size={page_size}."
            )
        if device_pool.page_size != page_size:
            raise ValueError(
                "Unified device and host page sizes must match; "
                f"got device={device_pool.page_size}, host={page_size}."
            )
        if layout not in {"layer_first", "page_first", "page_first_direct"}:
            raise ValueError(f"Unsupported layout: {layout}")

        if _shared_backing is None:
            _shared_backing = _SharedPageEnvelopeHostBacking(
                device_pools=(device_pool,),
                pool_names=(pool_label,),
                host_to_device_ratio=host_to_device_ratio,
                host_size=host_size,
                pin_memory=pin_memory,
                device=device,
                allocator_type=allocator_type,
            )

        self.device_pool = device_pool
        self.pool_label = pool_label
        self._shared_backing = _shared_backing
        self._device_page_buffer = device_page_buffer
        self._host_page_buffer_view = _shared_backing.page_buffer(pool_label)
        self.size_per_token = page_bytes // page_size
        self.item_bytes = page_bytes
        self.page_size = page_size
        self.page_num = _shared_backing.sides[pool_label].page_num
        self.size = self.page_num * page_size
        self.dcp_size = 1
        self.dcp_rank = 0
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.dtype = torch.uint8
        self.allocator = _shared_backing.allocator
        self.layer_num = 1
        self.can_use_write_back_jit = False
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.kv_buffer = (
            _shared_backing.raw
            if pool_label == _shared_backing.registration_owner
            else None
        )
        self.fd = _shared_backing.fd
        self.lock = _shared_backing.lock
        self._destroyed = False
        _shared_backing.retain(pool_label)

    @classmethod
    def build_hybrid_swa_pool_pair(
        cls,
        *,
        device_pools: Sequence,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
        layout: str,
        allocator_type: str,
        mtp_draft_device_pools: Sequence = (),
        pin_memory: bool = True,
        device: str = "cpu",
    ) -> tuple[HostKVCache, HostKVCache]:
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Unified page-envelope host pools do not support packed MTP draft pools."
            )
        if len(device_pools) != 2:
            raise ValueError("A hybrid-SWA host pool pair requires exactly two pools.")

        pool_names = ("full", "swa")
        backing = _SharedPageEnvelopeHostBacking(
            device_pools=device_pools,
            pool_names=pool_names,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
        )
        host_pools = []
        try:
            for device_pool, pool_name in zip(device_pools, pool_names, strict=True):
                host_pools.append(
                    cls(
                        device_pool,
                        host_to_device_ratio,
                        0,
                        page_size,
                        layout,
                        pin_memory=pin_memory,
                        device=device,
                        allocator_type=allocator_type,
                        pool_label=pool_name,
                        _shared_backing=backing,
                    )
                )
        except Exception:
            for host_pool in host_pools:
                host_pool.destroy()
            backing.destroy_if_unused()
            raise
        return tuple(host_pools)

    def get_size_per_token(self):
        return self.size_per_token

    def get_ksize_per_token(self):
        return self.size_per_token

    @property
    def shared_allocation_domain(self):
        return self._shared_backing

    def init_kv_buffer(self):
        return self._host_page_buffer_view

    def get_hybrid_pool_buffer(self):
        if self.pool_label == self._shared_backing.registration_owner:
            return [self._shared_backing.raw]
        return []

    def clear(self) -> None:
        self._shared_backing.clear(self.pool_label)

    def available_size(self):
        return self._shared_backing.available_size(self.pool_label)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        return self._shared_backing.alloc(self.pool_label, need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self._shared_backing.free(self.pool_label, indices)

    def prepare_transfer_indices(
        self, host_indices, device_indices, io_backend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        host_indices = self._shared_backing.translate_indices(
            self.pool_label, host_indices
        )
        if io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(
                    self._device_page_buffer.device, non_blocking=True
                )
            return host_indices, device_indices
        if io_backend == "direct":
            if self.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, order = host_indices.sort()
                return host_indices, device_indices.index_select(0, order)
            if self.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            raise ValueError(
                f"Unsupported layout {self.layout!r} for io backend 'direct'"
            )
        if io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        raise ValueError(f"Unsupported io backend: {io_backend}")

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.kv_buffer = None
        self._host_page_buffer_view = None
        self._shared_backing.release(self.pool_label)

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(dtype=torch.int64)
        if indices.numel() == 0:
            return indices
        if indices.numel() % self.page_size != 0:
            raise ValueError(
                "Unified page-envelope transfer index count must be page-aligned; "
                f"got {indices.numel()} indices for page_size={self.page_size}."
            )
        grouped = indices.reshape(-1, self.page_size)
        # Device indices are produced by the page allocator. Validating every
        # value here would turn both reductions into host-device synchronizations
        # on every offload. Keep the full check for CPU/debug inputs while the
        # hot CUDA/ROCm path relies on that allocator invariant.
        if indices.device.type == "cpu":
            offsets = torch.arange(self.page_size, dtype=torch.int64)
            if bool(torch.any(grouped[:, 0] % self.page_size != 0)) or not torch.equal(
                grouped, grouped[:, :1] + offsets
            ):
                raise ValueError("Unified page-envelope transfers require whole pages.")
        return grouped[:, 0] // self.page_size

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                "Unified page-envelope transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}."
            )
        return host_indices.numel() > 0

    def _transfer_pages(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_pages: torch.Tensor,
        dst_pages: torch.Tensor,
        io_backend: str,
    ) -> None:
        if io_backend not in {"kernel", "direct"}:
            raise ValueError(f"Unsupported io_backend: {io_backend}")
        if self._device_page_buffer.device.type == "cpu":
            dst.index_copy_(0, dst_pages, src.index_select(0, src_pages))
            return

        if not (_is_cuda or _is_hip):
            raise RuntimeError(
                "Unified page-envelope transfers require a CUDA or ROCm device."
            )
        if io_backend == "kernel":
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_pages,
                dst_indices=dst_pages,
                item_size=self.item_bytes,
            )
        else:
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_pages,
                dst_indices=dst_pages,
                page_size=1,
            )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        self._shared_backing.acquire_layout()
        finish_event = None
        transfer_key = None
        try:
            host_indices, device_indices = self.prepare_transfer_indices(
                host_indices, device_indices, io_backend
            )
            self.backup_from_device_all_layer_physical(
                device_pool, host_indices, device_indices, io_backend
            )
            if self._device_page_buffer.device.type != "cpu":
                finish_event = torch.cuda.Event()
                finish_event.record()
                transfer_key = (
                    "direct",
                    int(torch.cuda.current_stream().cuda_stream),
                )
        except Exception:
            if self._device_page_buffer.device.type != "cpu":
                torch.cuda.current_stream().synchronize()
            raise
        finally:
            self._shared_backing.release_layout(finish_event, transfer_key)

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        self._transfer_pages(
            self._device_page_buffer,
            self._host_page_buffer_view,
            self._to_page_indices(device_indices),
            self._to_page_indices(host_indices),
            io_backend,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError(
                "Unified page-envelope host pools do not contain draft KV."
            )
        if layer_id != 0 or not self._has_transfer_indices(
            host_indices, device_indices
        ):
            return
        self._shared_backing.acquire_layout()
        finish_event = None
        transfer_key = None
        try:
            host_indices, device_indices = self.prepare_transfer_indices(
                host_indices, device_indices, io_backend
            )
            self.load_to_device_per_layer_physical(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=is_draft,
            )
            if self._device_page_buffer.device.type != "cpu":
                finish_event = torch.cuda.Event()
                finish_event.record()
                transfer_key = (
                    "direct",
                    int(torch.cuda.current_stream().cuda_stream),
                )
        except Exception:
            if self._device_page_buffer.device.type != "cpu":
                torch.cuda.current_stream().synchronize()
            raise
        finally:
            self._shared_backing.release_layout(finish_event, transfer_key)

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError(
                "Unified page-envelope host pools do not contain draft KV."
            )
        if layer_id != 0 or not self._has_transfer_indices(
            host_indices, device_indices
        ):
            return
        self._transfer_pages(
            self._host_page_buffer_view,
            self._device_page_buffer,
            self._to_page_indices(host_indices),
            self._to_page_indices(device_indices),
            io_backend,
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        with self._shared_backing.layout_lease():
            physical = self._shared_backing.translate_index(self.pool_label, int(index))
            page = self._host_page_buffer_view[physical // self.page_size]
            return page.flatten() if flat else page.view(1, self.item_bytes)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.item_bytes,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def get_page_buffer_element_size(self, split_factor: int = 1) -> int:
        if split_factor != 1:
            raise ValueError(
                "Unified page-envelope host pools do not support split-head metadata."
            )
        return self.item_bytes

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        with self._shared_backing.layout_lease():
            physical = self._shared_backing.translate_index(self.pool_label, int(index))
            page = data_page.view(self.dtype).reshape(self.item_bytes)
            self._host_page_buffer_view[physical // self.page_size].copy_(page)

    def get_page_buffer_meta(self, indices):
        physical = self._shared_backing.translate_indices(self.pool_label, indices)
        pages = self._to_page_indices(physical).tolist()
        base_ptr = self._host_page_buffer_view.data_ptr()
        ptrs = [base_ptr + int(page) * self.item_bytes for page in pages]
        return ptrs, [self.item_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        buffer = self._host_page_buffer_view
        return (
            buffer.data_ptr() % page_size_bytes == 0
            and self.item_bytes % page_size_bytes == 0
        )
