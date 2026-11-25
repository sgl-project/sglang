"""Two-ended DRAM backing for Ascend's compressed FULL and native SWA pages."""

from __future__ import annotations

import math
import threading
from collections import deque
from contextlib import contextmanager

import torch
from sglang.srt.mem_cache.kv.memory.host_backing import _SharedPageEnvelopeHostBacking


class UnsupportedPageGeometry(ValueError):
    """The budget or the native row alignment cannot fit both page types."""


def _region_offsets(start, regions):
    """Find row-aligned contiguous layer arrays inside one exact-size FULL page."""

    def place(at, remaining):
        if not remaining:
            return {}
        for name in remaining:
            width, layers = regions[name]
            if at % width == 0:
                rest = place(
                    at + width * layers, tuple(k for k in remaining if k != name)
                )
                if rest is not None:
                    rest[name] = at // width
                    return rest
        return None

    offsets = place(start, tuple(regions))
    if offsets is None:
        raise UnsupportedPageGeometry(
            "FULL envelope cannot fit the native row alignment"
        )
    return offsets


class DualEndedHostArena(_SharedPageEnvelopeHostBacking):
    """Use the shared host backing's logical IDs, reservations and watermark rules.

    The 910B transfers address each component in native row units. FULL pages
    at different alignment phases can therefore have different internal layouts.
    SWA pages always have the same layout.
    """

    def __init__(self, full_regions, swa_region, full_pages, swa_pages, allocate):
        full_bytes = sum(w * n for w, n in full_regions.values())
        swa_bytes = math.prod(swa_region)
        self.budget = full_pages * full_bytes + swa_pages * swa_bytes
        self._allocatable_bytes = self.budget // swa_bytes * swa_bytes
        if self._allocatable_bytes < full_bytes + swa_bytes:
            raise UnsupportedPageGeometry("Need space for both page types")
        alignment = math.lcm(*(w for w, _ in full_regions.values()))
        period = alignment // math.gcd(full_bytes, alignment)
        offsets = [_region_offsets(i * full_bytes, full_regions) for i in range(period)]
        self.lock = threading.RLock()
        self._events = {}
        self._retired = deque()
        self._local = threading.local()
        self.sides = {
            name: self._Side(
                page_size=1,
                page_bytes=width,
                page_num=self._allocatable_bytes // width,
                grow_direction=direction,
            )
            for name, width, direction in (
                ("full", full_bytes, "up"),
                ("swa", swa_bytes, "down"),
            )
        }
        self.raw = self.buffer = allocate(self.budget)
        self.full, self.swa = (_Owner(self, name) for name in ("full", "swa"))
        self._regions = {
            name: _Region(self.full, w, n, [o[name] for o in offsets])
            for name, (w, n) in full_regions.items()
        }
        self._regions[None] = _Region(self.swa, *swa_region, offsets=[0])
        self.moves = 0

    def region(self, name=None):
        return self._regions[name]

    def acquire_layout(self):
        self.lock.acquire()
        self._local.depth = getattr(self._local, "depth", 0) + 1

    def release_layout(self, finish_event=None, transfer_key=None):
        try:
            if finish_event is not None:
                self._events[
                    transfer_key if transfer_key is not None else id(finish_event)
                ] = finish_event
        finally:
            self._local.depth -= 1
            self.lock.release()

    @contextmanager
    def submission(self, device):
        """Direct API callers also pin the layout until their copy stream finishes."""
        if getattr(self._local, "depth", 0):
            yield
            return
        stream = torch.npu.current_stream(device)
        event = torch.npu.Event()
        self.acquire_layout()
        try:
            yield
            event.record(stream)
        except BaseException:
            stream.synchronize()
            raise
        else:
            self._events[("direct", device, stream.npu_stream)] = event
        finally:
            self.release_layout()

    def _reap(self, wait=False):
        if wait:
            for event in self._events.values():
                event.synchronize()
        self._events = {k: e for k, e in self._events.items() if not e.query()}
        while self._retired:
            owner, ids, fences = self._retired[0]
            if not all(e.query() for e in fences):
                break
            super().free(owner.name, ids)
            owner.retired.difference_update(ids.tolist())
            self._retired.popleft()

    def _compact(self):
        if getattr(self._local, "depth", 0):
            raise RuntimeError("Cannot compact during host address preparation")
        self._reap(wait=True)
        for owner in (self.full, self.swa):
            side = self.sides[owner.name]
            self._release_center_holes(side)
            while side.free_physical_pages:
                src = side.watermark - 1 if owner is self.full else side.watermark + 1
                dst = next(iter(side.free_physical_pages))
                logical = side.physical_to_logical[src]
                if owner is self.swa:
                    self.page_buffer(owner.name)[dst].copy_(
                        self.page_buffer(owner.name)[src]
                    )
                else:
                    for name, region in self._regions.items():
                        if name is not None:
                            a, b = region.row(src), region.row(dst)
                            rows = region.layer_rows(0)
                            rows[b : b + region.layers].copy_(
                                rows[a : a + region.layers]
                            )
                # Destination was a hole; publish only after every component copied.
                side.logical_to_physical[logical] = dst
                side.physical_to_logical[dst] = logical
                del side.physical_to_logical[src]
                side.free_physical_pages.remove(dst)
                side.free_physical_pages.add(src)
                self._release_center_holes(side)
                self.moves += 1


class _Owner:
    def __init__(self, arena, name):
        self.arena, self.name = arena, name
        self.retired = set()

    def available(self):
        with self.arena.lock:
            self.arena._reap()
            return self.arena.available_size(self.name)

    def capacity(self):
        with self.arena.lock:
            return self.available() + self.arena.sides[self.name].live_page_count

    def alloc(self, count, page_size):
        if count < 0 or page_size <= 0:
            raise ValueError("Invalid host allocation size")
        with self.arena.lock:
            self.arena._reap()
            side = self.arena.sides[self.name]
            counts = {k: count if k == self.name else 0 for k in self.arena.sides}
            if not self.arena._can_reserve_without_compaction(counts):
                if not self.arena._can_fit_packed(counts):
                    return None
                self.arena._compact()
            # Stage all fallible tensor allocations ahead of the ownership update.
            extents = list(side.free_logical_extents)
            ids = self.arena._reserve_logical_pages(extents, count)
            if ids is None:
                return None
            slots = self.arena._expand_pages(ids, page_size)
            positions = self.arena._take_physical_pages(side, count)
            side.logical_to_physical.update(zip(ids, positions, strict=True))
            side.physical_to_logical.update(zip(positions, ids, strict=True))
            side.free_logical_extents = extents
            return slots

    def free(self, slots, page_size):
        with self.arena.lock:
            self.arena._reap()
            pages = self.arena._page_ids(slots, page_size)
            mapping = self.arena.sides[self.name].logical_to_physical
            ids = set(pages)
            if (
                len(ids) != len(pages)
                or not ids.issubset(mapping)
                or ids & self.retired
            ):
                raise ValueError("Unallocated or already released host page")
            if getattr(self.arena._local, "depth", 0):
                raise RuntimeError("Cannot release pages during transfer submission")
            p = torch.tensor(pages, dtype=torch.int64, device="cpu")
            if self.arena._events:
                self.retired.update(ids)
                self.arena._retired.append(
                    (self, p, tuple(self.arena._events.values()))
                )
            else:
                self.arena.free(self.name, p)
            return slots.numel()

    def clear(self):
        with self.arena.lock:
            if getattr(self.arena._local, "depth", 0):
                raise RuntimeError("Cannot clear pages during transfer submission")
            self.arena._reap(wait=True)
            self.arena.free(
                self.name,
                torch.tensor(
                    list(self.arena.sides[self.name].logical_to_physical),
                    dtype=torch.int64,
                    device="cpu",
                ),
            )


class _Region:
    def __init__(self, owner, width, layers, offsets):
        self.owner, self.arena = owner, owner.arena
        self.transfer_layout_domain = self.arena
        self.width, self.layers = width, layers
        self.offsets = tuple(offsets)
        self.table = torch.tensor(offsets, dtype=torch.int64, device="cpu")
        self.stride = self.arena.sides[owner.name].page_bytes * len(offsets) // width

    def layer_rows(self, layer):
        start = layer * self.width
        end = start + (self.arena.budget - start) // self.width * self.width
        return self.arena.buffer[start:end].view(-1, self.width)

    def row(self, physical):
        period = len(self.offsets)
        if isinstance(physical, int):
            return physical // period * self.stride + self.offsets[physical % period]
        if period == 1:
            return physical * self.stride + self.offsets[0]
        return physical // period * self.stride + self.table[physical % period]

    def physical_rows(self, logical):
        with self.arena.lock:
            mapping = self.arena.sides[self.owner.name].logical_to_physical
            if isinstance(logical, int):
                return self.row(mapping[logical])
            pages = torch.tensor(
                [mapping[x] for x in logical.tolist()], dtype=torch.int64, device="cpu"
            )
            return self.row(pages).to(logical.device)
