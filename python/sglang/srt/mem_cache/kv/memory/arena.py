"""The unified placement's device bytes.

One ``uint8`` region holds every sub-pool of a plan as a band of whole
entries. The two end bands own the region's ends and grow inward; float bands
live between their frontiers. A band is moved and reset by unit: a token
band's unit is a page of entries, a state band's unit one entry. What the
bytes mean belongs to the layouts (`kv/layout/`), where a band's frontier
sits to the allocator (`kv/alloc/`), and which units a request holds to the
request table.

The low bytes are the padding sink (`KVStoragePlan.sink_bytes`): every
band's dummy writes (entry 0) land there, so each band's first usable entry
lies past it.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import msgspec
import torch

from sglang.kernels.ops.kvcache.copy_pages import copy_pages
from sglang.kernels.ops.kvcache.zero_pages import zero_pages
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv.plan import KVStoragePlan
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmBufferOwner
from sglang.srt.mem_cache.memory_pool import KvBufferDesc
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class ArenaBand(msgspec.Struct, frozen=True, kw_only=True):
    """One sub-pool's band: its entry size, its unit and which end it grows from."""

    name: str
    entry_bytes: int
    unit_entries: int
    grow_direction: str  # "up" | "down" | "float"
    # Units handed out zeroed, for layouts whose readers load whole pages.
    zero_on_handout: bool

    @property
    def unit_bytes(self) -> int:
        return self.entry_bytes * self.unit_entries


def plan_bands(plan: KVStoragePlan) -> Tuple[ArenaBand, ...]:
    """The bands of a plan's sub-pools, in chain order."""
    return tuple(
        ArenaBand(
            name=s.name,
            entry_bytes=s.spec.entry_bytes(),
            unit_entries=s.unit,
            grow_direction=s.spec.grow_direction,
            zero_on_handout=s.spec.zero_units_on_handout,
        )
        for s in plan.sub_pools
    )


class KVArena:
    """The device bytes of the unified placement, as bands of entries."""

    def __init__(
        self,
        *,
        bands: Sequence[ArenaBand],
        total_bytes: int,
        sink_bytes: int,
        device: str,
        enable_memory_saver: bool = False,
        backed_after_capture: bool = False,
    ):
        names = [b.name for b in bands]
        assert len(bands) >= 2, f"an arena needs >= 2 bands; got {names}"
        assert len(set(names)) == len(names), f"band names must be unique: {names}"
        up = [b for b in bands if b.grow_direction == "up"]
        down = [b for b in bands if b.grow_direction == "down"]
        assert len(up) == 1 and len(down) == 1, (
            "an arena needs exactly one grow-up and one grow-down end band; got "
            f"{[b.grow_direction for b in bands]}"
        )
        floats = [b for b in bands if b.grow_direction == "float"]
        # Low byte end -> high: the grow-up end, the floats, the grow-down end.
        self.bands: Tuple[ArenaBand, ...] = (up[0], *floats, down[0])
        self._by_name: Dict[str, ArenaBand] = {b.name: b for b in self.bands}
        self.device = device
        self.total_bytes = total_bytes
        self.backed_after_capture = backed_after_capture
        self._sink = sink_bytes
        self._vmm_owner = None
        self.page_parts = {}

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            if backed_after_capture:
                dev = torch.device(device)
                if dev.type != "cuda":
                    raise ValueError(
                        "backing the KV arena after graph capture needs a CUDA device"
                    )
                device_id = (
                    dev.index if dev.index is not None else torch.cuda.current_device()
                )
                self._vmm_owner = KvVmmBufferOwner(
                    device=device,
                    device_id=device_id,
                    store_dtype=torch.uint8,
                    page_size=1,
                    reserved_num_tokens=total_bytes,
                    buffer_descs=[
                        KvBufferDesc(
                            "unified_raw",
                            (total_bytes,),
                            row_bytes=1,
                            tokens_per_row=1,
                        )
                    ],
                )
                self.raw = self._vmm_owner.tensors[0]
                # Capture routes every dummy write to entry 0, inside the sink:
                # back it before any graph is captured.
                self._vmm_owner.ensure_prefix(self._sink)
                self.accessible_bytes = self._sink
            else:
                self.raw = torch.empty(total_bytes, dtype=torch.uint8, device=device)
                self.accessible_bytes = total_bytes
        self._initialize(self.raw[: self.accessible_bytes])

        # A band's views and moves span the entries of the bytes reserved now;
        # its capacity follows the backed bytes (`finalize_backing`).
        self._extent: Dict[str, int] = {}
        self._num_entries: Dict[str, int] = {}
        self._first_entry: Dict[str, int] = {}
        for band in self.bands:
            num_entries = total_bytes // band.entry_bytes
            first_entry = -(-self._sink // band.entry_bytes)
            if num_entries <= first_entry:
                raise RuntimeError(
                    f"KVArena: band {band.name!r} fits only {num_entries} entries "
                    f"in {total_bytes} bytes, but the sink takes the first "
                    f"{first_entry}. Increase total_bytes."
                )
            self._extent[band.name] = num_entries
            self._num_entries[band.name] = num_entries
            self._first_entry[band.name] = first_entry

        logger.info(
            "[unified-memory-pool] KVArena %s: total_bytes=%.2f GB (=%d B), %d band(s)",
            "reserved VA upper bound" if backed_after_capture else "allocated",
            total_bytes / GB,
            total_bytes,
            len(self.bands),
        )
        for band in self.bands:
            logger.info(
                "[unified-memory-pool]   band %r: grow=%s, entry_bytes=%d, "
                "unit_entries=%d, entries=%d, first_entry=%d (entries [0,%d) are "
                "the sink)",
                band.name,
                band.grow_direction,
                band.entry_bytes,
                band.unit_entries,
                self._num_entries[band.name],
                self._first_entry[band.name],
                self._first_entry[band.name],
            )

    def _initialize(self, raw: torch.Tensor) -> None:
        if envs.SGLANG_DEBUG_POISON_POOL.get():
            # Debug: bf16-NaN-fill so NaN-unsafe reads of never-written bytes
            # fail deterministically.
            raw.view(torch.int16).fill_(0x7FC1)
            raw[: self._sink].zero_()
            logger.warning(
                "[unified-memory-pool] POISONED: backed arena bytes filled with "
                "bf16-NaN patterns (SGLANG_DEBUG_POISON_POOL)"
            )
        else:
            raw.zero_()  # never-written entries read as zeros

    # -- bands --

    def band(self, name: str) -> ArenaBand:
        return self._by_name[name]

    def num_entries(self, name: str) -> int:
        """The band's entries in the backed bytes (entry 0 included)."""
        return self._num_entries[name]

    def first_entry(self, name: str) -> int:
        """The band's first entry past the sink."""
        return self._first_entry[name]

    def extent_entries(self, name: str) -> int:
        """The band's entries in the reserved bytes: what its views span."""
        return self._extent[name]

    def entries(self, name: str) -> torch.Tensor:
        """The band's entries as rows of bytes, ``(extent, entry_bytes)``."""
        band = self._by_name[name]
        count = self._extent[name]
        return self.raw[: count * band.entry_bytes].view(count, band.entry_bytes)

    # -- units --

    def move_units(self, name: str, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy whole units ``src`` -> ``dst`` of one band (disjoint sets of
        physical unit ids), on the current stream."""
        band = self._by_name[name]
        copy_pages(
            self.raw,
            dst,
            src,
            self._extent[name] // band.unit_entries,
            band.unit_bytes,
        )
        if name in self.page_parts:
            self.page_parts[name].copy_rows(dst, src)

    def zero_units(self, name: str, units: torch.Tensor) -> None:
        """Zero whole units (physical unit ids) of one band."""
        band = self._by_name[name]
        zero_pages(
            self.raw,
            units,
            self._extent[name] // band.unit_entries,
            band.unit_bytes,
        )
        if name in self.page_parts:
            self.page_parts[name].reset_rows(units)

    # -- backing --

    @property
    def backed_bytes(self) -> int:
        """Bytes backed so far when the arena is backed after capture, else 0."""
        return 0 if self._vmm_owner is None else self._vmm_owner.backed_bytes

    def finalize_backing(self, total_bytes: int) -> None:
        """Back the final byte budget without moving any tensor address; band
        capacities follow it."""
        if self._vmm_owner is None:
            raise RuntimeError("the KV arena is not backed after capture")
        for band in self.bands:
            num_entries = total_bytes // band.entry_bytes
            first_unit = -(-self._first_entry[band.name] // band.unit_entries)
            if num_entries // band.unit_entries <= first_unit:
                raise ValueError(
                    f"arena size {total_bytes} leaves no allocatable unit for band "
                    f"{band.name!r} after the sink"
                )
        self._vmm_owner.ensure_prefix(total_bytes)
        self._initialize(self.raw[:total_bytes])
        self.total_bytes = total_bytes
        self.accessible_bytes = total_bytes
        self._num_entries = {
            band.name: total_bytes // band.entry_bytes for band in self.bands
        }
