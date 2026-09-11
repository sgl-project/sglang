# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Address space of a saved CUDA graph (design sections 6.4 and 7).

A saved graph never stores a device pointer. Every pointer word is a
:class:`RegionRef` into a region its owner named (``kv:k_buffer:0``,
``static:input_ids``, ``pool:seg:2``, ``comm:tp:ca_v2:table``). At load,
:meth:`MemoryMap.bind` maps each saved region onto the live region of the
same NAME:

* ``RELOCATE`` (default): every pointer word is patched by the region's
  delta. Relocation is complete for every kernel class SGLang launches on
  Blackwell (facts 9, 10, 14), so this is the correctness baseline.
* ``FIXED_VA``: SGLang-owned regions must sit at their saved address, which
  a ``FixedArena`` (``utils/cuda_vmm_utils.py``) guarantees by reserving VA
  at a requested base (deterministic, facts 7 and 13); everything else
  relocates. A pinned region whose live base differs is rejected, never
  silently relocated.

Owners in v1 (each implements :class:`MemoryOwner` on its own class; none
does yet): the model loader (one region per weight storage), the KV pool
(buffers, ``req_to_token``, and the ``k/v_data_ptrs`` tables it rebinds as
data), the graph runners (static input buffers, shared logits), the attention
backends (workspaces), the graph memory pool (whole segments, fact 19), the
cuBLAS workspace (fact 14) and the communicators (``comm.py``).
"""

from __future__ import annotations

import bisect
from enum import Enum
from typing import Mapping, Optional, Protocol, Sequence

import msgspec


class Placement(str, Enum):
    RELOCATE = "relocate"
    FIXED_VA = "fixed_va"


class Region(msgspec.Struct, frozen=True, kw_only=True):
    """One named, contiguous device range. ``base`` is the address at save;
    it is a diagnostic and the ``FIXED_VA`` target, never a correctness input."""

    id: str
    kind: str  # weight | kv | req_to_token | static | attn_ws | pool | cublas_ws | comm_table
    nbytes: int
    base: int


class RegionRef(msgspec.Struct, frozen=True, array_like=True):
    """A device address as ``(region, offset)``. Interior offsets are common
    (fact 9), so binding is range based."""

    region: str
    offset: int = 0


class MemoryOwner(Protocol):
    """Anything that owns graph-visible device memory."""

    def regions(self) -> list[Region]:
        """Live regions with stable names; called on both save and load."""

    def rebind(self, reloc: Relocation) -> None:
        """Rewrite pointers this owner stores AS DATA in device memory (the KV
        ``k/v_data_ptrs`` tables, design section 7.3). Default: nothing."""


class Relocation:
    """Saved region id -> live base, built by :meth:`MemoryMap.bind`.

    ``rejected`` names the saved regions the placement could not bind and
    why; a graph that references one of them must be recaptured.
    """

    def __init__(self, live_bases: Mapping[str, int], rejected: Mapping[str, str]):
        self.live_bases = dict(live_bases)
        self.rejected = dict(rejected)

    def rebase(self, ref: RegionRef) -> int:
        if ref.region in self.rejected:
            raise KeyError(f"{ref.region}: {self.rejected[ref.region]}")
        return self.live_bases[ref.region] + ref.offset


class MemoryMap:
    """Every live region of this process, across all owners."""

    def __init__(self, owners: Sequence[MemoryOwner]) -> None:
        self.owners = tuple(owners)
        # Regions nest (a static buffer lives inside a graph-pool segment).
        # Sorted by base, larger first on ties, so a backward walk from the
        # bisection point meets the innermost enclosing region first.
        live = sorted(
            (r for o in self.owners for r in o.regions()),
            key=lambda r: (r.base, -r.nbytes),
        )
        self._regions = live
        self._bases = [r.base for r in live]

    def regions(self) -> list[Region]:
        return list(self._regions)

    def classify(self, ptr: int) -> Optional[RegionRef]:
        """The innermost live region containing ``ptr``, by bisection over live
        ranges only (never ``/proc/self/maps``, fact 17), or ``None``."""
        i = bisect.bisect_right(self._bases, ptr) - 1
        while i >= 0:
            r = self._regions[i]
            if ptr < r.base + r.nbytes:
                return RegionRef(r.id, ptr - r.base)
            i -= 1
        return None

    def bind(self, saved: Sequence[Region], placement: Placement) -> Relocation:
        """Bind saved regions to live ones by name (design section 7.2).

        A saved region is rejected when no live region has its name, when the
        sizes differ, or, under ``FIXED_VA`` for a pinned kind, when the live
        base is not the saved base.
        """
        live = {r.id: r for r in self._regions}
        bases: dict[str, int] = {}
        rejected: dict[str, str] = {}
        for s in saved:
            l = live.get(s.id)
            if l is None:
                rejected[s.id] = "no live region with this name"
            elif l.nbytes != s.nbytes:
                rejected[s.id] = f"size changed {s.nbytes} -> {l.nbytes}"
            elif (
                placement is Placement.FIXED_VA
                and s.kind in PINNED_KINDS
                and l.base != s.base
            ):
                rejected[s.id] = f"pinned region moved 0x{s.base:x} -> 0x{l.base:x}"
            else:
                bases[s.id] = l.base
        return Relocation(bases, rejected)

    def rebind(self, reloc: Relocation) -> None:
        for owner in self.owners:
            rebind = getattr(owner, "rebind", None)
            if rebind is not None:
                rebind(reloc)


# Kinds a FIXED_VA placement pins inside FixedArena reservations; the rest
# relocate (design section 7.2: static and pool first, weights and KV with
# the vmm_fd weight transport).
PINNED_KINDS = frozenset({"static", "pool", "cublas_ws"})


def reserve_fixed_arenas(
    device_id: int, bases: Mapping[str, int], sizes: Mapping[str, int]
) -> Mapping[str, object]:
    """Reserve one ``FixedArena`` per pinned kind right after ``set_device``
    and before distributed init; a kind whose reservation collides falls back
    to ``RELOCATE`` (design section 7.2)."""
    raise NotImplementedError(
        "reserve_fixed_arenas: FIXED_VA arena reservation is not implemented in "
        "this draft; see DESIGN_cuda_graph_serialization.md section 7.2"
    )
