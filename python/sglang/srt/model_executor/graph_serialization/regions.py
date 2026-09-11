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
"""Symbolic device regions and their placement at load (design sections 6.4, 7).

The artifact never stores a bare device pointer (design section 6.1, decision
1). Every pointer-valued word of a captured node is a ``RegionRef(region_id,
offset)`` into a region an owner registered by a stable name: a weight
storage, a KV buffer, a buffer-registry slot, a graph-pool segment. This
module holds the three pieces that give those names meaning in a process:

* :class:`RegionProvider` implementations enumerate ``(RegionSpec, live base)``
  for one owner class; :class:`RegionRegistry` merges them into one sorted
  table and classifies raw words by bisecting the live ``[base, base + nbytes)``
  ranges only -- never through ``/proc/self/maps`` (fact 17: the ``slot_bytes``
  scalar coincides with the python3 ELF base).
* :class:`RegionPlacementPolicy` decides, per region kind, whether a saved
  region must land at its saved address (``PIN``, asserted by *name*), is
  patched by its delta (``RELOCATE``) or is unsupported (``REJECT``, the shape
  recaptures). :class:`RelocatePolicy` is the correctness baseline, proven
  complete for every kernel class SGLang launches on Blackwell (facts 9, 10,
  14); :class:`FixedVaArenaPolicy` pins what SGLang owns through
  :class:`~sglang.srt.utils.cuda_vmm_utils.FixedArena` (facts 7, 13).
* :class:`RelocationMap` is the per-load answer the codec consumes: one delta
  and one live base per accepted region, one reason per rejected region.

Everything here is pure Python and runs on CPU. The v1 provider skeletons at
the end of the module name their exact source of truth (design section 7.1)
and raise ``NotImplementedError`` from ``enumerate`` in this draft.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from bisect import bisect_right
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence

import msgspec

from sglang.srt.model_executor.graph_serialization.format import (
    RegionKind,
    RegionRef,
    RegionSpec,
)
from sglang.srt.utils.cuda_vmm_utils import FixedArena, FixedArenaCollision

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_DESIGN_DOC = "DESIGN_cuda_graph_serialization.md"
_KNOWN_KINDS: frozenset[str] = frozenset(kind.value for kind in RegionKind)


def _not_implemented(where: str, what: str, section: str) -> NotImplementedError:
    return NotImplementedError(
        f"{where}: {what} is not implemented in this draft; "
        f"see {_DESIGN_DOC} section {section}"
    )


class Disposition(str, Enum):
    """What a placement policy does with one saved region (design section 6.1).

    ``PIN``: the live region must sit at ``RegionSpec.base_at_save``, checked by
    region name, never inferred from an address. ``RELOCATE``: every pointer
    word into the region is patched by ``live_base - base_at_save``.
    ``REJECT``: unsupported; every graph that references the region recaptures.
    """

    PIN = "pin"
    RELOCATE = "relocate"
    REJECT = "reject"


class RegionRejected(RuntimeError):
    """``RelocationMap.rebase`` was asked for a region the policy rejected."""


class RegionProvider(Protocol):
    """One owner class of graph-visible device memory (design section 6.4)."""

    kind: str

    def enumerate(self) -> list[tuple[RegionSpec, int]]:
        """``(spec, live base)`` for every allocation of this kind, with stable
        names. Called after the owning object exists, on both save and load."""
        ...

    def rebind_data(self, reloc: RelocationMap) -> None:
        """Rewrite pointers this owner stores *as data* in device memory (the
        KV ``data_ptrs`` tables). Default no-op. Comm tables are handled by
        ``CommGraphState`` instead (design section 6.10)."""
        ...


class RelocationMap(msgspec.Struct, kw_only=True):
    """Per-load binding of saved region ids to live addresses.

    Built incrementally by :meth:`RegionPlacementPolicy.build`, hence not
    frozen. ``deltas`` is the word-patching view (``live_base - base_at_save``,
    ``0`` for a pinned region); ``live_bases`` is what :meth:`rebase` needs to
    turn a ``RegionRef`` into an absolute live address; ``rejected`` carries the
    reason per region the policy refused.
    """

    deltas: dict[str, int] = {}
    live_bases: dict[str, int] = {}
    rejected: dict[str, str] = {}

    def rebase(self, ref: RegionRef) -> int:
        """Absolute live address of ``ref``.

        Raises :class:`RegionRejected` when the policy rejected the region and
        ``KeyError`` when the region is unknown to this map (a saved id that
        the live registry never enumerated is a REJECT, not a miss, so a
        ``KeyError`` here means a caller bug).
        """
        region_id = ref.region_id
        reason = self.rejected.get(region_id)
        if reason is not None:
            raise RegionRejected(f"region {region_id!r} was rejected: {reason}")
        if region_id not in self.live_bases:
            raise KeyError(region_id)
        return self.live_bases[region_id] + int(ref.offset)

    def is_rejected(self, region_id: str) -> bool:
        return region_id in self.rejected

    @property
    def accepted(self) -> frozenset[str]:
        return frozenset(self.deltas)


class RegionRegistry:
    """Sorted table of every live region across all providers.

    The registry enumerates its providers at construction and again on every
    :meth:`refresh`. Two live regions may not share a name and may not overlap;
    either is a ``ValueError`` at refresh because a word inside an overlap
    would classify ambiguously.
    """

    def __init__(self, providers: Sequence[RegionProvider]) -> None:
        self._providers: list[RegionProvider] = list(providers)
        self._specs: dict[str, RegionSpec] = {}
        self._live_bases: dict[str, int] = {}
        self._order: list[str] = []
        # Parallel arrays sorted by live base for bisect classification.
        self._sorted_bases: list[int] = []
        self._sorted_ends: list[int] = []
        self._sorted_ids: list[str] = []
        self.refresh()

    @property
    def providers(self) -> tuple[RegionProvider, ...]:
        return tuple(self._providers)

    def refresh(self) -> None:
        """Re-enumerate every provider and rebuild the classification table."""
        specs: dict[str, RegionSpec] = {}
        live_bases: dict[str, int] = {}
        order: list[str] = []
        for provider in self._providers:
            for spec, base in provider.enumerate():
                if spec.region_id in specs:
                    raise ValueError(
                        f"region {spec.region_id!r} registered twice "
                        f"(kind {specs[spec.region_id].kind!r} and {spec.kind!r})"
                    )
                if spec.nbytes <= 0:
                    raise ValueError(
                        f"region {spec.region_id!r} has nbytes={spec.nbytes}; "
                        "a live region must be non-empty"
                    )
                specs[spec.region_id] = spec
                live_bases[spec.region_id] = int(base)
                order.append(spec.region_id)

        ordered = sorted(order, key=lambda rid: (live_bases[rid], rid))
        sorted_bases: list[int] = []
        sorted_ends: list[int] = []
        # Region reaching furthest so far: the one a newcomer can overlap.
        furthest: Optional[str] = None
        for rid in ordered:
            base = live_bases[rid]
            end = base + specs[rid].nbytes
            if furthest is not None and base < sorted_ends[-1]:
                prev_base = live_bases[furthest]
                prev_end = prev_base + specs[furthest].nbytes
                raise ValueError(
                    f"regions {furthest!r} [{prev_base:#x}, {prev_end:#x}) and "
                    f"{rid!r} [{base:#x}, {end:#x}) overlap"
                )
            sorted_bases.append(base)
            # Ends are non-decreasing once no overlap exists, so the last end
            # is the furthest; keep the id for the error message.
            sorted_ends.append(end)
            furthest = rid

        self._specs = specs
        self._live_bases = live_bases
        self._order = order
        self._sorted_bases = sorted_bases
        self._sorted_ends = sorted_ends
        self._sorted_ids = ordered

    def snapshot(self) -> tuple[RegionSpec, ...]:
        """Every live region, in provider enumeration order, with
        ``base_at_save`` filled from the live base (the PIN target)."""
        return tuple(
            msgspec.structs.replace(
                self._specs[rid], base_at_save=self._live_bases[rid]
            )
            for rid in self._order
        )

    def live(self) -> dict[str, tuple[RegionSpec, int]]:
        """``region_id -> (spec, live base)`` for :meth:`RegionPlacementPolicy.build`."""
        return {rid: (self._specs[rid], self._live_bases[rid]) for rid in self._order}

    def live_base(self, region_id: str) -> int:
        return self._live_bases[region_id]

    def spec(self, region_id: str) -> RegionSpec:
        return self._specs[region_id]

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, region_id: object) -> bool:
        return region_id in self._specs

    def classify(self, word: int) -> Optional[RegionRef]:
        """Resolve a raw 64-bit word to a region, or ``None`` when it lies in
        no live range.

        Bisects the sorted live ``[base, base + nbytes)`` ranges only (fact 17).
        Interior offsets resolve (fact 9: kernels take ``kv + 32768`` as a
        plain word), so a hit anywhere inside a region yields
        ``RegionRef(region_id, word - base)``.
        """
        word = int(word)
        if word <= 0 or not self._sorted_bases:
            return None
        idx = bisect_right(self._sorted_bases, word) - 1
        if idx < 0:
            return None
        if word >= self._sorted_ends[idx]:
            return None
        return RegionRef(self._sorted_ids[idx], word - self._sorted_bases[idx])

    def relocation_for(
        self, saved: Sequence[RegionSpec], policy: RegionPlacementPolicy
    ) -> RelocationMap:
        """Bind the saved regions of an artifact to this registry's live table."""
        return policy.build(saved, self.live())

    def rebind_data(self, reloc: RelocationMap) -> None:
        """Fan ``rebind_data`` out to every provider (design section 7.3)."""
        for provider in self._providers:
            hook = getattr(provider, "rebind_data", None)
            if hook is not None:
                hook(reloc)


class RegionPlacementPolicy(ABC):
    """Per-kind disposition plus the shared binding rules (design section 6.4).

    :meth:`build` is implemented once here so both policies agree on what a
    match is: a saved region binds to the live region of the same *name*; sizes
    must match exactly; a pinned region must sit at ``base_at_save``; a saved
    region with no live counterpart is rejected. Failing closed per region is
    what lets the per-graph verdict stay a pure function of the map (design
    section 6.1, decision 4).
    """

    name: str = ""

    @abstractmethod
    def reserve(self, device_id: int) -> None:
        """Reserve address space before weights load (may be a no-op)."""

    @abstractmethod
    def mem_pool_for(self, kind: str) -> torch.cuda.MemPool | None:
        """The pool an owner of ``kind`` must allocate from, or ``None`` for
        the caching allocator."""

    @abstractmethod
    def graph_pool_id(self) -> tuple[int, int] | None:
        """Pool id for ``torch.cuda.graph(pool=...)``; ``None`` means the
        runner keeps using ``graph_pool_handle()``."""

    @abstractmethod
    def disposition(self, kind: str) -> Disposition:
        """Disposition for every region of ``kind``."""

    def build(
        self,
        saved: Sequence[RegionSpec],
        live: Mapping[str, tuple[RegionSpec, int]],
    ) -> RelocationMap:
        reloc = RelocationMap()
        for spec in saved:
            rid = spec.region_id
            disposition = self.disposition(spec.kind)
            if disposition is Disposition.REJECT:
                reloc.rejected[rid] = (
                    f"kind {spec.kind!r} is rejected by placement policy {self.name!r}"
                )
                continue
            entry = live.get(rid)
            if entry is None:
                reloc.rejected[rid] = (
                    f"saved region {rid!r} (kind {spec.kind!r}, {spec.nbytes} "
                    "bytes) has no live region of that name"
                )
                continue
            live_spec, live_base = entry
            live_base = int(live_base)
            if live_spec.kind != spec.kind:
                reloc.rejected[rid] = (
                    f"region {rid!r} kind changed: saved {spec.kind!r}, "
                    f"live {live_spec.kind!r}"
                )
                continue
            if live_spec.nbytes != spec.nbytes:
                reloc.rejected[rid] = (
                    f"region {rid!r} size changed: saved {spec.nbytes} bytes, "
                    f"live {live_spec.nbytes} bytes"
                )
                continue
            if disposition is Disposition.PIN:
                if live_base != spec.base_at_save:
                    reloc.rejected[rid] = (
                        f"pinned region {rid!r} landed at {live_base:#x} but was "
                        f"saved at {spec.base_at_save:#x}"
                    )
                    continue
                delta = 0
            else:
                delta = live_base - spec.base_at_save
            reloc.deltas[rid] = delta
            reloc.live_bases[rid] = live_base
        return reloc


class RelocatePolicy(RegionPlacementPolicy):
    """Default placement (design section 7.2): owners allocate exactly as
    today; every known kind is ``RELOCATE``; an unknown kind is ``REJECT``.

    Works under weight-cache ``torch_ipc`` because relocation only needs plain
    words (design section 10); ``cuIpcOpenMemHandle`` addresses are not stable
    (fact 7), so nothing is pinned here.
    """

    name = "relocate"

    def reserve(self, device_id: int) -> None:
        """No address space to reserve: the caching allocator places
        everything and the loader patches by delta."""
        return None

    def mem_pool_for(self, kind: str) -> torch.cuda.MemPool | None:
        return None

    def graph_pool_id(self) -> tuple[int, int] | None:
        return None

    def disposition(self, kind: str) -> Disposition:
        if kind in _KNOWN_KINDS:
            return Disposition.RELOCATE
        return Disposition.REJECT


class FixedVaArenaPolicy(RegionPlacementPolicy):
    """Pin SGLang-owned kinds at fixed virtual addresses (design section 7.2).

    One :class:`FixedArena` per pinned kind, reserved with
    ``cuMemAddressReserve`` at a requested address -- deterministic in fresh
    processes and after torch has initialized (fact 7) -- and exposed as
    ``torch.cuda.MemPool(no_split=True)``, the configuration that produced
    byte-identical addresses across processes (fact 13). Pinned kinds are
    ``PIN``, every other known kind is ``RELOCATE``, unknown kinds ``REJECT``.
    A reservation collision degrades *that kind* to ``RELOCATE`` with a logged
    warning; ``effective_placement`` records the outcome so the fingerprint can
    carry it.
    """

    name = "fixed_va"

    def __init__(
        self,
        *,
        arena_bases: Mapping[str, int],
        arena_sizes: Mapping[str, int],
        device_id: Optional[int] = None,
    ) -> None:
        bases = {str(k): int(v) for k, v in arena_bases.items()}
        sizes = {str(k): int(v) for k, v in arena_sizes.items()}
        if set(bases) != set(sizes):
            raise ValueError(
                "arena_bases and arena_sizes must name the same kinds: "
                f"bases={sorted(bases)} sizes={sorted(sizes)}"
            )
        unknown = sorted(set(bases) - _KNOWN_KINDS)
        if unknown:
            raise ValueError(f"unknown region kinds cannot be pinned: {unknown}")
        for kind in bases:
            if bases[kind] <= 0 or sizes[kind] <= 0:
                raise ValueError(
                    f"arena for kind {kind!r} needs a positive base and size, got "
                    f"base={bases[kind]:#x} size={sizes[kind]}"
                )
        self._arena_bases = bases
        self._arena_sizes = sizes
        self._device_id = None if device_id is None else int(device_id)
        self._arenas: dict[str, FixedArena] = {}
        self._reserved = False
        # Planned placement until reserve() runs; collisions rewrite entries.
        self.effective_placement: dict[str, Disposition] = {
            kind: Disposition.PIN for kind in bases
        }

    @property
    def pinned_kinds(self) -> tuple[str, ...]:
        return tuple(
            kind
            for kind in sorted(self._arena_bases)
            if self.effective_placement[kind] is Disposition.PIN
        )

    @property
    def arenas(self) -> Mapping[str, FixedArena]:
        return dict(self._arenas)

    def reserve(self, device_id: int) -> None:
        """Reserve one arena per pinned kind, right after ``set_device`` and
        before distributed init (the driver places ``cudaMalloc`` segments in
        the same high range, design section 7.2)."""
        device_id = int(device_id)
        if self._device_id is not None and self._device_id != device_id:
            raise ValueError(
                f"FixedVaArenaPolicy was built for device {self._device_id} but "
                f"reserve() was called for device {device_id}"
            )
        if self._reserved:
            raise RuntimeError("FixedVaArenaPolicy.reserve called twice")
        self._device_id = device_id
        for kind in sorted(self._arena_bases):
            base = self._arena_bases[kind]
            size = self._arena_sizes[kind]
            try:
                arena = FixedArena(
                    device_id=device_id,
                    size=size,
                    requested_address=base,
                    name=f"graph_serialization:{kind}",
                )
            except FixedArenaCollision as exc:
                logger.warning(
                    "FixedVaArenaPolicy: arena for kind %r at %#x (%d bytes) "
                    "collided (%s); degrading that kind to RELOCATE",
                    kind,
                    base,
                    size,
                    exc,
                )
                self.effective_placement[kind] = Disposition.RELOCATE
                continue
            self._arenas[kind] = arena
            self.effective_placement[kind] = Disposition.PIN
        self._reserved = True

    def mem_pool_for(self, kind: str) -> torch.cuda.MemPool | None:
        arena = self._arenas.get(kind)
        if arena is None:
            return None
        return arena.mem_pool()

    def graph_pool_id(self) -> tuple[int, int] | None:
        arena = self._arenas.get(RegionKind.POOL.value)
        if arena is None:
            return None
        return tuple(arena.mem_pool().id)

    def disposition(self, kind: str) -> Disposition:
        if kind not in _KNOWN_KINDS:
            return Disposition.REJECT
        return self.effective_placement.get(kind, Disposition.RELOCATE)

    def close(self) -> None:
        for kind in sorted(self._arenas):
            self._arenas.pop(kind).close()


# --------------------------------------------------------------------------
# v1 provider skeletons (design section 7.1). Each names its exact source of
# truth; ``enumerate`` is a documented stub in this draft.
# --------------------------------------------------------------------------


class BaseRegionProvider:
    """Shared shape of the v1 providers: a ``kind`` class attribute, a no-op
    ``rebind_data`` (the Protocol default) and a stubbed ``enumerate``."""

    kind: str = RegionKind.MISC.value

    def enumerate(self) -> list[tuple[RegionSpec, int]]:
        raise _not_implemented(
            f"{type(self).__name__}.enumerate", "region enumeration", "7.1"
        )

    def rebind_data(self, reloc: RelocationMap) -> None:
        """This owner stores no device pointers as data; nothing to rewrite."""
        return None


class WeightsProvider(BaseRegionProvider):
    """``weight:storage:<n>`` and ``weight:attr:<module>.<attr>`` regions.

    Source of truth (design section 7.1): walk
    ``model.named_parameters(remove_duplicate=False)`` plus
    ``model.named_buffers()`` and group by ``untyped_storage()`` (one region
    per distinct storage, which covers tied and view parameters and
    IPC-imported tensors: ``ipc_loader._set_module_tensor`` registers them as
    parameters or buffers, ``model._ipc_imported_tensors`` is only the GC
    pin); then walk every ``module.__dict__`` for plain-attribute tensors and
    call the ``QuantizeMethodBase.graph_visible_tensors()`` hook for tensors
    held on quant-method objects (marlin workspaces, trtllm SwiGLU params,
    hpc_ops scales, modelopt interleaved weights, cutlass MoE stride and
    pointer tables, the logits-processor gather buffer). Without the attribute
    walk those models are ``needs_recapture`` (fact 22). Under ``torch_ipc``
    the region bounds come from ``untyped_storage()``; the driver range is
    only a containment check (fact 22).
    """

    kind = RegionKind.WEIGHT.value

    def __init__(self, model: Any) -> None:
        self.model = model


class KVPoolProvider(BaseRegionProvider):
    """``kv:<buffer>``, ``req_to_token``, ``kv_ptrs:<table>``, ``mamba:<state>``.

    Source of truth (design section 7.1): a new ``KVCache.graph_visible_buffers()``
    on ``token_to_kv_pool`` (every K/V buffer, the ``k_data_ptrs`` /
    ``v_data_ptrs`` / ``data_ptrs`` tables, mamba state) plus the
    ``req_to_token_pool.req_to_token`` table. :meth:`rebind_data` rewrites the
    pointer tables in device memory (design section 7.3), which is why this is
    the one provider whose ``rebind_data`` is not a no-op.
    """

    kind = RegionKind.KV.value

    def __init__(self, token_to_kv_pool: Any, req_to_token_pool: Any) -> None:
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token_pool = req_to_token_pool

    def rebind_data(self, reloc: RelocationMap) -> None:
        raise _not_implemented(
            "KVPoolProvider.rebind_data",
            "rewriting the k_data_ptrs/v_data_ptrs tables by region delta",
            "7.3",
        )


class StaticBufferProvider(BaseRegionProvider):
    """``static:<slot>``, ``prefill_static:<name>``, ``shared_out:<vocab>``,
    ``bcg_out``, ``spec:<name>``, ``ragged:<n>``, ``prefix:<n>``.

    Source of truth (design section 7.1): the input-buffer pool canonicals and
    buffer-registry slots, the prefill static buffers, ``GraphSharedOutput`` and the
    BCG shared output, and the runner extras (design section 6.4).
    """

    kind = RegionKind.STATIC.value

    def __init__(
        self,
        *,
        buffer_registry: Any = None,
        prefill_statics: Any = None,
        shared_outputs: Sequence[Any] = (),
        bcg_output: Any = None,
    ) -> None:
        self.buffer_registry = buffer_registry
        self.prefill_statics = prefill_statics
        self.shared_outputs = tuple(shared_outputs)
        self.bcg_output = bcg_output


class AttnWorkspaceProvider(BaseRegionProvider):
    """``attn_ws:<backend>:<attr>[<i>]`` regions.

    Source of truth (design section 7.1): a new
    ``AttentionBackend.graph_visible_buffers()`` whose default walks attributes
    named ``cuda_graph_*`` plus the FlashInfer named workspace and the
    per-shape wrapper buffers (design section 6.4).
    """

    kind = RegionKind.ATTN_WS.value

    def __init__(self, attn_backends: Sequence[Any]) -> None:
        self.attn_backends = tuple(attn_backends)


class GraphPoolProvider(BaseRegionProvider):
    """``pool:seg:<i>`` regions: whole graph-pool segments.

    Source of truth (design section 7.1): ``torch.cuda.memory_snapshot(pool_id)``
    taken after the last capture and before any graph is dropped, one region
    per *whole* segment. Block granularity is impossible: most pool pointer
    words reference blocks already freed inside the pool at save time
    (fact 19). The snapshot excludes the cuBLAS workspace even when the GEMM
    runs inside the capture; it lives in the default pool. At load under
    ``RelocatePolicy`` the loader allocates ONE block of the summed segment
    sizes and carves per-segment sub-ranges, because each 2 MiB request would
    otherwise cost a 20 MiB caching-allocator segment (fact 19); under
    ``FixedVaArenaPolicy`` it maps the pinned pool arena instead.
    """

    kind = RegionKind.POOL.value

    def __init__(self, pool_id: tuple[int, int]) -> None:
        self.pool_id = tuple(pool_id)


class CublasWorkspaceProvider(BaseRegionProvider):
    """``cublas_ws:<stream ordinal>`` regions.

    Source of truth (design section 7.1): the caching-allocator blocks whose
    allocation stack (``torch.cuda.memory._record_memory_history`` plus
    ``memory_snapshot``) contains ``setWorkspaceForHandle`` (fact 14): torch
    allocates one 32 MiB workspace per (handle, stream) at handle creation, at
    an unstable address, and only split-K kernels reference it. At load the
    workspace for the replay stream does not exist until a GEMM runs on it, so
    the loader must create it first -- one tiny GEMM under that stream, or an
    allocate-and-pin -- before binding regions (fact 21).
    """

    kind = RegionKind.CUBLAS_WS.value


class CommTableProvider(BaseRegionProvider):
    """``comm:<group>:ca_v2:{table,slab:<rank>,mcast,push_counter}`` and
    ``comm:<group>:ca_v1:rank_data`` regions.

    Source of truth (design section 7.1): ``CommGraphState.graph_visible_regions()``
    of every communicator state of the group: ``graph_params``, the per-peer
    slabs, the multicast workspace and the rank-local ``_push_counter``
    (fact 17). Row contents are restored by ``CommGraphState``, not by
    ``rebind_data`` (design section 6.10).
    """

    kind = RegionKind.COMM_TABLE.value

    def __init__(self, states: Sequence[Any]) -> None:
        self.states = tuple(states)
