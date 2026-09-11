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
"""Graph codec: a live ``CUgraph`` to and from ``SerializedGraph`` (design
section 6.6).

Encoding walks the nodes of a captured graph in the deterministic topological
order the dedup mixin's ``graph_signature`` uses, reads every node's raw
parameter bytes and hands them to :class:`ParamScanner`, which turns each
device-pointer word into a ``RegionRef`` slot. Materializing rebuilds the graph
from scratch with ``cuGraphAddKernelNode`` and friends rather than patching a
live exec (fact 4: a graph rebuilt with functions re-resolved by name launches
and matches eager), so a loaded graph never depends on a template exec and the
loader, not the driver, checks kernel identity (fact 8).

Implemented and unit tested in this draft: :class:`ParamScanner` (pure),
:class:`EventTable` (pure) and the guarded driver calls of
:class:`LoadedCudaGraph`. ``GraphCodec.encode`` and ``GraphCodec.materialize``
are stubs whose docstrings list the exact driver-call sequence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

import msgspec
import torch

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:
    cuda_drv = None

from sglang.srt.model_executor.graph_serialization.format import (
    KernelIdentity,
    RegionRef,
    SerializedGraph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.graph_serialization.kernels import KernelResolver
    from sglang.srt.model_executor.graph_serialization.regions import (
        RegionRegistry,
        RelocationMap,
    )
    from sglang.srt.model_executor.graph_serialization.safety import SafetyPolicy

logger = logging.getLogger(__name__)

WORD_BYTES = 8

# Host virtual-address range in which every unknown pointer-like word of torch
# and cuBLAS parameters was dead host garbage (stack, anonymous mappings or
# padding) rejected by ``cuPointerGetAttribute``; zeroing them was safe in all
# 93 mutations (fact 14). The range is a constructor knob because it is an
# x86-64 Linux observation, not a driver guarantee.
DEFAULT_HOST_GARBAGE_RANGE = (0x5000_0000_0000, 0x8000_0000_0000)


def _not_implemented(where: str, what: str, section: str) -> NotImplementedError:
    return NotImplementedError(
        f"{where}: {what} is not implemented in this draft; see "
        f"DESIGN_cuda_graph_serialization.md section {section}"
    )


class ScanResult(msgspec.Struct, kw_only=True):
    """What :meth:`ParamScanner.scan` found in one node's parameter bytes.

    ``refs`` has one entry per 8-byte-aligned word (``None`` for non-pointer
    words); ``unknown_device_words`` lists the byte offsets of live device
    pointers that no provider registered, each of which makes the graph
    ``needs_recapture``; ``sanitized`` is ``raw`` with host-garbage words
    zeroed (fact 14) and everything else, including a partial trailing word,
    verbatim.
    """

    refs: list[Optional[RegionRef]]
    unknown_device_words: list[int]
    sanitized: bytes


class ParamScanner:
    """Classify the 8-byte-aligned words of a node's parameter bytes.

    Pointers are not separate kernel parameters (fact 2: cuBLAS passes one
    384-byte struct, torch TensorIterator kernels a 648-byte functor struct;
    nvjet passes an 1856-byte ``extra`` blob, fact 10), so the only way to
    find them is to scan aligned little-endian words and match them against
    tracked device ranges. Every device pointer of every probed kernel class
    appears as such a word, including the global address in qword 0 of a
    ``CUtensorMap`` (fact 9). The checks run in the design's order (design
    section 6.6):

    1. ``registry.classify(word)`` -> :class:`RegionRef`. Range based, because
       interior pointers such as ``kv + 32768`` are common (fact 9).
    2. Else ``is_device_pointer(word)`` (``cuPointerGetAttribute`` with
       ``CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`` in production): a live device
       address no provider registered. Recorded by byte offset in
       ``unknown_device_words`` and kept verbatim; the codec turns it into a
       ``needs_recapture`` verdict naming node and offset.
    3. Else a word inside ``host_garbage_range`` is dead host garbage and is
       zeroed in ``sanitized`` so artifacts are byte-stable across processes
       (fact 14).
    4. Else the word is a scalar and is kept verbatim.

    ``/proc/self/maps`` is never consulted: the ``slot_bytes`` scalar
    ``0x400000`` coincides with the non-PIE python3 ELF base (fact 17), so a
    host-mapping based classification would misfile plain scalars. The zero
    word is never queried: the driver rejects the null pointer, and zero is
    the most common padding value.
    """

    def __init__(
        self,
        registry: RegionRegistry,
        is_device_pointer: Callable[[int], bool],
        *,
        host_garbage_range: tuple[int, int] = DEFAULT_HOST_GARBAGE_RANGE,
    ) -> None:
        lo, hi = host_garbage_range
        if lo < 0 or hi < lo:
            raise ValueError(
                f"ParamScanner: host_garbage_range must be a non-negative half-open "
                f"interval, got {host_garbage_range!r}"
            )
        self._registry = registry
        self._is_device_pointer = is_device_pointer
        self._host_garbage_range = (int(lo), int(hi))

    @property
    def host_garbage_range(self) -> tuple[int, int]:
        return self._host_garbage_range

    def scan(self, raw: bytes) -> ScanResult:
        refs: list[Optional[RegionRef]] = []
        unknown: list[int] = []
        sanitized = bytearray(raw)
        garbage_lo, garbage_hi = self._host_garbage_range
        for offset in range(0, len(raw) - WORD_BYTES + 1, WORD_BYTES):
            word = int.from_bytes(raw[offset : offset + WORD_BYTES], "little")
            ref = self._registry.classify(word)
            if ref is not None:
                refs.append(ref)
                continue
            refs.append(None)
            if word == 0:
                continue
            if self._is_device_pointer(word):
                unknown.append(offset)
                continue
            if garbage_lo <= word < garbage_hi:
                sanitized[offset : offset + WORD_BYTES] = bytes(WORD_BYTES)
        return ScanResult(
            refs=refs, unknown_device_words=unknown, sanitized=bytes(sanitized)
        )


def driver_is_device_pointer(word: int) -> bool:
    """``cuPointerGetAttribute(RANGE_START_ADDR)`` accepts ``word``.

    The production ``is_device_pointer`` for :class:`ParamScanner`. Any
    address the driver knows (device, managed or host-registered) answers
    ``True``; that is the conservative direction, because an unknown live
    address only costs a recapture.
    """
    if cuda_drv is None:
        raise _not_implemented(
            "driver_is_device_pointer",
            "cuPointerGetAttribute without cuda.bindings",
            "6.6",
        )
    err, _ = cuda_drv.cuPointerGetAttribute(
        cuda_drv.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        int(word),
    )
    return int(err) == 0


class EventTable:
    """``role -> live external event`` for rebinding event nodes at load
    (design section 6.6).

    Event nodes are bound by role, never by handle: ``torch.cuda.Event(
    external=True)`` produces ``EVENT_RECORD`` / ``WAIT_EVENT`` nodes holding
    process-local ``CUevent`` handles (fact 15). The only v1 role is
    ``EVENT_ROLE_METADATA_PREP_DONE``, bound to
    ``DecodeCudaGraphRunner.in_graph_metadata_prep_done``: one event per
    runner, shared by every decode shape, created lazily inside the captured
    body today. On a ``capture=False`` runner the loader creates it with
    ``make_external_event()`` and primes it with one host-side ``record()``
    plus ``synchronize()`` so the ``CUevent`` exists.

    :meth:`handle` refuses handle ``0`` (fact 18): torch creates the
    ``CUevent`` lazily, ``cuda_event`` is ``0`` until the first ``record()``,
    and ``cuGraphAddEventRecordNode`` rejects it with
    ``CUDA_ERROR_INVALID_VALUE``. Bind at the graph level before instantiate;
    exec-level rebinding does not change the ``CUgraph`` and must be redone
    after any reinstantiate or exec update.
    """

    def __init__(self, roles: Mapping[str, object]) -> None:
        self._roles: dict[str, object] = dict(roles)

    def roles(self) -> tuple[str, ...]:
        return tuple(self._roles)

    def __contains__(self, role: object) -> bool:
        return role in self._roles

    def handle(self, role: str) -> int:
        try:
            event = self._roles[role]
        except KeyError:
            raise ValueError(
                f"EventTable: unknown event role {role!r}; known roles: "
                f"{sorted(self._roles)}"
            ) from None
        handle = int(event.cuda_event)
        if handle == 0:
            raise ValueError(
                f"EventTable: event for role {role!r} has CUevent handle 0; torch "
                "creates the event lazily, record() it once before binding (fact 18)"
            )
        return handle


class LoadedCudaGraph:
    """A materialized graph that duck-types ``torch.cuda.CUDAGraph`` for the
    backend tables (design section 6.6).

    ``raw_graph`` is the rebuilt ``CUgraph``; ``exec`` is its own
    ``CUgraphExec``, or ``None`` when the dedup registry is enabled: the
    registry instantiates its own execs from :meth:`raw_cuda_graph` and owns
    them, so replay goes through the registry (design section 6.1, decision
    3). Every driver call is guarded so the class is importable and
    constructible without ``cuda.bindings``.
    """

    def __init__(self, raw_graph: int, exec: Optional[int] = None) -> None:
        self.raw_graph = int(raw_graph)
        self.exec: Optional[int] = None if exec is None else int(exec)

    def replay(self) -> None:
        """``cuGraphLaunch`` on ``torch.cuda.current_stream()``."""
        if self.exec is None:
            raise RuntimeError(
                "LoadedCudaGraph.replay: this graph owns no exec (the dedup "
                "registry owns the executables); replay it through the registry"
            )
        if cuda_drv is None:
            raise _not_implemented(
                "LoadedCudaGraph.replay", "cuGraphLaunch without cuda.bindings", "6.6"
            )
        stream = torch.cuda.current_stream().cuda_stream
        checkCudaErrors(cuda_drv.cuGraphLaunch(self.exec, stream))

    def raw_cuda_graph(self) -> int:
        """The ``CUgraph`` handle, for the dedup registry when templates are
        enabled."""
        return self.raw_graph

    def reset(self) -> None:
        """Destroy the exec, then the graph. Idempotent."""
        if self.exec is None and self.raw_graph == 0:
            return
        if cuda_drv is None:
            raise _not_implemented(
                "LoadedCudaGraph.reset",
                "cuGraphExecDestroy / cuGraphDestroy without cuda.bindings",
                "6.6",
            )
        if self.exec is not None:
            checkCudaErrors(cuda_drv.cuGraphExecDestroy(self.exec))
            self.exec = None
        if self.raw_graph != 0:
            checkCudaErrors(cuda_drv.cuGraphDestroy(self.raw_graph))
            self.raw_graph = 0


class GraphCodec:
    """Encode a live ``CUgraph`` into a ``SerializedGraph`` and back (design
    section 6.6).

    ``is_device_pointer`` defaults to :func:`driver_is_device_pointer`; tests
    inject a fake. Both entry points are stubs in this draft.
    """

    def __init__(
        self,
        *,
        is_device_pointer: Optional[Callable[[int], bool]] = None,
        host_garbage_range: tuple[int, int] = DEFAULT_HOST_GARBAGE_RANGE,
    ) -> None:
        self._is_device_pointer = is_device_pointer or driver_is_device_pointer
        self._host_garbage_range = host_garbage_range

    def scanner_for(self, registry: RegionRegistry) -> ParamScanner:
        return ParamScanner(
            registry,
            self._is_device_pointer,
            host_garbage_range=self._host_garbage_range,
        )

    def encode(
        self,
        raw_graph: int,
        *,
        registry: RegionRegistry,
        resolver: KernelResolver,
        policy: SafetyPolicy,
        event_roles: Mapping[int, str],
    ) -> SerializedGraph:
        """SAVE: one ``CUgraph`` to a pointer-free record.

        Driver-call sequence (design section 6.6):

        1. ``cuGraphGetNodes(graph, 0)`` for the count, then
           ``cuGraphGetNodes(graph, n)``; same two-call idiom for
           ``cuGraphGetEdges``. Nodes are renumbered into the deterministic
           topological order of the dedup mixin's ``graph_signature`` so
           ``SerializedGraph.nodes`` and ``signature`` agree.
        2. Per node, ``cuGraphNodeGetType``. Child-graph, host, mem-alloc,
           mem-free, conditional and batch-memop nodes are refused through
           ``policy.node_type_verdict`` and make the graph ``needs_recapture``.
        3. Kernel nodes: ``cuGraphKernelNodeGetParams``; the name through the
           dedup mixin's ``kernel_name`` (``cuKernelGetName`` / ``cuFuncGetName``,
           ``func:<int>`` fallback refused by ``policy.kernel_verdict``);
           ``resolver.identify(func, kern)`` for the ``KernelIdentity``;
           parameter bytes either per formal parameter through the
           ``cuFuncGetParamInfo`` ``(offset, size)`` layout (fact 2) or, when
           ``extra`` is set, the single blob addressed by
           ``CU_LAUNCH_PARAM_BUFFER_POINTER`` / ``CU_LAUNCH_PARAM_BUFFER_SIZE``
           (fact 10, recorded with ``launch_form="extra"``); every non-default
           launch attribute through ``cuGraphKernelNodeGetAttribute``
           (mandatory, fact 19); function attributes through
           ``cuFuncGetAttribute``; the bytes through :class:`ParamScanner`,
           each ``RegionRef`` becoming a ``PointerSlot`` and each unknown
           device word a ``needs_recapture`` reason naming node and offset;
           ``policy.looks_like_philox_state`` refuses RNG kernels (fact 16).
        4. Memcpy and memset nodes: ``cuGraphMemcpyNodeGetParams`` /
           ``cuGraphMemsetNodeGetParams`` raw structs with their ``srcDevice``,
           ``dstDevice`` and ``dst`` fields slotted (``PARAM_MEMCPY_SRC``,
           ``PARAM_MEMCPY_DST``, ``PARAM_MEMSET_DST``).
        5. Event record and wait nodes: ``cuGraphEventRecordNodeGetEvent`` /
           ``cuGraphEventWaitNodeGetEvent`` mapped through ``event_roles``
           (``CUevent handle -> role``); an unmapped handle is
           ``needs_recapture`` (fact 15).
        6. Empty nodes pass through; edges are re-indexed into topological
           order; ``signature`` is the sha256 of the pointer-free signature.
        """
        raise _not_implemented(
            "GraphCodec.encode",
            "walking a live CUgraph (cuGraphGetNodes / cuGraphGetEdges, kernel "
            "params, launch and function attributes, memcpy / memset / event "
            "nodes)",
            "6.6",
        )

    def materialize(
        self,
        graph: SerializedGraph,
        *,
        kernels: Sequence[KernelIdentity],
        reloc: RelocationMap,
        resolver: KernelResolver,
        events: EventTable,
        device_ctx: int,
    ) -> LoadedCudaGraph:
        """LOAD: rebuild ``graph`` from scratch and instantiate it.

        Driver-call sequence (design section 6.6):

        1. Refuse a graph whose ``verdict`` is not ``serializable``.
        2. ``cuGraphCreate(0)``.
        3. Copy ``param_bytes``; for every ``PointerSlot`` write
           ``reloc.rebase(slot.ref)`` as a little-endian 8-byte word at its
           offset (``RegionRejected`` aborts the shape).
        4. Kernel nodes: ``resolver.resolve(kernels[node.identity])`` for the
           function, ``resolver.apply_func_attrs`` (fact 8), then
           ``cuGraphAddKernelNode`` with the rebased bytes in
           ``kernel_params`` form; the driver accepts that form for
           ``extra``-form originals (fact 10). Then
           ``cuGraphKernelNodeSetAttribute`` for every recorded launch
           attribute: mandatory, an nvjet 2-CTA node without its
           ``CLUSTER_DIMENSION`` fails instantiate with error 912 (fact 19).
        5. Memcpy / memset nodes: ``cuGraphAddMemcpyNode`` /
           ``cuGraphAddMemsetNode`` with the rebased raw struct and
           ``device_ctx``.
        6. Event nodes: ``cuGraphAddEventRecordNode`` /
           ``cuGraphAddEventWaitNode`` with ``events.handle(node.role)``, bound
           at the graph level before instantiate (fact 18).
        7. Empty nodes: ``cuGraphAddEmptyNode``.
        8. ``cuGraphAddDependencies`` for ``edges`` (nodes were added in
           topological order, so indices map one to one).
        9. Verify every kernel node's name (``cuFuncGetName`` of the resolved
           function) against the identity table and the recomputed signature;
           the driver never checks identity (fact 8).
        10. ``cuGraphInstantiate(graph, 0)``; with the dedup registry enabled,
            skip and return ``exec=None`` so the registry instantiates.
        """
        raise _not_implemented(
            "GraphCodec.materialize",
            "cuGraphCreate / cuGraphAddKernelNode / cuGraphKernelNodeSetAttribute "
            "/ cuGraphAddDependencies / cuGraphInstantiate",
            "6.6",
        )
