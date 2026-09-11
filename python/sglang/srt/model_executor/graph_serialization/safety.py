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
"""Fail-closed safety ladder for saved graphs (design section 6.11).

Six rungs: (1) a save-side self-check that re-materializes each exported graph
with the identity relocation and compares its replay to the live graph; (2)
exact fingerprint equality; (3) per-shape verdict agreement across the CPU
group before any import; (4) an optional shadow verify that recaptures one or
every loaded shape and diffs it node by node; (5) a smoke replay at the
smallest shape per loaded phase; (6) a strict mode for CI that turns any
fallback into a startup failure and logs the :class:`CoverageReport`.

This module holds the pure per-kernel and per-node verdict rules used at save
(:class:`SafetyPolicy`), the :class:`CoverageReport` record, and stubs for the
two rungs that need a device: :class:`SelfCheck` (rung 1) and
:class:`ShadowVerifier` (rung 4). Rungs 2, 3 and 5 live in ``fingerprint``,
``materializer`` and the lifecycle component respectively.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import msgspec

from sglang.srt.model_executor.graph_serialization.format import (
    GraphVerdict,
    SerializedGraph,
    ShapeArtifact,
)
from sglang.srt.model_executor.graph_serialization.plan import VerifyMode

logger = logging.getLogger(__name__)

# Kernel name prefixes refused at save (design section 6.5). The parameters of
# these kernels hold process-local handles no region provider can own.
REFUSED_KERNEL_PREFIXES = (
    "ncclDevKernel",
    "ncclKernel",
    "nvshmemi_",
    "cross_device_reduce_",
)

# Graph node types the v1 codec does not serialize (design section 6.6).
REFUSED_NODE_TYPES = (
    "child_graph",
    "host",
    "mem_alloc",
    "mem_free",
    "conditional",
    "batch_memop",
)

# ``cuda_graph_dedup_mixin.kernel_name`` falls back to ``func:<CUfunction>``
# when neither ``cuKernelGetName`` nor ``cuFuncGetName`` answers. The handle is
# process-specific and is not an identity.
FUNC_FALLBACK_PREFIX = "func:"

# ``sizeof(at::PhiloxCudaState)``: two 8-byte payload words (``seed_``,
# ``offset_``), a ``uint32_t offset_intragraph_`` and a ``bool captured_``
# padded to 8-byte alignment.
PHILOX_STATE_NBYTES = 24

_REFUSED_PREFIX_REASONS = {
    "ncclDevKernel": (
        "NCCL device kernel; the ncclComm_t handle in its parameters is "
        "process-local and not relocatable (design section 6.10)"
    ),
    "ncclKernel": (
        "NCCL device kernel; the ncclComm_t handle in its parameters is "
        "process-local and not relocatable (design section 6.10)"
    ),
    "nvshmemi_": (
        "NVSHMEM kernel; references the NVSHMEM symmetric heap, which no region "
        "provider owns (DeepEP, design section 3)"
    ),
    "cross_device_reduce_": (
        "custom all-reduce v1 kernel; IPC peer signals and rank_data are "
        "re-exchanged per process and not relocatable (design section 3)"
    ),
}

_REFUSED_NODE_TYPE_REASONS = {
    "child_graph": "child graph nodes are not flattened by the v1 codec",
    "host": "host nodes call a process-local host function pointer",
    "mem_alloc": "graph-owned allocations have no region owner",
    "mem_free": "graph-owned frees have no region owner",
    "conditional": "conditional nodes carry a process-local handle and body graphs",
    "batch_memop": "batch memory operations wait on process-local addresses",
}

# ``CUgraphNodeType`` spellings the driver binding reports, mapped onto the
# canonical names of ``REFUSED_NODE_TYPES``.
_NODE_TYPE_ALIASES = {
    "graph": "child_graph",
    "batch_mem_op": "batch_memop",
}
_NODE_TYPE_ENUM_PREFIX = "CU_GRAPH_NODE_TYPE_"


def _not_implemented(where: str, what: str, section: str) -> NotImplementedError:
    return NotImplementedError(
        f"{where}: {what} is not implemented in this draft; see "
        f"DESIGN_cuda_graph_serialization.md section {section}"
    )


def is_func_fallback_name(name: str) -> bool:
    """``True`` for the ``func:<int>`` names ``kernel_name`` emits when the
    driver has no name for a handle."""
    return (
        name.startswith(FUNC_FALLBACK_PREFIX)
        and name[len(FUNC_FALLBACK_PREFIX) :].isdigit()
    )


def canonical_node_type(node_type: str) -> str:
    """Normalize a node type spelling: ``CU_GRAPH_NODE_TYPE_GRAPH`` and
    ``child_graph`` both become ``child_graph``."""
    text = node_type.strip()
    if text.upper().startswith(_NODE_TYPE_ENUM_PREFIX):
        text = text[len(_NODE_TYPE_ENUM_PREFIX) :]
    text = text.lower()
    return _NODE_TYPE_ALIASES.get(text, text)


def _pointer_like(word: int) -> bool:
    # A per-generator seed / offset word is an 8-byte allocator-owned device
    # word: aligned, non-zero, inside the 48-bit canonical user VA range.
    return word != 0 and word % 8 == 0 and 0x10000 <= word < (1 << 48)


class SafetyPolicy:
    """Save-side refusal rules (design sections 6.5 and 6.11).

    Every method returns a human-readable reason for a refusal or ``None``
    when the input is acceptable; the codec collects reasons into the graph's
    ``verdict_reason``. ``strict`` is rung 6: :meth:`check_fallback` turns a
    fallback into a startup failure instead of a warning.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        refused_prefixes: Sequence[str] = REFUSED_KERNEL_PREFIXES,
    ) -> None:
        self.strict = strict
        self._refused_prefixes: tuple[str, ...] = tuple(refused_prefixes)

    @property
    def refused_prefixes(self) -> tuple[str, ...]:
        return self._refused_prefixes

    def kernel_verdict(self, name: str) -> str | None:
        """Reason a kernel name is refused, or ``None``.

        Refused: the ``func:<int>`` fallback (a process-specific handle, never
        a durable identity) and every name starting with one of the refused
        prefixes (NCCL, NVSHMEM and custom all-reduce v1 kernels, whose
        parameters hold process-local handles; design section 6.5).
        """
        if is_func_fallback_name(name):
            return (
                f"kernel name unavailable: {name!r} is the process-specific "
                "CUfunction handle fallback of kernel_name(), not a durable identity"
            )
        for prefix in self._refused_prefixes:
            if name.startswith(prefix):
                why = _REFUSED_PREFIX_REASONS.get(prefix, "refused kernel prefix")
                return f"kernel {name!r} matches refused prefix {prefix!r}: {why}"
        return None

    def node_type_verdict(self, node_type: str) -> str | None:
        """Reason a graph node type is refused, or ``None``.

        Accepts the canonical names of :data:`REFUSED_NODE_TYPES` and the
        driver's ``CU_GRAPH_NODE_TYPE_*`` spellings (design section 6.6).
        """
        canonical = canonical_node_type(node_type)
        if canonical in REFUSED_NODE_TYPES:
            why = _REFUSED_NODE_TYPE_REASONS.get(canonical, "unsupported node type")
            return f"node type {node_type!r} is not serializable: {why}"
        return None

    def looks_like_philox_state(self, raw: bytes) -> bool:
        """Heuristic: ``raw`` embeds a captured-form ``at::PhiloxCudaState``.

        Implemented as a documented heuristic, not a stub. RNG kernels bake a
        ``PhiloxCudaState`` whose ``seed_`` and ``offset_`` payloads point at
        per-generator, per-graph words that ``CUDAGraph.replay()`` writes in
        its prologue; a raw ``cuGraphLaunch`` repeats identical random numbers
        (fact 16), so such graphs are refused at save.

        Under capture torch always emits the captured form: at some 8-byte
        aligned offset, two distinct pointer-like words (``seed_.ptr``,
        ``offset_.ptr``), a ``uint32_t offset_intragraph_`` (any value), the
        byte ``captured_ == 1`` and three zero padding bytes. The check is
        window based, not size based, because the state is one field of a
        larger parameter struct. False positives cost one recaptured shape;
        false negatives would be silent-output bugs, so the check errs toward
        refusing.
        """
        limit = len(raw) - PHILOX_STATE_NBYTES
        for offset in range(0, limit + 1, 8):
            if raw[offset + 20] != 1 or raw[offset + 21 : offset + 24] != b"\0\0\0":
                continue
            seed = int.from_bytes(raw[offset : offset + 8], "little")
            counter = int.from_bytes(raw[offset + 8 : offset + 16], "little")
            if seed == counter:
                continue
            if _pointer_like(seed) and _pointer_like(counter):
                return True
        return False

    def graph_verdict(self, reasons: Sequence[str]) -> tuple[GraphVerdict, str]:
        """Fold the reasons collected for one graph into its verdict."""
        for reason in reasons:
            if reason:
                return GraphVerdict.NEEDS_RECAPTURE, reason
        return GraphVerdict.SERIALIZABLE, ""

    def check_fallback(self, reason: str) -> None:
        """Rung 6: log a fallback, or fail startup in strict mode."""
        if self.strict:
            raise RuntimeError(f"strict cuda-graph cache mode: {reason}")
        logger.warning("[CudaGraph][serialization] fallback: %s", reason)


class CoverageReport(msgspec.Struct, frozen=True, kw_only=True):
    """What the save or load did, logged on rank 0 and written into the
    artifact (design section 6.11, rung 6).

    ``graphs_by_verdict`` counts graphs per ``GraphVerdict`` value;
    ``reasons`` maps a shape label to its first refusal reason;
    ``slots_by_kind`` counts pointer slots per ``RegionKind``;
    ``kernels_by_provider`` counts resolved kernels per provider ``name``;
    ``harvest_forwards`` is the number of throwaway captures the live-harvest
    tier ran.
    """

    graphs_by_verdict: dict[str, int] = {}
    reasons: dict[str, str] = {}
    slots_by_kind: dict[str, int] = {}
    kernels_by_provider: dict[str, int] = {}
    harvest_forwards: int = 0


class SelfCheck:
    """Rung 1: save-side self-check without an extra forward (design section
    6.11).

    Re-materialize each exported graph through the codec with the *identity*
    relocation (every region's delta is zero), instantiate it, replay it on the
    capture-time dummy batch and compare the outputs bitwise with the live
    graph's replay. Catches missed pointer slots and codec bugs before anything
    is written. Stub in this draft.
    """

    def __init__(self, *, codec: Any, resolver: Any) -> None:
        self._codec = codec
        self._resolver = resolver

    def check(
        self,
        artifact: ShapeArtifact,
        *,
        live_graph: Any,
        live_output: Any,
        events: Any,
    ) -> None:
        raise _not_implemented(
            "SelfCheck.check",
            "identity-relocation re-materialization and bitwise replay comparison",
            "6.11",
        )


class ShadowVerifier:
    """Rung 4: ``--cuda-graph-cache-verify none | shadow-one | shadow-all``
    (design section 6.11).

    Recapture the selected loaded shapes and diff each against the loaded
    graph node by node: kernel names, grid, block, shared memory, launch
    attributes, edges, pointer slots as region references and parameter bytes
    with host-garbage words masked (fact 14). :meth:`select_shapes` is the
    pure mode logic; :meth:`diff` is a stub in this draft.
    """

    def __init__(self, mode: VerifyMode | str = VerifyMode.NONE) -> None:
        self.mode = VerifyMode(mode)

    def select_shapes(self, shape_labels: Sequence[str]) -> tuple[str, ...]:
        """Which loaded shapes to recapture: none, the first label given
        (runners capture largest first, so that is the largest shape), or all."""
        if self.mode is VerifyMode.NONE:
            return ()
        if self.mode is VerifyMode.SHADOW_ONE:
            return (shape_labels[0],) if shape_labels else ()
        return tuple(shape_labels)

    def diff(
        self,
        loaded: SerializedGraph,
        recaptured: SerializedGraph,
    ) -> list[str]:
        """Node-by-node differences between a loaded and a recaptured graph;
        an empty list means they agree."""
        raise _not_implemented(
            "ShadowVerifier.diff",
            "node-by-node comparison of a loaded and a recaptured graph",
            "6.11",
        )
