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
"""Artifact format for serialized CUDA graphs (design section 6.3).

Every record is a ``msgspec.Struct`` so a rank bundle round-trips through
msgpack (bulk) and JSON (manifests) with no hand-written codec. Two rules hold
for everything in this module:

* The artifact never stores a bare device pointer. Every pointer-valued word of
  a captured node is a :class:`RegionRef` (owner-assigned region id plus byte
  offset). ``RegionSpec.base_at_save`` is diagnostics and the fixed-VA ``PIN``
  target only; it is never a correctness input at load.
* Kernel identity is ``(container sha256, kernel name)`` when the container
  bytes exist on disk or were captured at save. Kernels with no bytes on disk
  (cuBLASLt ``nvjet_*``, CuTe-DSL) carry a live-only identity made of the
  owning module's name-set digest, the kernel name and its function attributes.

This module is importable without ``torch``: the runner ``ShapeKey`` is only
referenced through :class:`ShapeKeyRecord`.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import msgspec

from sglang.srt.model_executor.graph_serialization.fingerprint import (
    GraphArtifactFingerprint,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.runner.shape_key import ShapeKey

FORMAT_VERSION = 1

# Launch forms a kernel node was recorded with (fact 10: nvjet and cuDNN GEMMs
# pass one blob through CU_LAUNCH_PARAM_BUFFER_POINTER; the driver accepts a
# ``kernel_params``-form replacement when the node is rebuilt).
LAUNCH_FORM_KERNEL_PARAMS = "kernel_params"
LAUNCH_FORM_EXTRA = "extra"

# The only external event a v1 artifact may carry: the decode runner's
# ``in_graph_metadata_prep_done`` record (facts 15 and 18). Any other event
# handle in a saved graph is a ``needs_recapture`` verdict.
EVENT_ROLE_METADATA_PREP_DONE = "metadata_prep_done"

# ``PointerSlot.param`` values below zero address pointer words that do not
# live in a formal kernel parameter.
PARAM_EXTRA_BLOB = -1
PARAM_MEMCPY_SRC = -2
PARAM_MEMCPY_DST = -3
PARAM_MEMSET_DST = -4


class RegionKind(str, Enum):
    """Owner classes of graph-visible device memory (design section 7.1)."""

    WEIGHT = "weight"
    KV = "kv"
    REQ_TO_TOKEN = "req_to_token"
    STATIC = "static"
    ATTN_WS = "attn_ws"
    POOL = "pool"
    SHARED_OUT = "shared_out"
    BCG_OUT = "bcg_out"
    BCG_BRIDGE = "bcg_bridge"
    COMM_TABLE = "comm_table"
    CUBLAS_WS = "cublas_ws"
    SPEC = "spec"
    MISC = "misc"


class GraphVerdict(str, Enum):
    """Per-graph classification taken at save and re-checked at load."""

    SERIALIZABLE = "serializable"
    NEEDS_RECAPTURE = "needs_recapture"


class RegionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One named, contiguous device range an owner registered.

    ``region_id`` is owner-assigned and stable across processes, for example
    ``weight:storage:17``, ``kv:k_buffer:3``, ``static:input_ids``,
    ``pool:seg:2`` or ``cublas_ws:0``.
    """

    region_id: str
    kind: str
    nbytes: int
    alignment: int = 256
    base_at_save: int = 0
    dtype: Optional[str] = None
    shape: Optional[tuple[int, ...]] = None
    stride: Optional[tuple[int, ...]] = None


class RegionRef(msgspec.Struct, frozen=True, array_like=True):
    """A device address expressed as ``(region, offset)``.

    Interior offsets are common (fact 9): kernels take ``kv + 32768`` or
    ``workspace + 9472`` as plain words, so relocation is range based.
    """

    region_id: str
    offset: int = 0


class PointerSlot(msgspec.Struct, frozen=True, array_like=True):
    """Where a pointer word sits inside a node's parameter bytes."""

    node: int
    param: int
    byte_offset: int
    ref: RegionRef


class KernelIdentity(msgspec.Struct, frozen=True, kw_only=True):
    """Durable identity of a launched kernel (design section 6.5).

    ``image_sha256`` is set when the container bytes exist (tier 1 and 2
    providers); ``None`` marks a live-only identity that must be harvested
    from live handles at load. ``func_attrs`` are ``(CU_FUNC_ATTRIBUTE_*,
    value)`` pairs; they live on the CUfunction and must be re-applied after
    re-resolution (fact 8). ``param_layout`` is ``(offset, size)`` per formal
    parameter from ``cuFuncGetParamInfo`` (fact 2).
    """

    name: str
    image_sha256: Optional[str] = None
    module_names_digest: str = ""
    module_kernel_count: int = 0
    via_library: bool = True
    func_attrs: tuple[tuple[int, int], ...] = ()
    param_layout: tuple[tuple[int, int], ...] = ()

    @property
    def live_only(self) -> bool:
        return self.image_sha256 is None


class KernelImage(msgspec.Struct, frozen=True, kw_only=True):
    """A content-addressed kernel container persisted beside the bundles."""

    image_sha256: str
    provider: str
    source: str
    kernel_names: tuple[str, ...]
    nbytes: int


class KernelNode(msgspec.Struct, frozen=True, kw_only=True, tag="kernel"):
    """A ``CU_GRAPH_NODE_TYPE_KERNEL`` node.

    ``launch_attrs`` holds every non-default ``CUlaunchAttribute`` as
    ``(CUlaunchAttributeID, raw value)``. Recording them is mandatory: an
    nvjet 2-CTA node rebuilt without its ``CLUSTER_DIMENSION`` attribute fails
    instantiate with error 912 (fact 19).
    """

    identity: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int
    launch_form: str
    params_off: int
    params_len: int
    launch_attrs: tuple[tuple[int, bytes], ...] = ()


class MemcpyNode(msgspec.Struct, frozen=True, kw_only=True, tag="memcpy"):
    """A memcpy node; ``raw`` is the ``CUDA_MEMCPY3D`` struct with its pointer
    fields registered as :class:`PointerSlot` entries."""

    raw: bytes


class MemsetNode(msgspec.Struct, frozen=True, kw_only=True, tag="memset"):
    raw: bytes


class EventNode(msgspec.Struct, frozen=True, kw_only=True, tag="event"):
    """An event record or wait node bound by role at load, never by handle."""

    kind: str
    role: str


class EmptyNode(msgspec.Struct, frozen=True, tag="empty"):
    pass


Node = Union[KernelNode, MemcpyNode, MemsetNode, EventNode, EmptyNode]


class SerializedGraph(msgspec.Struct, frozen=True, kw_only=True):
    """One ``CUgraph`` with its pointer words replaced by region references.

    ``nodes`` are in the deterministic topological order the dedup registry's
    ``graph_signature`` uses; ``param_bytes`` are verbatim with dead host
    garbage words zeroed (fact 14); ``signature`` is the sha256 of the
    pointer-free signature so a loader can compare kernel names itself
    (``cuGraphExecUpdate`` never does, fact 8).
    """

    nodes: tuple[Node, ...]
    edges: tuple[tuple[int, int], ...]
    param_bytes: bytes
    slots: tuple[PointerSlot, ...]
    signature: str
    verdict: GraphVerdict = GraphVerdict.SERIALIZABLE
    verdict_reason: str = ""


class OutputSchema(msgspec.Struct, frozen=True, kw_only=True):
    """Shape of the Python object a backend returns from ``replay``.

    ``kind`` is one of ``logits_processor_output``, ``pp_proxy``, ``tensor``,
    ``tuple``, ``list`` or ``none``. Tensors are views into regions and are
    rebuilt at load with explicit shape, stride and dtype (fact 13).
    """

    kind: str
    fields: dict[str, OutputSchema] = {}
    items: tuple[OutputSchema, ...] = ()
    tensor: Optional[RegionRef] = None
    shape: Optional[tuple[int, ...]] = None
    stride: Optional[tuple[int, ...]] = None
    dtype: Optional[str] = None


class ArgSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One argument of a breakable-graph break site (design section 6.9).

    ``kind`` is ``tensor``, ``scalar``, ``none``, ``seq``, ``module`` or
    ``device``. ``module_path`` names a submodule in ``model.named_modules()``
    for ``kind="module"`` arguments (EP-MoE ``self``, Inkling
    ``prev_mlp_sconv``, MiniMax ``attention``). Dict-valued arguments have no
    ``ArgSpec``; such sites are ``needs_recapture`` in v1.
    """

    kind: str
    tensor: Optional[OutputSchema] = None
    scalar: Union[int, float, bool, str, None] = None
    items: tuple[ArgSpec, ...] = ()
    module_path: Optional[str] = None


class BreakSiteRecord(msgspec.Struct, frozen=True, kw_only=True):
    """A rebuildable description of one ``eager_on_graph`` break closure.

    ``site`` is keyed three ways (fact 22): ``op:sglang::<name>`` for the
    custom-op wrappers, ``py:<module>:<qualname>`` for plain functions and
    ``model:<submodule path>:<method>`` for bound methods. ``bridge_output``
    has ``kind="none"`` at the in-place sites whose real bridge is a mutated
    argument tensor.
    """

    site: str
    args: tuple[ArgSpec, ...]
    bridge_output: ArgSpec
    kwargs: dict[str, ArgSpec] = {}
    used_capture_stub: bool = False


class ShapeKeyRecord(msgspec.Struct, frozen=True, kw_only=True):
    """Serializable twin of the runner ``ShapeKey`` dataclass."""

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
    dsa_variant: Optional[str] = None

    @classmethod
    def from_shape_key(cls, shape_key: Any) -> ShapeKeyRecord:
        return cls(
            size=int(shape_key.size),
            stream_idx=shape_key.stream_idx,
            variant_label=shape_key.variant_label,
            dsa_variant=shape_key.dsa_variant,
        )

    def to_shape_key(self) -> ShapeKey:
        from sglang.srt.model_executor.runner.shape_key import ShapeKey

        return ShapeKey(
            size=self.size,
            stream_idx=self.stream_idx,
            variant_label=self.variant_label,
            dsa_variant=self.dsa_variant,
        )

    def label(self) -> str:
        """Stable text key for verdict tables (``RankManifest.verdicts``)."""
        parts = [f"size={self.size}"]
        if self.stream_idx is not None:
            parts.append(f"stream={self.stream_idx}")
        if self.variant_label is not None:
            parts.append(f"variant={self.variant_label}")
        if self.dsa_variant is not None:
            parts.append(f"dsa={self.dsa_variant}")
        return ";".join(parts)


class ShapeArtifact(msgspec.Struct, frozen=True, kw_only=True):
    """Everything a backend needs to rebuild one shape without a forward.

    ``graphs`` holds one entry for the Full backend and one per segment for
    the Breakable backend; ``kernels`` is the identity table ``KernelNode.
    identity`` indexes into; ``capture_inputs`` describes the DP-padding
    tensors a Breakable capture retained.
    """

    shape_key: ShapeKeyRecord
    backend: str
    graphs: tuple[SerializedGraph, ...]
    output: OutputSchema
    kernels: tuple[KernelIdentity, ...] = ()
    breaks: tuple[BreakSiteRecord, ...] = ()
    capture_inputs: tuple[OutputSchema, ...] = ()


class CommStateBlob(msgspec.Struct, frozen=True, kw_only=True):
    """Communicator rows a runner's graphs reference (design section 6.10).

    ``rows`` are ``(absolute row index, input RegionRef, nbytes)``. Row
    contents are never saved: they are process-local peer addresses and are
    re-exchanged at load. ``max_row`` lets the loader pre-advance the
    communicator's row counter so recaptured shapes allocate above the loaded
    rows (fact 17).
    """

    impl: str
    group: str
    rows: tuple[tuple[int, RegionRef, int], ...] = ()
    max_row: int = -1


class RunnerBundle(msgspec.Struct, frozen=True, kw_only=True):
    """All shapes of one runner (``decode``, ``prefill``, ...) on one rank."""

    runner: str
    backend: str
    shapes: tuple[ShapeArtifact, ...]
    comm: tuple[CommStateBlob, ...] = ()


class RankManifest(msgspec.Struct, frozen=True, kw_only=True):
    """Per-rank index written beside the runner bundles (``rank.json``)."""

    fingerprint: GraphArtifactFingerprint
    regions: tuple[RegionSpec, ...]
    placement: str
    format_version: int = FORMAT_VERSION
    arena_bases: dict[str, int] = {}
    runners: tuple[str, ...] = ()
    images: tuple[KernelImage, ...] = ()
    verdicts: dict[str, str] = {}


class ArtifactManifest(msgspec.Struct, frozen=True, kw_only=True):
    """Top-level ``manifest.json``; written last, by rank 0, after a barrier."""

    fingerprint_digest: str
    world_size: int
    ranks: tuple[int, ...]
    placement: str
    format_version: int = FORMAT_VERSION
    sglang_version: str = ""


def unsupported_shape_artifact(
    shape_key: Any, backend: str, reason: str
) -> ShapeArtifact:
    """The artifact a backend that cannot serialize returns from
    ``export_shape``: one empty graph carrying a ``needs_recapture`` verdict."""
    return ShapeArtifact(
        shape_key=ShapeKeyRecord.from_shape_key(shape_key),
        backend=backend,
        graphs=(
            SerializedGraph(
                nodes=(),
                edges=(),
                param_bytes=b"",
                slots=(),
                signature="",
                verdict=GraphVerdict.NEEDS_RECAPTURE,
                verdict_reason=reason,
            ),
        ),
        output=OutputSchema(kind="none"),
    )


def shape_artifact_verdict(artifact: ShapeArtifact) -> GraphVerdict:
    """A shape is serializable only if every one of its graphs is."""
    for graph in artifact.graphs:
        if graph.verdict is not GraphVerdict.SERIALIZABLE:
            return GraphVerdict.NEEDS_RECAPTURE
    if not artifact.graphs:
        return GraphVerdict.NEEDS_RECAPTURE
    return GraphVerdict.SERIALIZABLE
