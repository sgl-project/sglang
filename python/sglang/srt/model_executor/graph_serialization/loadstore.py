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
"""Save captured CUDA graphs to files and rebuild them (design sections 6.3,
6.6, 6.8, 11, 12).

What is saved
    :class:`Graph` is one ``CUgraph`` with every pointer word replaced by a
    :class:`RegionRef` (``memory.py``) and every kernel by a
    :class:`KernelIdentity` (``kernels.py``). :class:`ShapeArtifact` is one
    captured shape of one runner; :class:`Artifact` is everything one rank
    writes, including the communicator checkpoints (``comm.py``) and a flat
    compatibility fingerprint.

Graph <-> artifact
    :func:`save_graph` walks a live ``CUgraph`` (fact 2: pointers hide inside
    parameter structs and are found by scanning 8-byte words against the live
    regions). :func:`load_graph` rebuilds it node by node with re-resolved
    kernels, rebased pointer words and re-applied launch attributes (facts 4,
    19), binds the runner's external event by role (fact 18) and instantiates.

Files
    ``<cache_dir>/<fingerprint digest>/rank_<r>.msgpack`` plus
    ``images/<sha256>.img`` for kernel containers. Artifacts are strictly per
    rank (design section 9.4).

The seam
    :class:`GraphCache` is the one object a runner talks to: ``materialize``
    loads a shape when every rank can, else captures (and exports when saving);
    ``finish`` writes the rank's artifact or restores the communicators. With
    ``--cuda-graph-cache-mode off`` it is exactly ``backend.capture_one``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.model_executor.graph_serialization.comm import CommCheckpoint, CommState
from sglang.srt.model_executor.graph_serialization.kernels import (
    KernelIdentity,
    KernelResolver,
)
from sglang.srt.model_executor.graph_serialization.memory import (
    MemoryMap,
    Region,
    RegionRef,
    Relocation,
)
from sglang.srt.runtime_context import get_exec

FORMAT_VERSION = 1
MODES = ("off", "save", "load", "auto")

# ---- what is saved ---------------------------------------------------------------


class Node(msgspec.Struct, frozen=True, kw_only=True):
    """One CUDA graph node of any kind."""

    kind: str  # kernel | memcpy | memset | event
    # kernel params verbatim with host garbage zeroed (fact 14), or the
    # memcpy / memset struct
    params: bytes = b""
    kernel: int = -1  # index into Graph.kernels
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    smem: int = 0
    # required on rebuild, e.g. cluster dimensions (fact 19)
    launch_attrs: tuple[tuple[int, bytes], ...] = ()
    # event nodes bind by role at load, never by handle (fact 18)
    event_role: str = ""


class Graph(msgspec.Struct, frozen=True, kw_only=True):
    """One ``CUgraph``, pointer-free."""

    nodes: tuple[Node, ...]
    edges: tuple[tuple[int, int], ...]
    slots: tuple[tuple[int, int, RegionRef], ...]  # (node, byte offset, region ref)
    kernels: tuple[KernelIdentity, ...]
    recapture_reason: str = ""  # non-empty: do not load this graph, capture its shape


class TensorRef(msgspec.Struct, frozen=True, kw_only=True):
    """One tensor of the replay output, addressed by path (``"next_token_logits"``,
    ``"pp_proxy.residual"``); rebuilt as a view with explicit shape, stride and dtype.
    """

    path: str
    ref: RegionRef
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


class ShapeArtifact(msgspec.Struct, frozen=True, kw_only=True):
    """One captured shape of one runner."""

    shape_key: dict  # ShapeKey fields
    backend: str  # graph backend name: full | breakable | tc_piecewise | npu | xpu
    graphs: tuple[Graph, ...]  # Full: one; Breakable: one per segment
    outputs: tuple[TensorRef, ...] = ()


class Artifact(msgspec.Struct, frozen=True, kw_only=True):
    """Everything one rank writes."""

    fingerprint: dict[str, Any]
    regions: tuple[Region, ...]
    shapes: dict[str, tuple[ShapeArtifact, ...]]  # runner name -> shapes
    comm: tuple[CommCheckpoint, ...] = ()
    format_version: int = FORMAT_VERSION


def unsupported_shape(shape_key: Any, backend: str, reason: str) -> ShapeArtifact:
    """What a backend that cannot serialize returns from ``export_shape``."""
    return ShapeArtifact(
        shape_key=dataclasses.asdict(shape_key),
        backend=backend,
        graphs=(
            Graph(nodes=(), edges=(), slots=(), kernels=(), recapture_reason=reason),
        ),
    )


# ---- graph <-> artifact ----------------------------------------------------------


def save_graph(
    raw_graph: int,
    *,
    memory: MemoryMap,
    kernels: KernelResolver,
    events: Mapping[str, Any],
) -> Graph:
    """Encode a live ``CUgraph``.

    Nodes and edges via ``cuGraphGetNodes`` / ``cuGraphGetEdges``; kernel
    parameters via ``cuFuncGetParamInfo`` or the ``extra`` blob (fact 10);
    launch attributes via ``cuGraphKernelNodeGetAttribute``; every
    8-byte-aligned word that ``memory.classify`` places becomes a slot, a live
    device pointer in no region sets ``recapture_reason``, dead host words are
    zeroed (fact 14). An NCCL kernel (``comm.is_nccl_kernel``), an RNG kernel
    (fact 16) or an unsupported node type also sets ``recapture_reason``.
    """
    raise NotImplementedError(
        "save_graph is not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 6.6"
    )


def load_graph(
    graph: Graph,
    *,
    reloc: Relocation,
    kernels: KernelResolver,
    events: Mapping[str, Any],
) -> LoadedGraph:
    """Rebuild a ``CUgraph`` from :class:`Graph` and instantiate it.

    ``cuGraphCreate``; per node ``cuGraphAddKernelNode`` with the re-resolved
    function and the parameter bytes patched at every slot by
    ``reloc.rebase``; every recorded launch attribute re-applied (fact 19);
    memcpy / memset with the device context; event nodes bound through
    ``events[role]`` at graph level (fact 18); ``cuGraphAddDependencies``;
    kernel names checked against the artifact (fact 8); ``cuGraphInstantiate``.
    """
    raise NotImplementedError(
        "load_graph is not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 6.6"
    )


class LoadedGraph:
    """A rebuilt graph that duck-types ``torch.cuda.CUDAGraph`` for the backend
    tables (``replay`` / ``raw_cuda_graph`` / ``reset``)."""

    def __init__(self, raw_graph: int, exec_handle: int) -> None:
        self.raw_graph = raw_graph
        self.exec_handle = exec_handle

    def replay(self) -> None:
        raise NotImplementedError(
            "LoadedGraph.replay (cuGraphLaunch on the current stream) is not "
            "implemented in this draft; see DESIGN_cuda_graph_serialization.md "
            "section 6.6"
        )

    def raw_cuda_graph(self) -> int:
        return self.raw_graph

    def reset(self) -> None:
        raise NotImplementedError(
            "LoadedGraph.reset is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 6.6"
        )


# ---- files -----------------------------------------------------------------------


def fingerprint() -> dict[str, Any]:
    """Flat compatibility leaves (design section 11): torch / CUDA / driver /
    kernel-library versions, device name and compute capability, model config
    digest, every parallel size and rank, the resolved post-capture graph
    config and per-runner geometry, backend selections, weight-cache mode and
    config, weight layout digest, autotune cache digest, placement. Never the
    device UUID, so artifacts move between identical GPUs."""
    raise NotImplementedError(
        "fingerprint is not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 11"
    )


def fingerprint_digest(fp: Mapping[str, Any]) -> str:
    """Names the artifact directory."""
    return hashlib.sha256(msgspec.json.encode(fp, order="deterministic")).hexdigest()


def fingerprint_diff(saved: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    """Keys that differ, so a rejected artifact is explained."""
    return sorted(k for k in set(saved) | set(live) if saved.get(k) != live.get(k))


class ArtifactStore:
    """``<root>/<digest>/rank_<r>.msgpack`` and ``images/<sha256>.img``."""

    def __init__(self, root: str, digest: str) -> None:
        self.dir = Path(root) / digest

    def rank_path(self, rank: int) -> Path:
        return self.dir / f"rank_{rank}.msgpack"

    def has(self, rank: int) -> bool:
        return self.rank_path(rank).is_file()

    def save(self, rank: int, artifact: Artifact) -> Path:
        """Atomic: write to a temporary file, then ``os.replace``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.rank_path(rank)
        tmp = path.with_suffix(f".tmp-{os.getpid()}")
        tmp.write_bytes(msgspec.msgpack.encode(artifact))
        os.replace(tmp, path)
        return path

    def load(self, rank: int) -> Artifact:
        artifact = msgspec.msgpack.decode(
            self.rank_path(rank).read_bytes(), type=Artifact
        )
        if artifact.format_version != FORMAT_VERSION:
            raise ValueError(
                f"artifact format {artifact.format_version}, this build reads {FORMAT_VERSION}"
            )
        return artifact

    def put_image(self, sha256: str, data: bytes) -> Path:
        """Kernel container bytes, content-addressed and shared by all ranks."""
        raise NotImplementedError(
            "ArtifactStore.put_image is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 11"
        )


# ---- the seam --------------------------------------------------------------------


class GraphImportError(RuntimeError):
    """``backend.import_shape`` could not install an artifact; no partial state."""


def graph_cache_mode(model_runner: Any) -> str:
    """The resolved ``--cuda-graph-cache-mode`` for this runner: ``off`` for a
    draft worker (spec runners are capture-only in v1, design section 9.3) and
    for non-CUDA devices. Reads the published bag, never ``ServerArgs``."""
    if model_runner.is_draft_worker or model_runner.device != "cuda":
        return "off"
    return get_exec().graph.cuda_graph_cache_mode


def graph_cache_dir() -> str:
    return get_exec().graph.cuda_graph_cache_dir or os.path.join(
        os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), "cuda_graphs"
    )


class GraphCache:
    """One per runner: the capture-or-load decision for every shape.

    ``mode == "off"`` (the default) makes :meth:`materialize` exactly
    ``backend.capture_one`` and :meth:`finish` a no-op, so the server path is
    unchanged. The backend hooks ``export_shape(shape_key, cache)`` and
    ``import_shape(shape_key, artifact, cache)`` read ``memory``, ``kernels``,
    ``reloc`` and ``events`` from here.
    """

    def __init__(
        self,
        backend: Any,
        *,
        mode: str = "off",
        cache_dir: Optional[str] = None,
        runner_name: str = "",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown cuda graph cache mode {mode!r}")
        self.backend = backend
        self.mode = mode
        self.cache_dir = cache_dir
        self.runner_name = runner_name
        # Live collaborators, built on first use when the cache is enabled.
        self.memory: Optional[MemoryMap] = None
        self.kernels: Optional[KernelResolver] = None
        self.comm: list[CommState] = []
        self.reloc: Optional[Relocation] = None
        self.events: dict[str, Any] = {}
        self.store: Optional[ArtifactStore] = None
        self.exports: dict[Any, ShapeArtifact] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def saves(self) -> bool:
        return self.mode in ("save", "auto")

    def materialize(
        self,
        shape_key: Any,
        forward_fn: Callable[[], Any],
        *,
        capture_inputs: Any = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
        events: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Load ``shape_key`` if every rank of the group has a loadable artifact
        for it (one small collective per shape, design section 6.1 decision 4),
        else capture; when saving, keep ``backend.export_shape`` for
        :meth:`finish`. ``events`` names the runner's external events by role
        and is merged into :attr:`events`, which ``export_shape`` /
        ``import_shape`` read.
        """
        if not self.enabled:
            self.backend.capture_one(
                shape_key,
                forward_fn,
                capture_inputs=capture_inputs,
                post_warmup_hook=post_warmup_hook,
            )
            return
        if events:
            self.events.update(events)
        raise NotImplementedError(
            "GraphCache.materialize: load / capture+export is not implemented in "
            "this draft; see DESIGN_cuda_graph_serialization.md section 6.8"
        )

    def finish(self) -> None:
        """After the shape loop. Save: ``fingerprint()``, ``memory.regions()``,
        ``comm`` exports and the collected shapes become this rank's
        :class:`Artifact`. Load: ``comm`` restore (collective), one smoke replay.
        """
        if not self.enabled:
            return
        raise NotImplementedError(
            "GraphCache.finish is not implemented in this draft; see "
            "DESIGN_cuda_graph_serialization.md section 12"
        )


def finish_graph_caches(*runners: Any) -> None:
    """``GraphCache.finish`` for every runner that has one; called once by
    ``cuda_graph_setup`` after both captures (design section 12, step 5)."""
    for runner in runners:
        cache = getattr(runner, "graph_cache", None)
        if cache is not None:
            cache.finish()
