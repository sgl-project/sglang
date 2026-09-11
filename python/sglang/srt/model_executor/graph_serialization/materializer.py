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
"""The per-runner seam that decides capture versus import per shape.

Design section 6.1, decision 5: runners keep their capture skeleton and the
single place that used to call ``backend.capture_one`` calls a
:class:`GraphMaterializer` collaborator instead. Decision 4: every decision
is per graph, fail closed, and agreed across ranks before any import.

* :class:`CaptureOnlyMaterializer` is what ``--cuda-graph-cache-mode off``
  (the default) resolves to. Its ``materialize`` is exactly one
  ``backend.capture_one`` call with the arguments the runner passed, so the
  default server path is unchanged.
* :class:`ArtifactGraphMaterializer` carries the save and load paths of
  design sections 6.8 and 12. In this draft the per-shape verdict, the
  cross-rank agreement in ``plan`` (the runners call it before their shape
  loop; nothing loads without it), the capture-then-export and
  import-else-capture branches are implemented; the save-side self-check,
  the bundle write and the communicator restore in ``finish`` raise
  ``NotImplementedError``.

The runners receive their :class:`GraphSerializationPlan` from
``cuda_graph_setup`` and hand it to :func:`resolve_materializer`; that plan
is the single source of truth and nothing here re-reads the config.

The three records (:class:`ShapePlan`, :class:`GraphSaveContext`,
:class:`GraphLoadContext`) hold live objects and are never serialized; they
are ``msgspec.Struct`` for the frozen, keyword-only constructor only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.model_executor.graph_serialization.format import (
    GraphVerdict,
    ShapeArtifact,
    ShapeKeyRecord,
    shape_artifact_verdict,
)
from sglang.srt.model_executor.graph_serialization.plan import GraphSerializationPlan
from sglang.srt.model_executor.graph_serialization.store import GraphArtifactStore

if TYPE_CHECKING:
    from sglang.srt.model_executor.graph_serialization.safety import CoverageReport

logger = logging.getLogger(__name__)

# Per-shape decisions exchanged between ranks by ``plan()``.
VERDICT_LOAD = "load"
VERDICT_CAPTURE = "capture"
_KNOWN_VERDICTS = frozenset((VERDICT_LOAD, VERDICT_CAPTURE))


class GraphImportError(RuntimeError):
    """``import_shape`` could not install a replayable artifact.

    Backends raise it all-or-nothing (design section 6.7): after it, the
    backend's tables for that shape are exactly as before the attempt, so the
    materializer may fall through to a normal capture.
    """


class ShapePlan(msgspec.Struct, frozen=True, kw_only=True):
    """What a runner's ``prepare_one_shape`` produced for one shape.

    ``shape_key`` is the runner ``ShapeKey`` dataclass and is never encoded
    (its serializable twin is ``format.ShapeKeyRecord``). ``event_roles`` maps
    an event role such as ``metadata_prep_done`` to the live
    ``torch.cuda.Event`` the graph records (facts 15 and 18).
    """

    shape_key: Any
    forward_fn: Any
    capture_inputs: Any = None
    post_warmup_hook: Any = None
    event_roles: dict = {}


class GraphSaveContext(msgspec.Struct, frozen=True, kw_only=True):
    """Live collaborators ``backend.export_shape`` needs (design section 6.7):
    the codec, the region registry, the kernel resolver and the safety
    policy. ``event_roles`` maps event handles to roles for the encoder."""

    codec: Any
    registry: Any
    resolver: Any
    policy: Any
    event_roles: dict = {}


class GraphLoadContext(msgspec.Struct, frozen=True, kw_only=True):
    """Live collaborators ``backend.import_shape`` needs (design section 6.7):
    the codec, the relocation map, the kernel resolver, the event table, the
    model (break-site module lookups), the driver context and the optional
    dedup registry."""

    codec: Any
    reloc: Any
    resolver: Any
    events: Any
    model: Any
    device_ctx: int = 0
    dedup: Any = None


def _label(shape_key: Any) -> str:
    """Stable text key of a shape (``RankManifest.verdicts`` uses the same)."""
    return ShapeKeyRecord.from_shape_key(shape_key).label()


def _world_rank() -> int:
    """Artifacts are strictly per world rank (design section 9.4)."""
    try:
        import torch.distributed as dist
    except ImportError:  # pragma: no cover - torch is always present in the runtime
        return 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class GraphMaterializer(ABC):
    """Per-runner collaborator deciding capture versus import per shape.

    Composition, not inheritance: the runner owns one materializer next to
    its backend and calls ``session`` around the shape loop, ``plan`` before
    it, ``materialize`` per shape and ``finish`` after it (design section
    6.8).
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @property
    def backend(self) -> Any:
        return self._backend

    @contextmanager
    def session(self, stream: Any) -> Iterator[None]:
        """Wraps the runner's outer capture loop; the default is the backend's
        own ``capture_session``. A placement policy that pins regions adds its
        pool scope here in a full build."""
        with self._backend.capture_session(stream):
            yield

    def plan(self, runner: Any) -> None:
        """Decide before the shape loop. The default has nothing to decide:
        every shape captures, so no verdicts and no cross-rank exchange."""
        return None

    @abstractmethod
    def materialize(self, plan: ShapePlan) -> None:
        """Make ``plan.shape_key`` replayable, by capture or by import."""

    def finish(self) -> Optional[CoverageReport]:
        """After both phases' shape loops. The default has nothing to report."""
        return None


class CaptureOnlyMaterializer(GraphMaterializer):
    """``--cuda-graph-cache-mode off``: materialize is ``capture_one`` and
    nothing else, so the default server path is byte-for-byte unchanged."""

    def materialize(self, plan: ShapePlan) -> None:
        self._backend.capture_one(
            plan.shape_key,
            plan.forward_fn,
            capture_inputs=plan.capture_inputs,
            post_warmup_hook=plan.post_warmup_hook,
        )


class ArtifactGraphMaterializer(GraphMaterializer):
    """Save and load through a :class:`GraphArtifactStore` (design sections
    6.8 and 12).

    Verdicts: a shape is ``"load"`` only when the plan loads, the store holds
    this rank's bundle for this runner, the bundle has an artifact with the
    same shape label and every graph of that artifact is ``serializable``;
    anything else is ``"capture"`` (fail closed, design section 6.1 decision
    4). ``plan()`` computes the vector over ``runner.planned_shape_keys()``,
    all-gathers it over the CPU group and keeps the per-shape agreement, so
    every rank loads or recaptures the same shapes; until it ran, every
    shape captures. Fingerprint equality,
    region binding and kernel resolvability (safety ladder steps 2 and 3 of
    section 6.11) join the verdict in a full build; ``compute_fingerprint``
    is a stub in this draft.

    ``session`` is inherited: the relocate placement (the default) needs no
    pool scope, and the fixed-VA arena scope is not in this draft.
    """

    def __init__(
        self,
        backend: Any,
        *,
        plan: GraphSerializationPlan,
        store: GraphArtifactStore,
        runner_name: str,
        save_ctx: Optional[GraphSaveContext] = None,
        load_ctx: Optional[GraphLoadContext] = None,
        cpu_group: Any = None,
    ) -> None:
        super().__init__(backend)
        self._plan = plan
        self._store = store
        self._runner_name = runner_name
        self._save_ctx = save_ctx
        self._load_ctx = load_ctx
        self._cpu_group = cpu_group
        # label -> artifact read from the store for this rank and runner;
        # None until first needed.
        self._store_artifacts: Optional[dict[str, ShapeArtifact]] = None
        # label -> agreed verdict; None until plan() ran with a shape list.
        self._verdicts: Optional[dict[str, str]] = None
        self._exports: dict[str, ShapeArtifact] = {}
        self._imported: dict[str, ShapeArtifact] = {}

    # -- read-only views ----------------------------------------------------

    @property
    def plan_settings(self) -> GraphSerializationPlan:
        return self._plan

    @property
    def store(self) -> GraphArtifactStore:
        return self._store

    @property
    def runner_name(self) -> str:
        return self._runner_name

    @property
    def exports(self) -> MappingProxyType[str, ShapeArtifact]:
        """Artifacts exported after capture in this process, by shape label."""
        return MappingProxyType(self._exports)

    @property
    def imported(self) -> MappingProxyType[str, ShapeArtifact]:
        """Artifacts installed through ``import_shape``, by shape label."""
        return MappingProxyType(self._imported)

    # -- verdicts -----------------------------------------------------------

    def _read_store_artifacts(self) -> dict[str, ShapeArtifact]:
        if not self._plan.loads:
            return {}
        rank = _world_rank()
        if not self._store.has_rank(rank):
            logger.info(
                "no CUDA graph artifact for rank %d under %s; every %s shape captures",
                rank,
                self._store.artifact_dir,
                self._runner_name,
            )
            return {}
        try:
            _manifest, bundles = self._store.read_rank(rank)
        except (OSError, ValueError, msgspec.MsgspecError) as exc:
            if self._plan.strict:
                raise
            logger.warning(
                "CUDA graph artifact for rank %d under %s is unreadable; every "
                "%s shape captures: %s",
                rank,
                self._store.artifact_dir,
                self._runner_name,
                exc,
            )
            return {}
        bundle = bundles.get(self._runner_name)
        if bundle is None:
            logger.info(
                "CUDA graph artifact for rank %d has no %r bundle; every shape "
                "captures",
                rank,
                self._runner_name,
            )
            return {}
        return {artifact.shape_key.label(): artifact for artifact in bundle.shapes}

    def _artifacts(self) -> dict[str, ShapeArtifact]:
        if self._store_artifacts is None:
            self._store_artifacts = self._read_store_artifacts()
        return self._store_artifacts

    def _local_verdict(self, label: str) -> str:
        if not self._plan.loads:
            return VERDICT_CAPTURE
        artifact = self._artifacts().get(label)
        if artifact is None:
            return VERDICT_CAPTURE
        if shape_artifact_verdict(artifact) is not GraphVerdict.SERIALIZABLE:
            return VERDICT_CAPTURE
        return VERDICT_LOAD

    def verdict_for(self, shape_key: Any) -> str:
        """``"load"`` or ``"capture"`` for one shape.

        Only a verdict agreed across ranks in ``plan()`` can load. Before
        ``plan()`` ran, or for a shape it never saw, the answer is
        ``"capture"``: an import nobody agreed to is the per-rank fail-open
        design section 6.1 decision 4 forbids.
        """
        if self._verdicts is None:
            return VERDICT_CAPTURE
        return self._verdicts.get(_label(shape_key), VERDICT_CAPTURE)

    def _gather_verdicts(self, local: list[str]) -> list[list[str]]:
        """Every rank's local vector, this rank's included. Single-process
        (no CPU group, or ``torch.distributed`` not initialized) is the one
        rank's own vector."""
        if self._cpu_group is None:
            return [list(local)]
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - torch is always present
            return [list(local)]
        if not (dist.is_available() and dist.is_initialized()):
            return [list(local)]
        gathered: list[Any] = [None] * dist.get_world_size(group=self._cpu_group)
        dist.all_gather_object(gathered, list(local), group=self._cpu_group)
        return [list(vector) for vector in gathered]

    def plan(self, runner: Any) -> None:
        """Compute this rank's verdict vector over ``runner.planned_shape_keys()``
        and agree it across the CPU group (design section 6.1 decision 4,
        section 12 load step 3). The runners call it once before their shape
        loop; until it ran, ``verdict_for`` is ``"capture"`` for every shape.

        A runner without ``planned_shape_keys`` cannot have its shapes agreed,
        so every shape captures (fail closed) and a warning says so.
        Communicator counter pre-advance (section 6.10) belongs here too and
        is not in this draft.
        """
        planned = getattr(runner, "planned_shape_keys", None)
        if not callable(planned):
            logger.warning(
                "%s: %s cannot enumerate its shapes before capture, so no "
                "verdict can be agreed across ranks and every shape captures",
                self._runner_name,
                type(runner).__name__,
            )
            self._verdicts = {}
            return None
        shape_keys = list(planned())
        local = [self._local_verdict(_label(key)) for key in shape_keys]
        agreed = agree_verdicts(self._gather_verdicts(local))
        self._verdicts = {_label(key): v for key, v in zip(shape_keys, agreed)}
        demoted = sum(1 for a, b in zip(local, agreed) if a != b)
        if demoted:
            logger.info(
                "%s: %d shape(s) this rank could load capture because another "
                "rank cannot",
                self._runner_name,
                demoted,
            )
        return None

    # -- per shape ----------------------------------------------------------

    def materialize(self, plan: ShapePlan) -> None:
        label = _label(plan.shape_key)
        if self.verdict_for(plan.shape_key) == VERDICT_LOAD:
            artifact = self._artifacts()[label]
            try:
                if self._load_ctx is None:
                    raise NotImplementedError(
                        "ArtifactGraphMaterializer.materialize: importing "
                        "without a GraphLoadContext (codec, relocation map, "
                        "resolver, events) is not implemented in this draft; see "
                        "DESIGN_cuda_graph_serialization.md section 6.8"
                    )
                self._backend.import_shape(plan.shape_key, artifact, self._load_ctx)
            except GraphImportError as exc:
                if self._plan.strict:
                    raise
                logger.warning(
                    "%s shape %s: import failed, capturing instead: %s",
                    self._runner_name,
                    label,
                    exc,
                )
            else:
                self._imported[label] = artifact
                return

        if self._plan.saves and self._save_ctx is None:
            raise NotImplementedError(
                "ArtifactGraphMaterializer.materialize: exporting without a "
                "GraphSaveContext (codec, registry, resolver, policy) is not "
                "implemented in this draft; see "
                "DESIGN_cuda_graph_serialization.md section 6.8"
            )
        self._backend.capture_one(
            plan.shape_key,
            plan.forward_fn,
            capture_inputs=plan.capture_inputs,
            post_warmup_hook=plan.post_warmup_hook,
        )
        if self._plan.saves:
            self._exports[label] = self._backend.export_shape(
                plan.shape_key, self._save_ctx
            )

    # -- after the shape loop -----------------------------------------------

    def finish(self) -> Optional[CoverageReport]:
        """Save: self-check every export and write the rank bundle (design
        section 6.8, section 12 save step 5). Load: restore communicator rows
        and windows collectively (section 6.10, fact 17), then the optional
        shadow verify and smoke replay. Neither is in this draft."""
        if self._exports:
            raise NotImplementedError(
                "ArtifactGraphMaterializer.finish: the save-side self-check and "
                "the atomic rank bundle write are not implemented in this draft; "
                "see DESIGN_cuda_graph_serialization.md section 6.8 and section 12 "
                "(save step 5)"
            )
        if self._imported:
            raise NotImplementedError(
                "ArtifactGraphMaterializer.finish: the collective communicator "
                "row and window restore after the shape loop is not implemented "
                "in this draft; see DESIGN_cuda_graph_serialization.md section 6.10"
            )
        return None


def agree_verdicts(vectors: Sequence[Sequence[str]]) -> list[str]:
    """Per-shape agreement across ranks: ``"load"`` only when every rank says
    ``"load"``, else ``"capture"`` (design section 6.1 decision 4).

    Every vector must have the same length (the ranks iterate the same
    planned shapes in the same order); a mismatch is a ``ValueError`` rather
    than a silent truncation. Unknown verdict tokens are rejected too.
    """
    if not vectors:
        return []
    width = len(vectors[0])
    for index, vector in enumerate(vectors):
        if len(vector) != width:
            raise ValueError(
                f"verdict vector from rank {index} has {len(vector)} entries; "
                f"rank 0 has {width}"
            )
        unknown = sorted(set(vector) - _KNOWN_VERDICTS)
        if unknown:
            raise ValueError(
                f"verdict vector from rank {index} has unknown verdict(s) {unknown}"
            )
    return [
        (
            VERDICT_LOAD
            if all(vector[i] == VERDICT_LOAD for vector in vectors)
            else VERDICT_CAPTURE
        )
        for i in range(width)
    ]


def keeps_raw_graphs(plan: Optional[GraphSerializationPlan]) -> bool:
    """Whether a runner's captures must keep their ``CUgraph``.

    ``FullCudaGraphBackend`` then constructs
    ``torch.cuda.CUDAGraph(keep_graph=True)`` and instantiates explicitly so
    ``export_shape`` can encode ``raw_cuda_graph()``: design section 6.7,
    "when serialization is on". The runners pass this to
    ``resolve_decode_backend`` / ``resolve_prefill_backend``. ``None`` (a
    runner built outside the lifecycle) and a disabled plan keep nothing, so
    the default path builds ``CUDAGraph()`` exactly as before.
    """
    return plan is not None and plan.enabled


_RUNNER_BUNDLE_NAMES = {
    "DecodeCudaGraphRunner": "decode",
    "PrefillCudaGraphRunner": "prefill",
}


def _runner_bundle_name(runner: Any) -> str:
    """``RunnerBundle.runner`` key for a runner: ``decode`` / ``prefill`` for
    the two v1 runners (target-verify and draft phases are v2, design section
    9.3); the class name otherwise, so an unknown runner still gets its own
    bundle file."""
    name = type(runner).__name__
    return _RUNNER_BUNDLE_NAMES.get(name, name)


def resolve_materializer(
    runner: Any, plan: Optional[GraphSerializationPlan]
) -> GraphMaterializer:
    """The materializer a runner installs next to its backend.

    ``plan`` is the one ``plan_graph_serialization`` resolved in
    ``cuda_graph_setup.capture_cuda_graphs`` and threaded through the runner
    constructor. It is the single source of truth: this function never
    re-reads the config, so the device and draft-worker gates the component
    applied (design sections 9.3 and 13) hold here too. ``None`` (a runner
    built outside that lifecycle: speculative, platform and out-of-tree
    runners) and a disabled plan give :class:`CaptureOnlyMaterializer`; an
    enabled plan gives an :class:`ArtifactGraphMaterializer` over
    ``plan.cache_dir``. In this draft the store digest is the ``"pending"``
    placeholder and no save or load context exists, so save and load fail
    loudly at the first shape.
    """
    if plan is None or not plan.enabled:
        return CaptureOnlyMaterializer(runner.backend)
    logger.warning(
        "--cuda-graph-cache-mode %s: CUDA graph save/load are interfaces only in "
        "this build; the first shape fails with NotImplementedError",
        plan.mode.value,
    )
    tp_group = getattr(getattr(runner, "model_runner", None), "tp_group", None)
    return ArtifactGraphMaterializer(
        runner.backend,
        plan=plan,
        store=GraphArtifactStore(plan.cache_dir, digest="pending"),
        runner_name=_runner_bundle_name(runner),
        cpu_group=getattr(tp_group, "cpu_group", None),
    )
