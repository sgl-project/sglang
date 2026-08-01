# SPDX-License-Identifier: Apache-2.0
"""Compiled, validated form of a DAG topology.

``ExecutionPlan`` turns a ``DagSpec`` into the lookup tables the runtime needs
(adjacency, in-degree, topological order, stage ownership) and rejects
topologies that cannot execute.  Everything downstream -- the orchestrator's
request scheduler, the worker's capability bits, the pipeline's stage filter
-- reads the plan and never the raw spec, so a malformed topology fails at
startup rather than mid-request.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.dag.predicate import (
    PredicateError,
    compile_predicate,
    evaluate_predicate,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.spec import (
    DagSpec,
    JoinPolicy,
    PoolSpec,
    RoleSpec,
    RouteSpec,
    StageSpec,
)

logger = logging.getLogger(__name__)

# Modules each classic role needs, used by from_classic_roles().
_CLASSIC_ENCODER_MODULES = [
    "text_encoder",
    "tokenizer",
    "image_encoder",
    "image_processor",
    "processor",
    "connectors",
    "vae",
    "audio_vae",
    "video_vae",
]
_CLASSIC_DENOISER_MODULES = ["transformer", "scheduler", "vae", "audio_vae"]
_CLASSIC_DECODER_MODULES = ["vae", "audio_vae", "video_vae", "vocoder"]

# Stages whose presence means the node must build a per-request scheduler
# clone before compute.  Matched case-insensitively as a substring of the
# registered stage name, which is a class name for most stages but an explicit
# snake_case string for the ones registered through ``add_stage_factory``.
_SCHEDULER_STAGE_MARKERS = ("denoising", "refinement", "refiner")


class PlanValidationError(ValueError):
    """Raised when a DAG topology cannot be compiled into a runnable plan."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            "Invalid DAG topology:\n" + "\n".join(f"  - {e}" for e in errors)
        )


@dataclass
class CompiledEdge:
    """A DAG edge with its predicate pre-parsed."""

    src: str
    dst: str
    when: str | None = None
    fields: list[str] | None = None
    _ast: ast.Expression | None = None

    @property
    def edge_id(self) -> str:
        return f"{self.src}->{self.dst}"

    @property
    def is_conditional(self) -> bool:
        return self.when is not None

    def is_live(self, context: dict[str, Any]) -> bool:
        """Whether this edge is taken for a request with the given metadata."""
        if self._ast is None:
            return True
        return evaluate_predicate(self._ast, context)

    def filter_manifest(self, manifest: dict) -> dict:
        """Restrict a transfer manifest to the fields this edge carries.

        The manifest is keyed by Req field name, so an edge-specific view is a
        dict filter; the underlying staged bytes are shared across all
        outgoing edges of a fan-out.
        """
        if self.fields is None:
            return manifest
        allowed = set(self.fields)
        return {k: v for k, v in manifest.items() if k in allowed}

    def filter_scalars(self, scalar_fields: dict) -> dict:
        """Restrict scalar fields to those this edge carries.

        Control keys (``_``-prefixed) always travel: they carry trace context
        and slot bookkeeping rather than model state.
        """
        if self.fields is None:
            return scalar_fields
        allowed = set(self.fields)
        return {
            k: v for k, v in scalar_fields.items() if k in allowed or k.startswith("_")
        }


@dataclass
class CompiledNode:
    """A DAG node together with its resolved pool and graph position."""

    name: str
    stages: list[str]
    modules: list[str] | None
    join: JoinPolicy
    terminal: bool
    emit: list[str] | None
    pool: PoolSpec
    scheduler_override: bool | None = None
    in_edges: list[CompiledEdge] = field(default_factory=list)
    out_edges: list[CompiledEdge] = field(default_factory=list)

    @property
    def num_instances(self) -> int:
        return self.pool.num_instances

    @property
    def capacity(self) -> int:
        return self.pool.capacity

    @property
    def in_degree(self) -> int:
        return len(self.in_edges)

    @property
    def out_degree(self) -> int:
        return len(self.out_edges)

    @property
    def needs_scheduler_init(self) -> bool:
        """Whether this node runs a stage that mutates the diffusion scheduler.

        The classic implementation keyed this off ``role == DENOISER``.  With N
        pools a topology can have several denoising nodes (LTX-2 stage1 and
        stage2, for example), so it is derived from the stages themselves --
        unless the node declares no stages, in which case the spec must say.
        """
        if self.scheduler_override is not None:
            return self.scheduler_override
        return any(
            marker in stage.lower()
            for stage in self.stages
            for marker in _SCHEDULER_STAGE_MARKERS
        )


class ExecutionPlan:
    """A validated DAG topology ready for execution."""

    def __init__(
        self,
        nodes: dict[str, CompiledNode],
        edges: list[CompiledEdge],
        source: str,
        order: list[str],
        max_inflight: int | None = None,
    ):
        self._nodes = nodes
        self._edges = edges
        self._source = source
        self._order = order
        self._max_inflight = max_inflight

        self._stage_owner: dict[str, str] = {}
        for node in nodes.values():
            for stage in node.stages:
                self._stage_owner[stage] = node.name

        self._edges_by_id = {e.edge_id: e for e in edges}
        self._terminals = [n.name for n in nodes.values() if n.terminal]

    # ------------------------------------------------------------------
    # Graph accessors
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        return self._source

    @property
    def node_names(self) -> list[str]:
        return list(self._order)

    @property
    def nodes(self) -> dict[str, CompiledNode]:
        return self._nodes

    @property
    def edges(self) -> list[CompiledEdge]:
        return list(self._edges)

    @property
    def terminals(self) -> list[str]:
        return list(self._terminals)

    @property
    def max_inflight(self) -> int | None:
        return self._max_inflight

    def node(self, name: str) -> CompiledNode:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(
                f"Unknown DAG node {name!r}; known nodes: {sorted(self._nodes)}"
            ) from None

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def edge(self, edge_id: str) -> CompiledEdge | None:
        return self._edges_by_id.get(edge_id)

    def stage_owner(self, stage_name: str) -> str | None:
        """Which node runs a given registered stage, or None if unassigned."""
        return self._stage_owner.get(stage_name)

    def is_source(self, name: str) -> bool:
        return name == self._source

    def is_terminal(self, name: str) -> bool:
        return self.node(name).terminal

    def is_sender(self, name: str) -> bool:
        return self.node(name).out_degree > 0

    def is_receiver(self, name: str) -> bool:
        return self.node(name).in_degree > 0

    def live_out_edges(self, name: str, context: dict[str, Any]) -> list[CompiledEdge]:
        """Out-edges whose predicate holds for a request with this metadata."""
        return [e for e in self.node(name).out_edges if e.is_live(context)]

    # ------------------------------------------------------------------
    # Coverage validation against a live pipeline
    # ------------------------------------------------------------------

    def validate_stage_coverage(self, registered_stages: list[str]) -> list[str]:
        """Check the plan against the stages a pipeline actually registers.

        This is the guard against config/code drift: a stage that no node
        claims would silently never run, and a stage two nodes claim would run
        twice.  Returns a list of human-readable errors.
        """
        errors: list[str] = []
        registered = set(registered_stages)

        for stage, owner in sorted(self._stage_owner.items()):
            if stage not in registered:
                errors.append(
                    f"Node '{owner}' claims stage '{stage}', which the pipeline "
                    f"does not register"
                )

        for stage in registered_stages:
            if stage not in self._stage_owner:
                errors.append(f"Stage '{stage}' is not claimed by any node")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self._source,
            "order": list(self._order),
            "nodes": {
                name: {
                    "stages": list(n.stages),
                    "modules": list(n.modules) if n.modules is not None else None,
                    "terminal": n.terminal,
                    "join": n.join,
                    "instances": n.num_instances,
                    "capacity": n.capacity,
                }
                for name, n in self._nodes.items()
            },
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "when": e.when,
                    "fields": e.fields,
                }
                for e in self._edges
            ],
        }

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    @classmethod
    def compile(cls, spec: DagSpec) -> ExecutionPlan:
        errors: list[str] = []

        role_names = [r.name for r in spec.roles]
        duplicates = {n for n in role_names if role_names.count(n) > 1}
        if duplicates:
            errors.append(f"Duplicate role names: {sorted(duplicates)}")
        if not spec.roles:
            raise PlanValidationError(["DAG must define at least one role"])

        pools: dict[str, PoolSpec] = {}
        for role in spec.roles:
            pool = spec.get_pool(role.name)
            if pool is None:
                pool = PoolSpec(role=role.name)
            pools[role.name] = pool

        for pool in spec.pools:
            if pool.role not in role_names:
                errors.append(f"Pool references unknown role '{pool.role}'")

        edges: list[CompiledEdge] = []
        seen_edges: set[str] = set()
        for route in spec.routes:
            if route.src not in role_names:
                errors.append(f"Route '{route.edge_id}' has unknown src '{route.src}'")
                continue
            if route.dst not in role_names:
                errors.append(f"Route '{route.edge_id}' has unknown dst '{route.dst}'")
                continue
            if route.src == route.dst:
                errors.append(f"Route '{route.edge_id}' is a self-loop")
                continue
            if route.edge_id in seen_edges:
                errors.append(f"Duplicate route '{route.edge_id}'")
                continue
            seen_edges.add(route.edge_id)

            parsed: ast.Expression | None = None
            if route.when is not None:
                try:
                    parsed = compile_predicate(route.when)
                except PredicateError as e:
                    errors.append(str(e))
            edges.append(
                CompiledEdge(
                    src=route.src,
                    dst=route.dst,
                    when=route.when,
                    fields=list(route.fields) if route.fields is not None else None,
                    _ast=parsed,
                )
            )

        nodes: dict[str, CompiledNode] = {}
        for role in spec.roles:
            nodes[role.name] = CompiledNode(
                name=role.name,
                stages=role.stage_names,
                modules=list(role.modules) if role.modules is not None else None,
                join=role.join,
                terminal=role.terminal,
                emit=list(role.emit) if role.emit is not None else None,
                pool=pools[role.name],
                scheduler_override=role.needs_scheduler,
            )

        for edge in edges:
            nodes[edge.src].out_edges.append(edge)
            nodes[edge.dst].in_edges.append(edge)

        errors.extend(cls._validate_stage_ownership(spec))
        errors.extend(cls._validate_source(spec, nodes))
        errors.extend(cls._validate_terminals(nodes))
        errors.extend(cls._validate_pools(nodes))

        order, cycle_errors = cls._topological_order(nodes)
        errors.extend(cycle_errors)

        if errors:
            raise PlanValidationError(errors)

        return cls(
            nodes=nodes,
            edges=edges,
            source=spec.source,
            order=order,
            max_inflight=spec.max_inflight,
        )

    @staticmethod
    def _validate_stage_ownership(spec: DagSpec) -> list[str]:
        errors: list[str] = []
        owner: dict[str, str] = {}
        for role in spec.roles:
            for stage in role.stage_names:
                if stage in owner:
                    errors.append(
                        f"Stage '{stage}' is claimed by both '{owner[stage]}' "
                        f"and '{role.name}'; each stage must belong to exactly one role"
                    )
                else:
                    owner[stage] = role.name
        return errors

    @staticmethod
    def _validate_source(spec: DagSpec, nodes: dict[str, CompiledNode]) -> list[str]:
        if not spec.source:
            return ["DAG must declare a 'source' node (the request entry point)"]
        if spec.source not in nodes:
            return [f"Source '{spec.source}' is not a declared role"]
        if nodes[spec.source].in_degree > 0:
            return [f"Source '{spec.source}' must have no incoming routes"]

        unreachable = ExecutionPlan._unreachable_from(spec.source, nodes)
        if unreachable:
            return [
                f"Nodes unreachable from source '{spec.source}': {sorted(unreachable)}"
            ]
        return []

    @staticmethod
    def _unreachable_from(source: str, nodes: dict[str, CompiledNode]) -> set[str]:
        seen = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            for edge in nodes[current].out_edges:
                if edge.dst not in seen:
                    seen.add(edge.dst)
                    stack.append(edge.dst)
        return set(nodes) - seen

    @staticmethod
    def _validate_terminals(nodes: dict[str, CompiledNode]) -> list[str]:
        errors: list[str] = []
        terminals = [n for n in nodes.values() if n.terminal]
        if not terminals:
            errors.append("DAG must declare at least one terminal node")

        for node in nodes.values():
            if node.out_degree == 0 and not node.terminal:
                errors.append(
                    f"Node '{node.name}' has no outgoing routes but is not "
                    f"marked terminal; its output would be discarded"
                )
            if node.terminal and node.out_degree > 0:
                errors.append(
                    f"Terminal node '{node.name}' must not have outgoing routes; "
                    f"it returns output to the client rather than to another node"
                )

        # Terminal nodes each contribute a slice of the final OutputBatch, so
        # two of them writing the same field would race.
        claimed: dict[str, str] = {}
        for node in terminals:
            for field_name in node.emit or []:
                if field_name in claimed:
                    errors.append(
                        f"Terminal nodes '{claimed[field_name]}' and '{node.name}' "
                        f"both emit output field '{field_name}'"
                    )
                else:
                    claimed[field_name] = node.name
        return errors

    @staticmethod
    def _validate_pools(nodes: dict[str, CompiledNode]) -> list[str]:
        errors: list[str] = []
        for node in nodes.values():
            if node.num_instances < 1:
                errors.append(
                    f"Node '{node.name}' has no pool instances; declare at "
                    f"least one work endpoint under pools[].urls"
                )
            if node.capacity < 1:
                errors.append(
                    f"Node '{node.name}' has capacity {node.capacity}; "
                    f"must be at least 1"
                )
            if node.join == "any" and node.in_degree > 1:
                # "any" would let a node start before its other producers
                # resolve, leaving their staged buffers with no reader.
                errors.append(
                    f"Node '{node.name}' uses join='any', which is not yet "
                    f"supported; use join='all' (a pruned edge still counts as "
                    f"resolved, so conditional branches work)"
                )
        return errors

    @staticmethod
    def _topological_order(
        nodes: dict[str, CompiledNode],
    ) -> tuple[list[str], list[str]]:
        in_degree = {name: node.in_degree for name, node in nodes.items()}
        ready = sorted(n for n, d in in_degree.items() if d == 0)
        order: list[str] = []

        while ready:
            current = ready.pop(0)
            order.append(current)
            for edge in nodes[current].out_edges:
                in_degree[edge.dst] -= 1
                if in_degree[edge.dst] == 0:
                    ready.append(edge.dst)
            ready.sort()

        if len(order) != len(nodes):
            remaining = sorted(set(nodes) - set(order))
            return order, [f"DAG contains a cycle involving: {remaining}"]
        return order, []

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    @classmethod
    def from_classic_roles(
        cls,
        encoder_urls: list[str],
        denoiser_urls: list[str],
        decoder_urls: list[str],
        *,
        encoder_capacity: int = 4,
        denoiser_capacity: int = 2,
        decoder_capacity: int = 4,
        dispatch_policy: str = "round_robin",
        encoder_result_endpoint: str | None = None,
        denoiser_result_endpoint: str | None = None,
        decoder_result_endpoint: str | None = None,
        stages: dict[str, list[str]] | None = None,
    ) -> ExecutionPlan:
        """Compile the legacy encoder/denoiser/decoder topology into a plan.

        Stage assignment stays with ``role_affinity`` in this mode (the
        pipeline filter falls back to it when a node declares no stages), so
        existing deployments keep working without listing every stage in
        config.
        """
        stages = stages or {}

        def role(
            name: str,
            modules: list[str],
            *,
            terminal: bool = False,
            needs_scheduler: bool = False,
        ) -> RoleSpec:
            return RoleSpec(
                name=name,
                stages=[StageSpec(name=s) for s in stages.get(name, [])],
                modules=modules,
                terminal=terminal,
                emit=(["output", "audio", "audio_sample_rate"] if terminal else None),
                needs_scheduler=needs_scheduler,
            )

        spec = DagSpec(
            source="encoder",
            roles=[
                role("encoder", _CLASSIC_ENCODER_MODULES),
                role("denoiser", _CLASSIC_DENOISER_MODULES, needs_scheduler=True),
                role("decoder", _CLASSIC_DECODER_MODULES, terminal=True),
            ],
            pools=[
                PoolSpec(
                    role="encoder",
                    urls=list(encoder_urls),
                    capacity=encoder_capacity,
                    dispatch_policy=dispatch_policy,
                    result_endpoint=encoder_result_endpoint,
                ),
                PoolSpec(
                    role="denoiser",
                    urls=list(denoiser_urls),
                    capacity=denoiser_capacity,
                    dispatch_policy=dispatch_policy,
                    result_endpoint=denoiser_result_endpoint,
                ),
                PoolSpec(
                    role="decoder",
                    urls=list(decoder_urls),
                    capacity=decoder_capacity,
                    dispatch_policy=dispatch_policy,
                    result_endpoint=decoder_result_endpoint,
                ),
            ],
            routes=[
                RouteSpec(src="encoder", dst="denoiser"),
                RouteSpec(src="denoiser", dst="decoder"),
            ],
        )
        return cls.compile(spec)
