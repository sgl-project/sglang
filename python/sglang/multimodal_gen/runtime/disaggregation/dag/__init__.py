# SPDX-License-Identifier: Apache-2.0
"""Declarative DAG topology for disaggregated diffusion pipelines.

A ``DagSpec`` describes the deployment as a directed acyclic graph of named
compute nodes (``RoleSpec``) backed by GPU pools (``PoolSpec``) and connected
by edges (``RouteSpec``) that may carry a routing predicate.  The legacy
encoder/denoiser/decoder topology is one particular three-node linear DAG.

``ExecutionPlan`` is the compiled, validated form consumed by the runtime;
``DagRequestScheduler`` executes it without knowing any role names.
"""

from sglang.multimodal_gen.runtime.disaggregation.dag.plan import (
    CompiledEdge,
    CompiledNode,
    ExecutionPlan,
    PlanValidationError,
)
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

__all__ = [
    "CompiledEdge",
    "CompiledNode",
    "DagSpec",
    "ExecutionPlan",
    "JoinPolicy",
    "PlanValidationError",
    "PoolSpec",
    "PredicateError",
    "RoleSpec",
    "RouteSpec",
    "StageSpec",
    "compile_predicate",
    "evaluate_predicate",
]
