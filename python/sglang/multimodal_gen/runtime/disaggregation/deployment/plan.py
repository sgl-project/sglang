# SPDX-License-Identifier: Apache-2.0
"""Compile stage contracts and placement into a validated execution plan."""

from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

import msgspec

REQUEST_CONTEXT = "__request_context__"


class StageSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """stage-local data dependencies and components, independent of placement"""

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    after: tuple[str, ...] | None = None
    colocate_with: tuple[str, ...] = ()


class RoleSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    stages: tuple[str | int, ...]
    endpoints: tuple[str, ...]
    node: str = "local"
    server_args: dict = {}


class Deployment(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    roles: dict[str, RoleSpec]
    stages: tuple[StageSpec, ...] = ()
    outputs: tuple[str, ...] | None = None

    @classmethod
    def load(cls, path: str) -> "Deployment":
        return msgspec.json.decode(Path(path).read_bytes(), type=cls)


@dataclass(frozen=True)
class RolePlan:
    name: str
    stages: tuple[str, ...]
    components: frozenset[str]
    # input fields are bound to the role which produces their latest value
    inputs: dict[str, str]
    outputs: tuple[str, ...]
    dependencies: frozenset[str]
    endpoints: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    """resolved role graph shared by the coordinator and workers"""

    roles: dict[str, RolePlan]
    order: tuple[str, ...]
    outputs: dict[str, str]

    @classmethod
    def compile(
        cls, deployment: Deployment, defaults: tuple[StageSpec, ...] = ()
    ) -> "ExecutionPlan":
        stages = deployment.stages or defaults
        names = [stage.name for stage in stages]
        if (
            not names
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("Stage contracts must have unique, nonempty names")
        specs = {stage.name: stage for stage in stages}
        # 1. resolve selectors before filtering so indices always name the full pipeline
        owner = {}
        endpoints = set()
        for name, role in deployment.roles.items():
            if not name or not role.stages or not role.endpoints:
                raise ValueError("Each role needs a name, stages, and worker endpoints")
            for endpoint in role.endpoints:
                if not endpoint.startswith("tcp://") or endpoint in endpoints:
                    raise ValueError(
                        f"Invalid or duplicate worker endpoint: {endpoint}"
                    )
                endpoints.add(endpoint)
            for selector in role.stages:
                if type(selector) is int:
                    if not 0 <= selector < len(names):
                        raise ValueError(
                            f"Stage index {selector} is outside [0, {len(names)})"
                        )
                    stage = names[selector]
                elif isinstance(selector, str):
                    stage = selector
                else:
                    raise ValueError("Stage selectors must be names or integer indices")
                if stage not in specs or stage in owner:
                    raise ValueError(f"Unknown or multiply assigned stage: {stage}")
                owner[stage] = name

        if not any(stage.inputs or stage.outputs for stage in stages):
            return cls._compile_context(deployment, stages, owner)

        # 2. bind each read to its preceding writer and retain each role's final writer
        selected = set(owner)
        producers = {}
        final_writers = {}
        stage_dependencies = {}
        bindings = {}
        for index, stage in enumerate(stages):
            if stage.name not in selected:
                continue
            dependencies = set(
                stage.after if stage.after is not None else names[index - 1 : index]
            )
            inputs = {}
            for path in stage.inputs:
                if path in producers:
                    producer = producers[path]
                    dependencies.add(producer)
                    inputs[path] = producer
            if not dependencies <= selected:
                raise ValueError(
                    f"Stage {stage.name} has missing dependencies: {dependencies - selected}"
                )
            if not dependencies <= set(names[:index]):
                raise ValueError(
                    f"Stage {stage.name} has a forward dependency; order stage contracts topologically"
                )
            for other in stage.colocate_with:
                if owner.get(other) != owner[stage.name]:
                    raise ValueError(f"Stages {stage.name} and {other} must colocate")
            bindings[stage.name] = inputs
            stage_dependencies[stage.name] = dependencies
            for path in stage.outputs:
                if path in producers:
                    # a mutation must be ordered after its previous writer
                    dependencies.add(producers[path])
                producers[path] = stage.name
                final_writers[owner[stage.name], path] = stage.name

        # repeated output fields can add self-edges even after validating explicit edges
        tuple(TopologicalSorter(stage_dependencies).static_order())
        # 3. export only fields crossing role boundaries; colocation must not create cycles
        dependencies = {name: set() for name in deployment.roles}
        role_inputs = {name: {} for name in deployment.roles}
        role_outputs = {name: set() for name in deployment.roles}
        for stage, upstream in stage_dependencies.items():
            role = owner[stage]
            dependencies[role].update(
                owner[item] for item in upstream if owner[item] != role
            )
            for path, producer in bindings[stage].items():
                source = owner[producer]
                if source != role:
                    if final_writers[source, path] != producer:
                        raise ValueError(
                            f"Stage {stage} needs an overwritten version of {path}; split role {source}"
                        )
                    previous = role_inputs[role].setdefault(path, source)
                    if previous != source:
                        raise ValueError(
                            f"Role {role} needs multiple versions of {path}; split it"
                        )
                    role_outputs[source].add(path)
        outputs = {}
        output_paths = (
            deployment.outputs
            if deployment.outputs is not None
            else tuple(path for path in producers if path.startswith("result."))
        )
        if not output_paths:
            raise ValueError("Declare at least one terminal output for this deployment")
        for path in output_paths:
            if path not in producers:
                raise ValueError(f"No selected stage produces output {path}")
            source = owner[producers[path]]
            outputs[path] = source
            role_outputs[source].add(path)
        order = tuple(TopologicalSorter(dependencies).static_order())
        roles = {
            name: RolePlan(
                name=name,
                stages=tuple(stage for stage in names if owner.get(stage) == name),
                components=frozenset(
                    component
                    for stage in names
                    if owner.get(stage) == name
                    for component in specs[stage].components
                ),
                inputs=role_inputs[name],
                outputs=tuple(sorted(role_outputs[name])),
                dependencies=frozenset(dependencies[name]),
                endpoints=role.endpoints,
            )
            for name, role in deployment.roles.items()
        }
        return cls(roles, order, outputs)

    @classmethod
    def _compile_context(cls, deployment, stages, owner):
        """carry one mutable request through consecutive groups without merging branches"""
        names = [stage.name for stage in stages]
        if set(owner) != set(names):
            raise ValueError(
                "Automatic request transfer requires every stage to be placed"
            )
        order = []
        for index, stage in enumerate(stages):
            if stage.after is not None and stage.after != tuple(
                names[index - 1 : index]
            ):
                raise ValueError(
                    "Automatic request transfer preserves sequential stage order"
                )
            role = owner[stage.name]
            if not order or order[-1] != role:
                if role in order:
                    raise ValueError("A role must contain consecutive stages")
                order.append(role)
            for other in stage.colocate_with:
                if owner.get(other) != role:
                    raise ValueError(f"Stages {stage.name} and {other} must colocate")
        output_paths = (
            ("result.output",) if deployment.outputs is None else deployment.outputs
        )
        if not output_paths:
            raise ValueError("Declare at least one terminal output for this deployment")
        outputs = {path: order[-1] for path in output_paths}
        roles = {}
        for index, name in enumerate(order):
            previous = order[index - 1] if index else None
            selected = [stage for stage in stages if owner[stage.name] == name]
            roles[name] = RolePlan(
                name=name,
                stages=tuple(stage.name for stage in selected),
                components=frozenset(c for stage in selected for c in stage.components),
                inputs={REQUEST_CONTEXT: previous} if previous else {},
                outputs=tuple(outputs) if name == order[-1] else (REQUEST_CONTEXT,),
                dependencies=frozenset((previous,)) if previous else frozenset(),
                endpoints=deployment.roles[name].endpoints,
            )
        return cls(roles, tuple(order), outputs)
