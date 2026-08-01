# SPDX-License-Identifier: Apache-2.0
"""User-facing DAG topology description.

A deployment is described by four kinds of spec:

``StageSpec``  one pipeline stage, referenced by its registered name.
``RoleSpec``   a DAG node: the set of stages that run together on one pool.
``PoolSpec``   the physical instances backing a role.
``RouteSpec``  a DAG edge, optionally guarded by a routing predicate.

These are plain data.  Validation and graph analysis live in ``plan.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

JoinPolicy = Literal["all", "any"]


@dataclass
class StageSpec:
    """One pipeline stage assigned to a role.

    ``name`` must match the stage's registered name exactly (the name passed
    to ``add_stage``, defaulting to the class name).  Glob patterns are
    deliberately not supported: silent non-matches were the main source of
    drift between deployment config and pipeline code.
    """

    name: str
    emit: list[str] | None = None

    @classmethod
    def parse(cls, data: Any) -> StageSpec:
        if isinstance(data, str):
            return cls(name=data)
        return cls(name=data["name"], emit=data.get("emit"))

    def to_dict(self) -> dict[str, Any]:
        if self.emit is None:
            return {"name": self.name}
        return {"name": self.name, "emit": list(self.emit)}


@dataclass
class RoleSpec:
    """A DAG node: a named unit of computation."""

    name: str
    stages: list[StageSpec] = field(default_factory=list)
    modules: list[str] | None = None
    join: JoinPolicy = "all"
    terminal: bool = False
    emit: list[str] | None = None
    # Overrides the "does this node run a denoising stage" inference, which is
    # needed when stages are assigned by role_affinity rather than listed here.
    needs_scheduler: bool | None = None

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    @classmethod
    def parse(cls, data: dict[str, Any]) -> RoleSpec:
        return cls(
            name=data["name"],
            stages=[StageSpec.parse(s) for s in data.get("stages", [])],
            modules=data.get("modules"),
            join=data.get("join", "all"),
            terminal=bool(data.get("terminal", False)),
            emit=data.get("emit"),
            needs_scheduler=data.get("needs_scheduler"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "stages": [s.to_dict() for s in self.stages],
        }
        if self.modules is not None:
            out["modules"] = list(self.modules)
        if self.join != "all":
            out["join"] = self.join
        if self.terminal:
            out["terminal"] = True
        if self.emit is not None:
            out["emit"] = list(self.emit)
        if self.needs_scheduler is not None:
            out["needs_scheduler"] = self.needs_scheduler
        return out


@dataclass
class PoolSpec:
    """Physical instances backing a role.

    ``urls`` are the per-instance work endpoints, in the same order the
    orchestrator builds its PUSH sockets; instance indices refer to this list.
    """

    role: str
    urls: list[str] = field(default_factory=list)
    parallelism: dict[str, int] = field(default_factory=dict)
    capacity: int = 4
    dispatch_policy: str = "round_robin"
    prealloc_slots: int = 2
    result_endpoint: str | None = None

    @property
    def num_instances(self) -> int:
        return len(self.urls)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> PoolSpec:
        return cls(
            role=data["role"],
            urls=list(data.get("urls", [])),
            parallelism=dict(data.get("parallelism", {})),
            capacity=int(data.get("capacity", 4)),
            dispatch_policy=data.get("dispatch_policy", "round_robin"),
            prealloc_slots=int(data.get("prealloc_slots", 2)),
            result_endpoint=data.get("result_endpoint"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "urls": list(self.urls)}
        if self.parallelism:
            out["parallelism"] = dict(self.parallelism)
        if self.capacity != 4:
            out["capacity"] = self.capacity
        if self.dispatch_policy != "round_robin":
            out["dispatch_policy"] = self.dispatch_policy
        if self.prealloc_slots != 2:
            out["prealloc_slots"] = self.prealloc_slots
        if self.result_endpoint is not None:
            out["result_endpoint"] = self.result_endpoint
        return out


@dataclass
class RouteSpec:
    """A DAG edge from ``src`` to ``dst``.

    ``when`` is a predicate over per-request scalar metadata; when it is
    ``None`` the edge is unconditional.  ``fields`` restricts which Req fields
    travel on this edge; since the transfer manifest is per-field, restricting
    an edge is a manifest filter and needs no extra staging.
    """

    src: str
    dst: str
    when: str | None = None
    fields: list[str] | None = None

    @property
    def edge_id(self) -> str:
        return f"{self.src}->{self.dst}"

    @classmethod
    def parse(cls, data: dict[str, Any]) -> RouteSpec:
        return cls(
            src=data["src"],
            dst=data["dst"],
            when=data.get("when"),
            fields=data.get("fields"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"src": self.src, "dst": self.dst}
        if self.when is not None:
            out["when"] = self.when
        if self.fields is not None:
            out["fields"] = list(self.fields)
        return out


@dataclass
class DagSpec:
    """A complete disaggregation topology."""

    roles: list[RoleSpec] = field(default_factory=list)
    pools: list[PoolSpec] = field(default_factory=list)
    routes: list[RouteSpec] = field(default_factory=list)
    source: str = ""
    max_inflight: int | None = None

    def get_role(self, name: str) -> RoleSpec | None:
        for r in self.roles:
            if r.name == name:
                return r
        return None

    def get_pool(self, role: str) -> PoolSpec | None:
        for p in self.pools:
            if p.role == role:
                return p
        return None

    @classmethod
    def parse(cls, data: dict[str, Any]) -> DagSpec:
        roles = [RoleSpec.parse(r) for r in data.get("roles", [])]
        source = data.get("source", "")
        if not source and roles:
            # Convenience: a single-source DAG need not name its entry node.
            dsts = {r["dst"] for r in data.get("routes", [])}
            candidates = [r.name for r in roles if r.name not in dsts]
            if len(candidates) == 1:
                source = candidates[0]
        return cls(
            roles=roles,
            pools=[PoolSpec.parse(p) for p in data.get("pools", [])],
            routes=[RouteSpec.parse(r) for r in data.get("routes", [])],
            source=source,
            max_inflight=data.get("max_inflight"),
        )

    @classmethod
    def load(cls, source: str) -> DagSpec:
        """Load a topology from a file path or an inline JSON/YAML string.

        A leading ``@`` forces the file interpretation, which matters for the
        rare path that would otherwise parse as inline YAML.
        """
        path: str | None = None
        if source.startswith("@"):
            path = source[1:]
        elif not source.lstrip().startswith("{") and os.path.exists(source):
            path = source

        if path is not None:
            with open(path) as f:
                return cls._parse_text(f.read(), path)
        return cls._parse_text(source, "<inline>")

    @classmethod
    def _parse_text(cls, raw: str, origin: str) -> DagSpec:
        stripped = raw.lstrip()
        if stripped.startswith("{"):
            return cls.parse(json.loads(raw))
        try:
            import yaml
        except ImportError:
            raise ValueError(
                f"DAG spec {origin} is not JSON and PyYAML is not installed; "
                "install pyyaml or provide JSON"
            ) from None
        return cls.parse(yaml.safe_load(raw))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "roles": [r.to_dict() for r in self.roles],
            "pools": [p.to_dict() for p in self.pools],
            "routes": [r.to_dict() for r in self.routes],
        }
        if self.max_inflight is not None:
            out["max_inflight"] = self.max_inflight
        return out
