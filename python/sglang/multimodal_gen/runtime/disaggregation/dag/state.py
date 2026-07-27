# SPDX-License-Identifier: Apache-2.0
"""Per-request execution state for a DAG topology.

One ``RequestDagState`` tracks where a single request is in the graph: which
nodes have run, which edges turned out to be live, which instance each node
was bound to, and the partial outputs collected from terminal nodes.

This module is pure bookkeeping with no I/O so the traversal rules -- in
particular skip propagation, which is easy to get subtly wrong -- can be
tested directly.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any


class NodeStatus(enum.Enum):
    """Where one DAG node sits for one request."""

    PENDING = "pending"  # upstream has not resolved yet
    WAITING = "waiting"  # inputs ready, waiting for a free slot
    RUNNING = "running"  # dispatched to an instance
    DONE = "done"
    SKIPPED = "skipped"  # every incoming edge was pruned


_RESOLVED = (NodeStatus.DONE, NodeStatus.SKIPPED)


@dataclass
class TransferHandle:
    """Where a producer's staged output lives and what it contains."""

    src_node: str = ""
    src_instance: int = -1
    session_id: str = ""
    pool_ptr: int = 0
    slot_offset: int = 0
    data_size: int = 0
    manifest: dict = field(default_factory=dict)
    scalar_fields: dict = field(default_factory=dict)


@dataclass
class InputHandle:
    """One arrived input for a node, tagged with the edge it came from."""

    edge_id: str
    transfer: TransferHandle


class RequestDagState:
    """Traversal state for a single request."""

    def __init__(
        self,
        request_id: str,
        client_identity: bytes | None,
        node_names: list[str],
        in_degrees: dict[str, int],
        route_ctx: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ):
        self.request_id = request_id
        self.client_identity = client_identity
        self.route_ctx: dict[str, Any] = dict(route_ctx or {})
        # The original pickled request, consumed by the entry node's dispatch.
        self.payload = payload
        self.submit_time = time.monotonic()
        self.error: str | None = None

        self.node_status: dict[str, NodeStatus] = {
            name: NodeStatus.PENDING for name in node_names
        }
        self.node_instance: dict[str, int] = {}
        self.remaining_inputs: dict[str, int] = dict(in_degrees)
        self.arrived_inputs: dict[str, list[InputHandle]] = {
            name: [] for name in node_names
        }
        self.partial_output: dict[str, Any] = {}

        # Out-edges of a node that have been dispatched but whose RDMA push has
        # not completed.  The producer's slot is held until this drains, since
        # a fan-out reads the same staged buffer several times.
        self.pending_pushes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def status(self, node: str) -> NodeStatus:
        return self.node_status[node]

    def is_resolved(self, node: str) -> bool:
        return self.node_status[node] in _RESOLVED

    def all_resolved(self) -> bool:
        return all(s in _RESOLVED for s in self.node_status.values())

    def any_done(self) -> bool:
        return any(s is NodeStatus.DONE for s in self.node_status.values())

    def elapsed_s(self) -> float:
        return time.monotonic() - self.submit_time

    def held_slots(self) -> list[tuple[str, int]]:
        """Nodes currently holding a capacity slot, as (node, instance) pairs.

        A node holds its slot from the moment it is bound (which for a join is
        when its first input arrives, before it is dispatched) until it
        completes, so cancellation has to release both RUNNING and WAITING
        nodes.
        """
        return [
            (node, instance)
            for node, instance in self.node_instance.items()
            if self.node_status[node] in (NodeStatus.WAITING, NodeStatus.RUNNING)
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def merge_route_context(self, scalar_fields: dict[str, Any] | None) -> None:
        """Fold a node's scalar output into the predicate evaluation context.

        Downstream predicates see values as they are after upstream ran, which
        is what makes a resolution-dependent route work when an upstream stage
        is the thing that changes the resolution.
        """
        if not scalar_fields:
            return
        for key, value in scalar_fields.items():
            if not key.startswith("_"):
                self.route_ctx[key] = value

    def bind_instance(self, node: str, instance: int) -> None:
        self.node_instance[node] = instance

    def mark(self, node: str, status: NodeStatus) -> None:
        self.node_status[node] = status

    def add_input(self, node: str, handle: InputHandle) -> None:
        self.arrived_inputs[node].append(handle)

    def consume_input(self, edge_id: str) -> None:
        """Record that one incoming edge of the destination has resolved."""
        dst = edge_id.split("->", 1)[1]
        self.remaining_inputs[dst] = max(0, self.remaining_inputs[dst] - 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "elapsed_s": round(self.elapsed_s(), 3),
            "nodes": {n: s.value for n, s in self.node_status.items()},
            "instances": dict(self.node_instance),
            "error": self.error,
        }
