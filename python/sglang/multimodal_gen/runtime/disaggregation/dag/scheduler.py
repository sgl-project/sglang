# SPDX-License-Identifier: Apache-2.0
"""Request scheduling over a DAG topology.

``DagRequestScheduler`` owns everything the orchestrator used to hardcode for
three roles: per-node capacity, dispatch policies, ready queues, and the
traversal rules that decide what runs next.  It performs no I/O -- every
decision comes back as an action for the caller to execute over ZMQ -- which
keeps the interesting logic (skip propagation, join readiness, fan-out slot
accounting) unit-testable without a cluster.

Traversal rules, in one place:

* A node becomes dispatchable when every incoming edge has resolved, either by
  delivering an input or by being pruned.  Waiting for *all* edges to resolve
  before dispatching any of them is what lets a join know how many inputs to
  expect, which in turn lets a predicate depend on a value an upstream node
  only produces mid-flight.
* A node whose every incoming edge was pruned is skipped, and skipping
  propagates downstream.
* A producer holds its capacity slot until each of its live outgoing edges has
  finished its RDMA push, because a fan-out reads one staged buffer N times.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.runtime.disaggregation.dag.plan import (
    CompiledEdge,
    ExecutionPlan,
)
from sglang.multimodal_gen.runtime.disaggregation.dag.state import (
    InputHandle,
    NodeStatus,
    RequestDagState,
    TransferHandle,
)
from sglang.multimodal_gen.runtime.disaggregation.dispatch_policy import (
    create_dispatch_policy,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Actions returned to the orchestrator
# ----------------------------------------------------------------------


@dataclass
class SourceDispatch:
    """Send the raw pickled request to the entry node."""

    request_id: str
    node: str
    instance: int
    payload: bytes


@dataclass
class EdgeTransfer:
    """Begin moving one producer's output along one edge."""

    request_id: str
    edge: CompiledEdge
    src_node: str
    src_instance: int
    dst_node: str
    dst_instance: int
    input_index: int
    expected_inputs: int
    transfer: TransferHandle
    # How many edges share the producer's staged buffer, so it can refcount
    # its slot and free it only after the last push.
    fanout_total: int = 1

    @property
    def edge_id(self) -> str:
        return self.edge.edge_id


@dataclass
class CompleteRequest:
    """Every live terminal has reported; return the merged output."""

    request_id: str
    client_identity: bytes | None
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailRequest:
    request_id: str
    client_identity: bytes | None
    error: str


Action = SourceDispatch | EdgeTransfer | CompleteRequest | FailRequest


@dataclass
class _Admitted:
    request_id: str
    client_identity: bytes | None
    payload: bytes
    route_ctx: dict[str, Any]


class DagRequestScheduler:
    """Executes an ``ExecutionPlan`` for many concurrent requests."""

    def __init__(
        self,
        plan: ExecutionPlan,
        *,
        max_inflight: int | None = None,
        timeout_s: float = 600.0,
    ):
        self._plan = plan
        self._timeout_s = timeout_s
        self._lock = threading.RLock()

        self._free_slots: dict[str, list[int]] = {}
        self._policies: dict[str, Any] = {}
        self._ready: dict[str, deque[str]] = {}

        for name in plan.node_names:
            node = plan.node(name)
            self._free_slots[name] = [node.capacity] * node.num_instances
            self._policies[name] = create_dispatch_policy(
                node.pool.dispatch_policy, node.num_instances
            )
            self._ready[name] = deque()

        if max_inflight is None:
            max_inflight = plan.max_inflight
        if max_inflight is None:
            # Bound concurrency by the scarcest pool.  Without this, a request
            # can hold a slot on a join node while its slow sibling branch
            # queues behind other requests' reservations.
            max_inflight = min(
                plan.node(n).capacity * max(plan.node(n).num_instances, 1)
                for n in plan.node_names
            )
        self._max_inflight = max(1, max_inflight)

        self._states: dict[str, RequestDagState] = {}
        self._admission: deque[_Admitted] = deque()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    def get(self, request_id: str) -> RequestDagState | None:
        return self._states.get(request_id)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "inflight": len(self._states),
                "max_inflight": self._max_inflight,
                "admission_queue": len(self._admission),
                "nodes": {
                    name: {
                        "instances": self._plan.node(name).num_instances,
                        "free_slots": list(self._free_slots[name]),
                        "ready_depth": len(self._ready[name]),
                    }
                    for name in self._plan.node_names
                },
                "requests": [s.snapshot() for s in self._states.values()],
            }

    # ------------------------------------------------------------------
    # Ingress
    # ------------------------------------------------------------------

    def submit(
        self,
        request_id: str,
        client_identity: bytes | None,
        payload: bytes,
        route_ctx: dict[str, Any] | None = None,
    ) -> list[Action]:
        """Admit a new request, or queue it if the cluster is saturated."""
        with self._lock:
            if request_id in self._states:
                raise ValueError(f"Duplicate request_id: {request_id}")

            self._admission.append(
                _Admitted(
                    request_id=request_id,
                    client_identity=client_identity,
                    payload=payload,
                    route_ctx=dict(route_ctx or {}),
                )
            )
            return self._drain_locked()

    # ------------------------------------------------------------------
    # Node lifecycle callbacks
    # ------------------------------------------------------------------

    def on_node_staged(
        self, request_id: str, node: str, transfer: TransferHandle
    ) -> list[Action]:
        """A node finished computing and staged its output for downstream."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                logger.debug("Staged output for unknown request %s", request_id)
                return []

            state.merge_route_context(transfer.scalar_fields)

            compiled = self._plan.node(node)
            live: list[CompiledEdge] = []
            for edge in compiled.out_edges:
                if edge.is_live(state.route_ctx):
                    live.append(edge)
                else:
                    self._prune_edge(state, edge)

            if not live:
                # Every downstream route was pruned; nothing holds this node's
                # buffer, so it can retire immediately.
                self._retire_node(state, node)
                return self._settle(state)

            state.pending_pushes[node] = len(live)
            for edge in live:
                handle = TransferHandle(
                    src_node=node,
                    src_instance=state.node_instance.get(node, -1),
                    session_id=transfer.session_id,
                    pool_ptr=transfer.pool_ptr,
                    slot_offset=transfer.slot_offset,
                    data_size=transfer.data_size,
                    manifest=edge.filter_manifest(transfer.manifest),
                    scalar_fields=edge.filter_scalars(transfer.scalar_fields),
                )
                state.add_input(edge.dst, InputHandle(edge.edge_id, handle))
                state.consume_input(edge.edge_id)
                self._maybe_enqueue(state, edge.dst)

            return self._drain_locked()

    def on_edge_pushed(self, request_id: str, edge_id: str) -> list[Action]:
        """One edge's RDMA push completed; the producer may now release."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return []
            edge = self._plan.edge(edge_id)
            if edge is None:
                return []

            remaining = state.pending_pushes.get(edge.src)
            if remaining is None:
                return []
            remaining -= 1
            state.pending_pushes[edge.src] = remaining
            if remaining > 0:
                return []

            state.pending_pushes.pop(edge.src, None)
            self._retire_node(state, edge.src)
            return self._settle(state)

    def on_terminal_result(
        self,
        request_id: str,
        node: str,
        fields: dict[str, Any],
        error: str | None = None,
    ) -> list[Action]:
        """A terminal node returned its slice of the final output."""
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return []

            if error:
                return self._fail_locked(state, f"{node}: {error}")

            allowed = self._plan.node(node).emit
            for key, value in fields.items():
                if allowed is not None and key not in allowed:
                    continue
                if value is not None:
                    state.partial_output[key] = value

            self._retire_node(state, node)
            return self._settle(state)

    def on_node_error(self, request_id: str, node: str, error: str) -> list[Action]:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return []
            return self._fail_locked(state, f"{node}: {error}")

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------

    def drain(self) -> list[Action]:
        """Dispatch whatever the current free capacity allows."""
        with self._lock:
            return self._drain_locked()

    def check_timeouts(self) -> list[Action]:
        with self._lock:
            expired = [
                s.request_id
                for s in self._states.values()
                if s.elapsed_s() > self._timeout_s
            ]
            actions: list[Action] = []
            for request_id in expired:
                state = self._states.get(request_id)
                if state is None:
                    continue
                actions.extend(
                    self._fail_locked(state, f"timed out after {self._timeout_s:.0f}s")
                )
            return actions

    def cancel(self, request_id: str, error: str = "cancelled") -> list[Action]:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return []
            return self._fail_locked(state, error)

    # ------------------------------------------------------------------
    # Traversal internals
    # ------------------------------------------------------------------

    def _prune_edge(self, state: RequestDagState, edge: CompiledEdge) -> None:
        """Mark one edge dead and propagate the consequences downstream."""
        state.consume_input(edge.edge_id)
        self._maybe_enqueue(state, edge.dst)

    def _maybe_enqueue(self, state: RequestDagState, node: str) -> None:
        """Re-evaluate whether a node is now dispatchable, skipped, or neither."""
        if state.status(node) is not NodeStatus.PENDING:
            return

        if state.remaining_inputs[node] != 0:
            return

        if not state.arrived_inputs[node]:
            self._skip_node(state, node)
            return

        state.mark(node, NodeStatus.WAITING)
        self._ready[node].append(state.request_id)

    def _skip_node(self, state: RequestDagState, node: str) -> None:
        state.mark(node, NodeStatus.SKIPPED)
        for edge in self._plan.node(node).out_edges:
            self._prune_edge(state, edge)

    def _retire_node(self, state: RequestDagState, node: str) -> None:
        state.mark(node, NodeStatus.DONE)
        instance = state.node_instance.get(node)
        if instance is not None:
            self._free_slots[node][instance] += 1

    def _drain_locked(self) -> list[Action]:
        actions: list[Action] = []

        self._admit_locked()

        # Iterate until nothing more can be dispatched: retiring a node can
        # free capacity that unblocks a queue visited earlier in this pass.
        progress = True
        while progress:
            progress = False
            for node in self._plan.node_names:
                queue = self._ready[node]
                while queue:
                    instance = self._policies[node].select_with_capacity(
                        self._free_slots[node]
                    )
                    if instance is None:
                        break
                    request_id = queue.popleft()
                    state = self._states.get(request_id)
                    if state is None or state.status(node) is not NodeStatus.WAITING:
                        continue
                    self._free_slots[node][instance] -= 1
                    state.bind_instance(node, instance)
                    state.mark(node, NodeStatus.RUNNING)
                    actions.extend(self._dispatch(state, node, instance))
                    progress = True

        return actions

    def _dispatch(
        self, state: RequestDagState, node: str, instance: int
    ) -> list[Action]:
        if self._plan.is_source(node):
            return [
                SourceDispatch(
                    request_id=state.request_id,
                    node=node,
                    instance=instance,
                    payload=state.payload,
                )
            ]

        inputs = state.arrived_inputs[node]
        expected = len(inputs)
        actions: list[Action] = []
        for index, handle in enumerate(inputs):
            edge = self._plan.edge(handle.edge_id)
            if edge is None:
                continue
            actions.append(
                EdgeTransfer(
                    request_id=state.request_id,
                    edge=edge,
                    src_node=edge.src,
                    src_instance=handle.transfer.src_instance,
                    dst_node=node,
                    dst_instance=instance,
                    input_index=index,
                    expected_inputs=expected,
                    transfer=handle.transfer,
                    fanout_total=state.pending_pushes.get(edge.src, 1),
                )
            )
        return actions

    def _settle(self, state: RequestDagState) -> list[Action]:
        """Emit completion once every node has reached a terminal status."""
        if not state.all_resolved():
            return self._drain_locked()

        self._states.pop(state.request_id, None)
        actions: list[Action] = [
            CompleteRequest(
                request_id=state.request_id,
                client_identity=state.client_identity,
                fields=dict(state.partial_output),
            )
        ]
        actions.extend(self._drain_locked())
        return actions

    def _fail_locked(self, state: RequestDagState, error: str) -> list[Action]:
        state.error = error
        for node, instance in state.held_slots():
            self._free_slots[node][instance] += 1
            state.mark(node, NodeStatus.SKIPPED)
        for node in self._plan.node_names:
            if not state.is_resolved(node):
                state.mark(node, NodeStatus.SKIPPED)
            try:
                self._ready[node].remove(state.request_id)
            except ValueError:
                pass

        self._states.pop(state.request_id, None)
        actions: list[Action] = [
            FailRequest(
                request_id=state.request_id,
                client_identity=state.client_identity,
                error=error,
            )
        ]
        actions.extend(self._drain_locked())
        return actions

    def _admit_locked(self) -> None:
        while self._admission and len(self._states) < self._max_inflight:
            entry = self._admission.popleft()
            source = self._plan.source
            state = RequestDagState(
                request_id=entry.request_id,
                client_identity=entry.client_identity,
                node_names=self._plan.node_names,
                in_degrees={
                    n: self._plan.node(n).in_degree for n in self._plan.node_names
                },
                route_ctx=entry.route_ctx,
                payload=entry.payload,
            )
            self._states[entry.request_id] = state
            state.mark(source, NodeStatus.WAITING)
            self._ready[source].append(entry.request_id)
