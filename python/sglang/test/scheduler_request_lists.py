"""Checks on the scheduler's request lists and their requests' KV lengths, and
a partial scheduler with its registry and step planner over what a test gave
it, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List
from unittest.mock import Mock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_registry import (
    RequestRegistry,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


def make_request_retirer(**deps) -> RequestRetirer:
    """A real retirer over a mocked tree cache, with no beam groups, HiSparse
    or outputs; `deps` replaces any constructor argument."""
    args = dict(
        kv=SimpleNamespace(tree_cache=Mock()),
        beam_coordinator=Mock(),
        hisparse_coordinator=None,
        decode_offload_manager=None,
        pending=PendingSteps(),
        output_streamer=Mock(),
        ipc_channels=Mock(),
    )
    args.update(deps)
    return RequestRetirer(**args)


def make_request_registry(**deps) -> RequestRegistry:
    """A registry over empty core lists, with no mode owners and inert
    collaborators; `deps` replaces any constructor argument."""
    args = dict(
        core_lists=CoreLists(),
        grammar_manager=SimpleNamespace(grammar_queue=[], abort_requests=Mock()),
        request_retirer=Mock(),
        ipc_channels=Mock(),
        kv=SimpleNamespace(tree_cache=Mock()),
        mm_receiver=None,
        pending=PendingSteps(),
        decode_offload_manager=None,
        mode_owners=ModeOwners(),
        deferred_responses=DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=Mock(),
            request_retirer=Mock(),
        ),
    )
    args.update(deps)
    return RequestRegistry(**args)


def partial_scheduler() -> Scheduler:
    """A scheduler built with __new__ over empty parts, mode and hub (and an
    empty KV, workers, streams and capacity bundle in the parts), for a test
    to fill."""
    from sglang.srt.managers.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.parts = SimpleNamespace(
        kv=SimpleNamespace(),
        workers=SimpleNamespace(),
        streams=SimpleNamespace(),
        capacity=SimpleNamespace(),
    )
    scheduler.mode = SimpleNamespace()
    scheduler.hub = SimpleNamespace()
    return scheduler


# The registry's constructor arguments a partial scheduler's parts may carry.
_SCHEDULER_FIELDS = {
    "core_lists": "lists",
    "grammar_manager": "grammar_manager",
    "request_retirer": "request_retirer",
    "ipc_channels": "ipc_channels",
    "mm_receiver": "mm_receiver",
    "pending": "pending",
    "decode_offload_manager": "decode_offload_manager",
    "mode_owners": "mode_owners",
    "deferred_responses": "deferred_responses",
}


def request_registry_of(scheduler: Scheduler, **deps) -> RequestRegistry:
    """The registry the assembly would build over the parts a test's partial
    scheduler carries now."""
    parts = scheduler.parts
    args = {
        name: getattr(parts, field)
        for name, field in _SCHEDULER_FIELDS.items()
        if hasattr(parts, field)
    }
    if hasattr(parts.kv, "tree_cache"):
        args["kv"] = SimpleNamespace(tree_cache=parts.kv.tree_cache)
    args.update(deps)
    return make_request_registry(**args)


# The step planner's fields a partial scheduler may carry, by where the
# scheduler holds each.
_PLANNER_FIELDS = {
    "core_lists": ("parts", "lists"),
    **{
        name: ("parts", name)
        for name in (
            "pending",
            "flags",
            "counters",
            "new_token_ratio_tracker",
            "beam_coordinator",
            "step_builder",
            "request_retirer",
            "metrics_sink",
            "ipc_channels",
            "decode_offload_manager",
            "mode_owners",
            "spec_algorithm",
        )
    },
    **{
        name: ("parts", "kv", name)
        for name in ("req_to_token_pool", "token_to_kv_pool_allocator", "tree_cache")
    },
    "hisparse_coordinator": ("parts", "mode_owners", "hisparse_coordinator"),
    "request_intake": ("hub", "request_intake"),
}


def step_planner_of(scheduler: Scheduler, **fields) -> StepPlanner:
    """A step planner over the parts a test's partial scheduler carries now
    and its registry; `fields` replaces any planner field."""
    planner = StepPlanner.__new__(StepPlanner)
    for name, path in _PLANNER_FIELDS.items():
        holder = scheduler
        for attr in path[:-1]:
            holder = getattr(holder, attr, None)
        if hasattr(holder, path[-1]):
            setattr(planner, name, getattr(holder, path[-1]))
    planner.request_registry = scheduler.hub.request_registry
    planner.__dict__.update(fields)
    return planner


def request_lists(scheduler: Scheduler) -> Dict[str, List[Req]]:
    """The lists that own the scheduler's live requests."""
    return {
        name: entry.reqs
        for name, entry in scheduler.hub.request_registry.request_lists().items()
    }


def assert_one_list_per_request(scheduler: Scheduler) -> None:
    """At an iteration boundary: each live request is in exactly one list, and
    every request a pending step runs is either finished or in a list."""
    owners: Dict[int, str] = {}
    for name, reqs in request_lists(scheduler).items():
        for req in reqs:
            assert id(req) not in owners, (
                f"request {req.rid} is in both {owners[id(req)]} and {name}"
            )
            owners[id(req)] = name
    for req in scheduler.parts.lists.decoding:
        assert not req.finished(), f"finished request {req.rid} is still decoding"
    for req in request_lists(scheduler).get("releasing", []):
        assert req.finished(), f"releasing request {req.rid} is not finished"
    for record in scheduler.parts.pending:
        for req in record.step.reqs:
            assert req.finished() or id(req) in owners, (
                f"request {req.rid} of a pending step is in no list"
            )


def assert_progress_lengths(scheduler: Scheduler) -> None:
    """At an iteration boundary: every held row's lengths are in order (a
    request's, a session's, or one released and held for a pending step), and a
    live request's KV allocated past its committed length belongs to its pending
    decode steps: one token per step without speculation; with it, at least the
    bonus token per step, and the rest is the verify reservation."""
    reqs: Dict[int, Req] = {}
    for entries in request_lists(scheduler).values():
        for entry in entries:
            req = getattr(entry, "req", entry)
            reqs[id(req)] = req
    num_inflight: Dict[int, int] = {}
    for record in scheduler.parts.pending:
        for req in record.step.decode_reqs():
            num_inflight[id(req)] = num_inflight.get(id(req), 0) + 1
    spec = not scheduler.parts.spec_algorithm.is_none()
    for req in reqs.values():
        kv = req.kv
        if not kv.holds_kv:
            continue
        kv.check_lengths()
        if req.finished():
            # Its commit skips a pending step's token, which the release frees.
            continue
        n = num_inflight.get(id(req), 0)
        committed, allocated = kv.kv_committed_len, kv.kv_allocated_len
        if spec:
            ok = committed + n <= allocated
        else:
            ok = committed + n == allocated
        assert ok, (
            f"request {req.rid}: {committed=}, {allocated=}, {n} pending decode steps"
        )
    for kv in scheduler.parts.kv.tree_cache.session_records().values():
        kv.check_lengths()
    # A record held for a pending step is its row as the request released it:
    # in order, but no live request's, so the pending-step relation skips it.
    for held in scheduler.parts.pending.held():
        held.kv.check_lengths()
