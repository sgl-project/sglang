"""Checks on the scheduler's request lists, and its registry and step planner
over a partial scheduler, for tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List
from unittest.mock import Mock

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_registry import (
    RequestRegistry,
)
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler


def make_request_registry(**deps) -> RequestRegistry:
    """A registry over empty core lists, with no mode owners and inert
    collaborators; `deps` replaces any constructor argument."""
    args = dict(
        core_lists=CoreLists(),
        grammar_manager=SimpleNamespace(grammar_queue=[], abort_requests=Mock()),
        request_retirer=Mock(),
        beam_coordinator=Mock(),
        ipc_channels=Mock(),
        kv=SimpleNamespace(tree_cache=Mock()),
        mm_receiver=None,
        pending=PendingSteps(),
        decode_offload_manager=None,
        mode_owners=ModeOwners(),
    )
    args.update(deps)
    return RequestRegistry(**args)


# The registry's constructor arguments a partial scheduler may carry.
_SCHEDULER_FIELDS = {
    "core_lists": "lists",
    "grammar_manager": "grammar_manager",
    "request_retirer": "request_retirer",
    "beam_coordinator": "beam_coordinator",
    "ipc_channels": "ipc_channels",
    "mm_receiver": "mm_receiver",
    "pending": "pending",
    "decode_offload_manager": "decode_offload_manager",
    "mode_owners": "mode_owners",
}


def request_registry_of(scheduler: Scheduler, **deps) -> RequestRegistry:
    """The registry `init_request_registry` would build over the fields a
    test's partial scheduler carries now."""
    args = {
        name: getattr(scheduler, field)
        for name, field in _SCHEDULER_FIELDS.items()
        if hasattr(scheduler, field)
    }
    if hasattr(scheduler, "tree_cache"):
        args["kv"] = SimpleNamespace(tree_cache=scheduler.tree_cache)
    args.update(deps)
    return make_request_registry(**args)


# The step planner's fields a partial scheduler may carry, by the scheduler's
# name for each.
_PLANNER_FIELDS = {
    "core_lists": "lists",
    **{
        name: name
        for name in (
            "pending",
            "flags",
            "counters",
            "req_to_token_pool",
            "token_to_kv_pool_allocator",
            "tree_cache",
            "new_token_ratio_tracker",
            "beam_coordinator",
            "step_builder",
            "hisparse_coordinator",
            "metrics_sink",
            "ipc_channels",
            "decode_offload_manager",
            "mode_owners",
            "request_intake",
            "spec_algorithm",
        )
    },
}


def step_planner_of(scheduler: Scheduler, **fields) -> StepPlanner:
    """A step planner over the fields a test's partial scheduler carries now
    and its registry; `fields` replaces any planner field."""
    planner = StepPlanner.__new__(StepPlanner)
    for name, field in _PLANNER_FIELDS.items():
        if hasattr(scheduler, field):
            setattr(planner, name, getattr(scheduler, field))
    planner.request_registry = scheduler.request_registry
    planner.__dict__.update(fields)
    return planner


def request_lists(scheduler: Scheduler) -> Dict[str, List[Req]]:
    """The lists that own the scheduler's live requests."""
    return {
        name: entry.reqs
        for name, entry in scheduler.request_registry.request_lists().items()
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
    for req in scheduler.lists.decoding:
        assert not req.finished(), f"finished request {req.rid} is still decoding"
    for req in request_lists(scheduler).get("releasing", []):
        assert req.finished(), f"releasing request {req.rid} is not finished"
    for record in scheduler.pending:
        for req in record.step.reqs:
            assert req.finished() or id(req) in owners, (
                f"request {req.rid} of a pending step is in no list"
            )
