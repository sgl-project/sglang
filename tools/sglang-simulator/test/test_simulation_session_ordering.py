"""Session lifecycle requests must be ordered on the simulated timeline.

Regression guard: a close released at wall-clock receipt drops the session's KV
references before its turns have run, so every session-referenced eviction
decision downstream is made against the wrong tree.
"""

import os
import pytest
from types import SimpleNamespace

os.environ.setdefault("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")

from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher
from sglang_simulator.simulation.sglang.session_timeline import SessionTimeline
from sglang_simulator.simulation.types import SimulationMode

# ===== SessionTimeline: no SGLang types, no globals, injected predicate =====


def _timeline(finished: set[str]):
    return SessionTimeline(is_request_finished=lambda rid: rid in finished)


def test_close_is_held_while_its_session_has_a_pending_turn():
    finished = set()
    timeline = _timeline(finished)
    timeline.hold_close(session_id="s1", req="close-s1")

    assert timeline.take_settled_closes({"s1"}) == []

    # The queued turn is dispatched and runs to completion; only then may the
    # close follow it.
    timeline.note_dispatched(session_id="s1", rid="r1")
    assert timeline.take_settled_closes(set()) == []
    finished.add("r1")
    assert timeline.take_settled_closes(set()) == ["close-s1"]


def test_close_is_held_when_no_turn_has_been_dispatched_yet():
    """A close must not overtake turns that have not reached the dispatcher.

    `all()` over an empty rid set is vacuously true, so an unguarded settle
    check releases the close of a session whose turns are still in flight --
    the reordering this class exists to prevent. A client that pipelines its
    close alongside the final turn produces exactly this interleaving.
    """
    timeline = _timeline(finished=set())
    timeline.hold_close(session_id="s1", req="close-s1")

    assert timeline.take_settled_closes(set()) == []


def test_drain_releases_closes_whose_turns_never_finished():
    """An aborted turn never reports completion, so its close never settles."""
    timeline = _timeline(finished=set())
    timeline.hold_close(session_id="s1", req="close-s1")
    timeline.note_dispatched(session_id="s1", rid="r1")

    assert timeline.take_settled_closes(set()) == []
    assert timeline.has_pending() is True
    assert timeline.pending_session_ids() == ["s1"]
    assert timeline.drain() == ["close-s1"]
    assert timeline.has_pending() is False


def test_has_pending_is_false_with_nothing_held():
    assert _timeline(finished=set()).has_pending() is False


def test_close_is_held_while_a_dispatched_turn_is_unfinished():
    finished = set()
    timeline = _timeline(finished)
    timeline.hold_close(session_id="s1", req="close-s1")
    timeline.note_dispatched(session_id="s1", rid="r1")

    assert timeline.take_settled_closes(set()) == []
    finished.add("r1")
    assert timeline.take_settled_closes(set()) == ["close-s1"]


# ===== ReqDispatcher integration, against real io_struct types =====


def _generate_req(rid, session_id, created_time_ms, total_request, output_len=2):
    sampling_params = SimpleNamespace(
        custom_params={
            "simulation": {
                "created_time_ms": created_time_ms,
                "total_request": total_request,
            }
        },
        max_new_tokens=output_len,
    )
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=[1, 2, 3],
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=sampling_params,
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=None if session_id is None else SessionParams(id=session_id),
    )


def _finish(rid):
    req_stats = request_stats_manager.get_req_stats(rid)
    req_stats.gen_token_latencies = [0.01] * req_stats.output_length


def _types(reqs):
    return [type(req).__name__ for req in reqs]


@pytest.fixture(autouse=True)
def _isolate_dispatcher_singleton():
    """Drop the process-global `ReqDispatcher` around every test in this file.

    `ReqDispatcher` is a singleton shared with the scheduler hooks, so a
    dispatcher built here would otherwise outlive the test and be inherited by
    whatever runs next in the same interpreter.
    """
    ReqDispatcher._instance = None
    yield
    ReqDispatcher._instance = None


def _fresh_dispatcher():
    request_stats_manager.reset()
    StateManager.reset()
    dispatcher = ReqDispatcher(SimulationMode.OFFLINE)
    dispatcher.reset()
    return dispatcher


def test_close_waits_for_every_turn_to_finish():
    dispatcher = _fresh_dispatcher()

    dispatcher.add(
        [
            _generate_req("r1", "s1", created_time_ms=1000, total_request=2),
            _generate_req("r2", "s1", created_time_ms=2000, total_request=2),
        ]
    )
    dispatcher.add([CloseSessionReqInput(rid="c1", session_id="s1")])

    StateManager.set_global_clock(0.0)
    assert _types(dispatcher.dispatch()) == []

    StateManager.set_global_clock(1.0)
    assert _types(dispatcher.dispatch()) == ["TokenizedGenerateReqInput"]
    _finish("r1")

    # The second turn has not arrived, so the session is not settled yet.
    assert _types(dispatcher.dispatch()) == []

    StateManager.set_global_clock(2.0)
    assert _types(dispatcher.dispatch()) == ["TokenizedGenerateReqInput"]

    # Dispatched but still decoding: releasing here would free KV mid-turn.
    assert _types(dispatcher.dispatch()) == []

    _finish("r2")
    assert _types(dispatcher.dispatch()) == ["CloseSessionReqInput"]


def test_open_releases_immediately_rather_than_waiting_for_its_turns():
    """`open_session` awaits a scheduler response, so holding it would deadlock."""
    dispatcher = _fresh_dispatcher()

    dispatcher.add(
        [OpenSessionReqInput(rid="o1", capacity_of_str_len=0, session_id="s1")]
    )
    dispatcher.add([_generate_req("r1", "s1", created_time_ms=1000, total_request=1)])

    StateManager.set_global_clock(0.0)
    assert _types(dispatcher.dispatch()) == ["OpenSessionReqInput"]


def test_timeline_independent_control_requests_still_release_immediately():
    dispatcher = _fresh_dispatcher()

    dispatcher.add([_generate_req("r1", None, created_time_ms=5000, total_request=1)])
    dispatcher.add([SimpleNamespace(rid="f1")])

    StateManager.set_global_clock(0.0)
    assert _types(dispatcher.dispatch()) == ["SimpleNamespace"]
