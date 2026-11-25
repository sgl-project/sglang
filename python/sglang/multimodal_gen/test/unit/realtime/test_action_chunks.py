# SPDX-License-Identifier: Apache-2.0
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.action_client import ActionChunkBuffer
from sglang.multimodal_gen.runtime.realtime.action_chunk import (
    ActionDeadlineExceeded,
    parse_chunk_request,
)
from sglang.multimodal_gen.runtime.realtime.action_graph import (
    ActionNode,
    ActionStageGraph,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSession,
    RealtimeSessionCache,
)


def reply(tag, commands):
    return {
        "realtime": {k: v for k, v in tag.items() if k != "action_prefix"},
        "data": [{"action": {"values": commands}}],
    }


def test_controller_keeps_the_remaining_old_commands(monkeypatch):
    now = 1000
    monkeypatch.setattr(time, "time_ns", lambda: now * 1_000_000)
    buf = ActionChunkBuffer("robot")
    bootstrap = buf.request(1010, 0)
    assert buf.accept(reply(bootstrap, [[1], [2], [3], [4]]))
    assert buf.pop().item() == 1
    pending = buf.request(1010, 2)
    assert pending["action_prefix"] == [[2], [3]]
    assert buf.pop().item() == 2
    now = 1010
    assert not buf.accept(reply(pending, [[2], [3], [8]]))
    assert buf.pop().item() == 3
    older = buf.request(1050, 1)
    newer = buf.request(1050, 1)
    assert not buf.accept(reply(older, [[4], [8]]))
    assert buf.accept(reply(newer, [[-4], [9]]))
    assert [buf.pop().item(), buf.pop().item()] == [4, 9]
    assert buf.pop() is None


def test_request_prefix_contract():
    config = SimpleNamespace(
        action_dim=2, action_horizon=4, supports_action_prefix=True
    )
    tag = {
        "session_id": "left",
        "start_step": 1,
        "deadline_unix_ms": time.time_ns() // 1_000_000 + 60000,
    }
    assert parse_chunk_request({**tag, "action_prefix": [[0, 1]]}, config)[
        "action_prefix"
    ].shape == (1, 2)
    for malformed in ([[1]], [[0, np.nan]], 4):
        with pytest.raises(ValueError):
            parse_chunk_request({**tag, "action_prefix": malformed}, config)
    with pytest.raises(ActionDeadlineExceeded):
        parse_chunk_request({**tag, "deadline_unix_ms": 1}, config)


def req(name, step=0):
    return SimpleNamespace(
        session=None,
        realtime_session_id=name,
        block_idx=0,
        prompt="tidy",
        extra={
            "vla": {"observation": {}},
            "action_session": True,
            "action_realtime": {"start_step": step},
        },
    )


class History(BaseRealtimeState):
    rows = None


def test_interleaved_sessions_and_uncommitted_work():
    cache = RealtimeSessionCache(max_sessions=2)
    a, b = req("a"), req("b")
    for row, item in enumerate((a, b)):
        cache.attach(item)
        item.session.get_or_create_state(History).rows = row
        cache.commit(item)
    failed = req("a", 1)
    cache.attach(failed)
    failed.session.get_or_create_state(History).rows = 100
    retry = req("a", 1)
    cache.attach(retry)
    assert retry.session.get_or_create_state(History).rows == 0
    cache.commit(retry)
    with pytest.raises(ValueError, match="action_chunk_conflict"):
        cache.attach(req("a", 1))
    assert cache.release("a")
    again = req("b", 1)
    cache.attach(again)
    assert again.session.get_or_create_state(History).rows == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Checks resident CUDA storage"
)
def test_shared_storage_is_charged_once_and_evicts_lru():
    cache = RealtimeSessionCache(max_sessions=3, max_bytes=128)
    for key in ("first", "second"):
        item = req(key)
        cache.attach(item)
        whole = torch.zeros(100, device="cuda", dtype=torch.uint8)
        item.session.get_or_create_state(History).rows = (whole[:2], whole[2:])
        assert item.session.resident_bytes() == 100
        cache.commit(item)
    assert not cache.release("first")
    assert cache.release("second")


def test_slow_stream_serves_previous_value_and_runs_on_its_clock():
    updated = Event()
    allow_refresh = Event()

    def producer(state, emit):
        state["turn"] = state.get("turn", 0) + 1
        if state["turn"] == 2:
            assert allow_refresh.wait(3)
        emit(state["turn"])
        if state["turn"] == 2:
            updated.set()

    graph = ActionStageGraph(
        [ActionNode("subgoal", ("state", "emit"), producer, 0.1)],
        {"subgoal": "subgoal"},
    )
    batch = req("robot")
    batch.session = RealtimeSession()
    try:
        graph.forward(batch, None)
        assert batch.extra["vla"]["observation"]["subgoal"] == 1
        graph.forward(batch, None)
        assert batch.extra["vla"]["observation"]["subgoal"] == 1
        batch.session.committed()
        allow_refresh.set()
        assert updated.wait(3)
        graph.forward(batch, None)
        assert batch.extra["vla"]["observation"]["subgoal"] == 2
    finally:
        allow_refresh.set()
        batch.session.dispose()
        graph.pools["subgoal"].shutdown(wait=True)


def test_periodic_producer_can_tick_without_another_policy_request():
    emitted_twice = Event()

    def produce(state):
        state["calls"] = state.get("calls", 0) + 1
        if state["calls"] == 2:
            emitted_twice.set()
        return state["calls"]

    graph = ActionStageGraph(
        [ActionNode("goal", ("state",), produce, period_s=0.05, repeat_latest=True)],
        {"subgoal": "goal"},
    )
    batch = req("background")
    batch.session = RealtimeSession()
    try:
        graph.forward(batch, None)
        assert emitted_twice.wait(3)
    finally:
        batch.session.dispose()
        graph.pools["goal"].shutdown(wait=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Checks resident CUDA storage"
)
def test_background_output_uses_session_reservation():
    batch = req("planner")
    batch.session = RealtimeSession()
    graph = ActionStageGraph(
        [
            ActionNode(
                "goal",
                (),
                lambda: torch.zeros(8, device="cuda"),
                period_s=1,
                retained_gpu_bytes=32,
            )
        ],
        {"goal": "goal"},
    )
    try:
        graph.forward(batch, None)
        assert batch.session.resident_bytes() == 32
    finally:
        batch.session.dispose()
        graph.pools["goal"].shutdown(wait=True)

    batch.session = RealtimeSession()
    graph = ActionStageGraph(
        [
            ActionNode(
                "large",
                (),
                lambda: torch.zeros(9, device="cuda"),
                period_s=1,
                retained_gpu_bytes=32,
            )
        ],
        {"goal": "large"},
    )
    try:
        with pytest.raises(ValueError, match="reservation"):
            graph.forward(batch, None)
    finally:
        batch.session.dispose()
        graph.pools["large"].shutdown(wait=True)
