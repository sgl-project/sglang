import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import SharedReadBoundary
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SpecAlgorithm:
    def __init__(self, target_verify_war: bool = False):
        self._target_verify_war = target_verify_war

    def is_war_publish_phase(self, forward_mode) -> bool:
        return self._target_verify_war and forward_mode.is_target_verify()


def _attn_backend(*, breakable_metadata=False):
    def shared_read_boundary(forward_mode):
        if breakable_metadata and forward_mode.is_target_verify():
            return SharedReadBoundary.POST_REPLAY
        if forward_mode.is_decode() or forward_mode.is_target_verify():
            return SharedReadBoundary.IN_REPLAY
        return SharedReadBoundary.UNKNOWN

    return SimpleNamespace(shared_read_boundary=shared_read_boundary)


def _runner(*, target_verify_war: bool = False, planted: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(target_verify_war),
        device_timer=None,
        is_draft_worker=False,
        war_read_done_event=None,
        war_fastpath_read_done_event=None,
    )
    runner._war_read_done_node_planted = planted
    return runner


def test_war_read_done_record():
    # Planted node: the graph re-arms it every replay.
    assert (
        _runner(planted=True)._war_read_done_record(_attn_backend(), ForwardMode.DECODE)
        is SharedReadBoundary.IN_REPLAY
    )
    # No planted node: fall back to a pre-replay record.
    assert (
        _runner()._war_read_done_record(_attn_backend(), ForwardMode.DECODE)
        is SharedReadBoundary.PRE_REPLAY
    )
    # Unrelated modes never publish from the decode graph runner.
    assert (
        _runner(planted=True)._war_read_done_record(_attn_backend(), ForwardMode.EXTEND)
        is SharedReadBoundary.UNKNOWN
    )
    # The algorithm gate precedes the backend declaration.
    assert (
        _runner(planted=True)._war_read_done_record(
            _attn_backend(breakable_metadata=True), ForwardMode.TARGET_VERIFY
        )
        is SharedReadBoundary.UNKNOWN
    )
    # Captured-metadata verify keeps reading throughout the graph, even planted.
    assert (
        _runner(target_verify_war=True, planted=True)._war_read_done_record(
            _attn_backend(breakable_metadata=True), ForwardMode.TARGET_VERIFY
        )
        is SharedReadBoundary.POST_REPLAY
    )


def test_publish_war_read_done():
    runner = _runner()
    graph_event = object()
    runner.model_runner.war_read_done_event = graph_event
    runner._publish_war_read_done(in_graph=True)
    assert runner.model_runner.war_fastpath_read_done_event is graph_event

    recorded = []

    class Event:
        def record(self):
            recorded.append(self)

    runner.device_module = SimpleNamespace(Event=Event)
    runner._publish_war_read_done(in_graph=False)
    published = runner.model_runner.war_fastpath_read_done_event
    assert isinstance(published, Event) and recorded == [published]


def _execute_harness(runner, calls, mode=ForwardMode.DECODE):
    key = ShapeKey(size=1)
    output = PPProxyTensors({"hidden_states": torch.ones(1, 1)})
    runner.ragged_verify_mode = False
    runner.bs = 1
    runner.load_batch = lambda *_: setattr(runner, "_replay_graph_key", key)

    class Backend:
        def replay_session(self):
            return contextlib.nullcontext()

        def replay(self, replay_key, _forward_batch):
            assert replay_key == key
            calls.append("replay")
            return output

    runner.backend = Backend()
    return SimpleNamespace(forward_mode=mode, batch_size=1)


def test_execute_publishes_the_planted_graph_event():
    runner = _runner(planted=True)
    graph_event = object()
    runner.model_runner.war_read_done_event = graph_event
    runner.attn_backend = _attn_backend()
    runner.device_module = SimpleNamespace(
        Event=lambda: (_ for _ in ()).throw(
            AssertionError("execute must reuse the graph-recorded event")
        )
    )
    calls = []
    forward_batch = _execute_harness(runner, calls)

    result = runner.execute(forward_batch)

    assert result.tensors["hidden_states"].shape == (1, 1)
    assert runner.model_runner.war_fastpath_read_done_event is graph_event


def test_execute_records_pre_replay_for_snapshot_backends():
    runner = _runner()
    runner.attn_backend = _attn_backend()
    calls = []

    class Event:
        def record(self):
            calls.append("record")

    runner.device_module = SimpleNamespace(Event=Event)
    forward_batch = _execute_harness(runner, calls)

    runner.execute(forward_batch)

    # The eager record lands before the replay so the fence stays truthful.
    assert calls == ["record", "replay"]
    assert isinstance(runner.model_runner.war_fastpath_read_done_event, Event)


@pytest.mark.parametrize("supported", [False, True])
def test_target_verify_requires_war_capability(supported):
    runner = _runner(target_verify_war=supported, planted=True)
    graph_event = object()
    runner.model_runner.war_read_done_event = graph_event
    runner.attn_backend = _attn_backend()
    runner.device_module = SimpleNamespace(Event=lambda: None)

    runner.execute(_execute_harness(runner, [], ForwardMode.TARGET_VERIFY))

    expected = graph_event if supported else None
    assert runner.model_runner.war_fastpath_read_done_event is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
