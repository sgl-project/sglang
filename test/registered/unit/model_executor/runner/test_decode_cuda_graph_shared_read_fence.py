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

    def is_last_shared_read_phase(self, forward_mode) -> bool:
        return self._target_verify_war and forward_mode.is_target_verify()


def _attn_backend(boundary=SharedReadBoundary.IN_REPLAY):
    """Backend stub declaring one fixed read-end boundary for every mode."""
    return SimpleNamespace(shared_read_boundary=lambda _forward_mode: boundary)


def _runner(*, target_verify_war: bool = False, has_marker: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(target_verify_war),
        device_timer=None,
        is_draft_worker=False,
        shared_read_done_event=None,
    )
    runner.in_graph_metadata_prep_done = object() if has_marker else None
    return runner


def test_unrelated_modes_never_publish():
    # This runner owns the fence for decode / target verify only; every other
    # mode stays on the coarse wait, even with a marker available.
    assert (
        _runner(has_marker=True)._resolve_shared_read_boundary(
            _attn_backend(), ForwardMode.EXTEND
        )
        is SharedReadBoundary.UNKNOWN
    )


def test_post_replay_declaration_is_not_advanced():
    # A backend that keeps reading shared state across the whole graph declares
    # POST_REPLAY. Having an in-graph marker must not pull the fence earlier.
    assert (
        _runner(target_verify_war=True, has_marker=True)._resolve_shared_read_boundary(
            _attn_backend(SharedReadBoundary.POST_REPLAY), ForwardMode.TARGET_VERIFY
        )
        is SharedReadBoundary.POST_REPLAY
    )


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


def test_execute_publishes_the_in_graph_marker():
    runner = _runner(has_marker=True)
    marker = runner.in_graph_metadata_prep_done
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
    assert runner.model_runner.shared_read_done_event is marker


def test_execute_falls_back_to_pre_replay_without_marker():
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
    assert isinstance(runner.model_runner.shared_read_done_event, Event)


@pytest.mark.parametrize("supported", [False, True])
def test_target_verify_requires_war_capability(supported):
    runner = _runner(target_verify_war=supported, has_marker=True)
    marker = runner.in_graph_metadata_prep_done
    runner.attn_backend = _attn_backend()
    runner.device_module = SimpleNamespace(Event=lambda: None)

    runner.execute(_execute_harness(runner, [], ForwardMode.TARGET_VERIFY))

    expected = marker if supported else None
    assert runner.model_runner.shared_read_done_event is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
