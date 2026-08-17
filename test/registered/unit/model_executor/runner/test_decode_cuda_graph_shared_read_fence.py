import contextlib
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.layers.attention.linear.kda_route_telemetry import (
    KDACudaGraphRoutePlans,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DECODE = ForwardMode.DECODE


def _runner(*, has_marker: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        device_timer=None,
        is_draft_worker=False,
        shared_read_done_event=None,
    )
    runner.enable_pdmux = False
    runner.in_graph_metadata_prep_done = object() if has_marker else None
    return runner


def _backend(declared: SharedReadEnds):
    # Spec'd against the real ABC so a rename fails here, not at runtime.
    backend = create_autospec(AttentionBackend, instance=True)
    backend.shared_read_ends.return_value = declared
    return backend


@pytest.mark.parametrize(
    "declared, has_marker, expected",
    [
        # The backend's declaration decides where the record lands.
        (SharedReadEnds.IN_REPLAY, True, SharedReadEnds.IN_REPLAY),
        # Nowhere to record in-graph -> fall back to the pre-replay record.
        (SharedReadEnds.IN_REPLAY, False, SharedReadEnds.PRE_REPLAY),
        # Only an in-graph declaration is demoted; the rest pass through.
        (SharedReadEnds.POST_REPLAY, False, SharedReadEnds.POST_REPLAY),
    ],
)
def test_resolve_shared_read_ends(declared, has_marker, expected):
    runner = _runner(has_marker=has_marker)
    backend = _backend(declared)

    assert runner._resolve_shared_read_ends(backend, DECODE) is expected
    backend.shared_read_ends.assert_called_once_with(DECODE)


def _execute_harness(runner, calls):
    key = ShapeKey(size=1)
    runner.kda_cuda_graph_route_plans = KDACudaGraphRoutePlans()
    runner.kda_cuda_graph_route_plans.bind("decode", key, ())
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
    return SimpleNamespace(forward_mode=DECODE, batch_size=1)


def test_execute_publishes_the_in_graph_marker_with_kda_route_replay():
    runner = _runner(has_marker=True)
    marker = runner.in_graph_metadata_prep_done
    runner.attn_backend = _backend(SharedReadEnds.IN_REPLAY)
    runner.device_module = SimpleNamespace(
        Event=lambda: (_ for _ in ()).throw(
            AssertionError("execute must reuse the graph-recorded event")
        )
    )
    calls = []

    result = runner.execute(_execute_harness(runner, calls))

    assert result.tensors["hidden_states"].shape == (1, 1)
    assert calls == ["replay"]
    assert runner.model_runner.shared_read_done_event is marker


def test_execute_falls_back_before_kda_route_replay_without_marker():
    runner = _runner()
    runner.attn_backend = _backend(SharedReadEnds.IN_REPLAY)
    calls = []

    class Event:
        def record(self):
            calls.append("record")

    runner.device_module = SimpleNamespace(Event=Event)

    runner.execute(_execute_harness(runner, calls))

    assert calls == ["record", "replay"]
    assert isinstance(runner.model_runner.shared_read_done_event, Event)


def test_publish_read_done():
    runner = _runner(has_marker=True)
    recorded = []
    runner.device_module = SimpleNamespace(
        Event=lambda: SimpleNamespace(record=lambda: recorded.append("record"))
    )

    runner._publish_read_done(in_graph=True)
    # In-graph: hand over the graph-recorded marker, do not record a new event.
    marker = runner.in_graph_metadata_prep_done
    assert runner.model_runner.shared_read_done_event is marker
    assert recorded == []

    runner._publish_read_done(in_graph=False)
    assert recorded == ["record"]
    assert runner.model_runner.shared_read_done_event is not marker


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
