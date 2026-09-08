from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from sglang.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    SharedReadEnds,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DECODE = ForwardMode.DECODE


def _runner(*, has_marker: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(shared_read_done_event=None)
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
