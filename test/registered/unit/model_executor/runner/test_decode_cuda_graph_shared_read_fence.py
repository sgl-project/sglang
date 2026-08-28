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
from sglang.srt.utils.cuda_event_ring import ReusableEventRing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DECODE = ForwardMode.DECODE


def _runner(*, has_marker: bool = False, read_done_events=None):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    # The publisher draws read-done events from the ring that sits next to the
    # mailbox it writes, so the fake runner carries both.
    runner.model_runner = SimpleNamespace(
        shared_read_done_event=None,
        shared_read_done_events=read_done_events,
    )
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
    recorded = []
    ring = ReusableEventRing(
        lambda: SimpleNamespace(record=lambda: recorded.append("record")), depth=2
    )
    runner = _runner(has_marker=True, read_done_events=ring)

    runner._publish_read_done(in_graph=True)
    # In-graph: hand over the graph-recorded marker, do not record a new event.
    marker = runner.in_graph_metadata_prep_done
    assert runner.model_runner.shared_read_done_event is marker
    assert recorded == []

    runner._publish_read_done(in_graph=False)
    assert recorded == ["record"]
    assert runner.model_runner.shared_read_done_event is not marker


def test_publish_read_done_reuses_ring_events():
    # Out-of-graph publishes re-record ring slots instead of allocating an
    # event per step; depth 2 means slot 0 comes back on the third publish.
    ring = ReusableEventRing(lambda: SimpleNamespace(record=lambda: None), depth=2)
    runner = _runner(read_done_events=ring)

    published = []
    for _ in range(3):
        runner._publish_read_done(in_graph=False)
        published.append(runner.model_runner.shared_read_done_event)

    assert published[1] is not published[0]
    assert published[2] is published[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
