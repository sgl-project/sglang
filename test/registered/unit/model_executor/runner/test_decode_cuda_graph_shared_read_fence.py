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

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

DECODE = ForwardMode.DECODE
VERIFY = ForwardMode.TARGET_VERIFY
EXTEND = ForwardMode.EXTEND


def _runner(*, owns_verify: bool = False, has_marker: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(
            is_last_shared_read_phase=lambda fm: owns_verify and fm.is_target_verify()
        ),
        shared_read_done_event=None,
    )
    runner.in_graph_metadata_prep_done = object() if has_marker else None
    return runner


def _backend(declared: SharedReadEnds):
    # Spec'd against the real ABC so a rename fails here, not at runtime.
    backend = create_autospec(AttentionBackend, instance=True)
    backend.shared_read_ends.return_value = declared
    return backend


@pytest.mark.parametrize(
    "mode, owns_verify, declared, has_marker, expected",
    [
        # Only decode / target verify publish; anything else keeps the coarse fence.
        (EXTEND, False, SharedReadEnds.IN_REPLAY, True, SharedReadEnds.UNKNOWN),
        # Target verify publishes only when it is the step's last reading phase.
        (VERIFY, False, SharedReadEnds.IN_REPLAY, True, SharedReadEnds.UNKNOWN),
        (VERIFY, True, SharedReadEnds.IN_REPLAY, True, SharedReadEnds.IN_REPLAY),
        # A backend that keeps reading through the graph is never advanced.
        (VERIFY, True, SharedReadEnds.POST_REPLAY, True, SharedReadEnds.POST_REPLAY),
        # Nothing to demote: the declaration is honored as-is.
        (DECODE, False, SharedReadEnds.IN_REPLAY, True, SharedReadEnds.IN_REPLAY),
        # Nowhere to record in-graph -> fall back to the pre-replay record.
        (DECODE, False, SharedReadEnds.IN_REPLAY, False, SharedReadEnds.PRE_REPLAY),
    ],
)
def test_resolve_shared_read_ends(mode, owns_verify, declared, has_marker, expected):
    runner = _runner(owns_verify=owns_verify, has_marker=has_marker)
    assert runner._resolve_shared_read_ends(_backend(declared), mode) is expected


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
