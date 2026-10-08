import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sglang.srt.arg_groups import parallel_hook
from sglang.srt.fault_tolerance.ft_state import FaultToleranceState
from sglang.srt.fault_tolerance.manager import FaultToleranceManager
from sglang.srt.fault_tolerance.protocol import parse_apply_request
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("attn_dp_size", [1, 2])
def test_fault_tolerance_validates_attention_dp(monkeypatch, attn_dp_size):
    cfg = SimpleNamespace(
        enable_fault_tolerance=True,
        dp_size=1,
        attn_dp_size=attn_dp_size,
        enable_dp_attention=False,
        disaggregation_mode="null",
        fault_tolerance_on_error_strategy="pause",
        fault_tolerance_timeout=60,
        fault_tolerance_pause_timeout=300,
    )
    monkeypatch.setattr(parallel_hook, "resolving_view", lambda _: cfg)
    if attn_dp_size == 1:
        with pytest.raises(AssertionError, match="--attn-dp-size"):
            parallel_hook.handle_fault_tolerance(cfg)
    else:
        parallel_hook.handle_fault_tolerance(cfg)


def test_protocol_and_state_contract():
    request = parse_apply_request(
        b'{"instruction":"scale_down","params":{"removed_dp_ranks":[1]}}'
    )
    assert request.params.removed_dp_ranks == [1]
    with pytest.raises(ValueError, match="Invalid instruction"):
        parse_apply_request(b'{"instruction":"recover"}')
    state = FaultToleranceState(dp_size=2, strategy="pause", global_rank_count=4)
    state.observe_process_active_ranks([2], active=False)
    assert state.process_alive_dp_mask() == [True, False]
    assert state.status_response()["engines"][1]["status"] == "dead"
    assert state.expand_dp_mask_to_global_rank_mask([True, False]) == [
        True,
        True,
        False,
        False,
    ]
    manager = FaultToleranceManager(
        server_args=SimpleNamespace(
            dp_size=1,
            attn_dp_size=2,
            num_dp_ranks=2,
            tp_size=2,
            fault_tolerance_on_error_strategy="pause",
        ),
        zmq_context=Mock(),
        send_to_scheduler=AsyncMock(),
    )
    assert manager.state.dp_size == 2
    assert manager._route_dp_mask == [True, True]
    manager._finish_submitted_apply("request-1", None)
    status = manager.status()[1]
    assert status["last_ft_request_id"] == "request-1"
    assert "ft_error" not in status
    assert "last_ft_request_id" not in status["engines"][0]
    manager._finish_submitted_apply("request-2", "failed")
    assert manager.status()[1]["ft_error"] == "failed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
