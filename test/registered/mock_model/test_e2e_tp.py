import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_tp2_cross_rank_violation_lockstep_raise() -> None:
    """TP=2 real launch: injecting a perturb on rank0 makes both ranks raise in the same forward step."""
    assert False, "phase-2 placeholder"


def test_tp4_canary_overhead_within_budget() -> None:
    """TP=4 with canary fully enabled keeps end-to-end overhead inside the §3.3 self-bench budget over 60s."""
    assert False, "phase-2 placeholder"
