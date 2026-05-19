import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="phase-2; awaits mock_mode/pseudo_mode subsystem reimplementation"
)


def test_launch_qwen3_dummy_weights_one_layer() -> None:
    raise NotImplementedError


def test_admit_returns_req_handle() -> None:
    raise NotImplementedError


def test_step_drives_one_event_loop() -> None:
    raise NotImplementedError


def test_step_until_target_steps() -> None:
    raise NotImplementedError


def test_force_preempt_pauses_req() -> None:
    raise NotImplementedError


def test_resume_returns_req_to_active() -> None:
    raise NotImplementedError


def test_abort_removes_req() -> None:
    raise NotImplementedError


def test_canary_violations_returns_list() -> None:
    raise NotImplementedError


def test_assert_no_canary_violations_raises_on_violation() -> None:
    raise NotImplementedError


def test_engine_idle_60s_no_violation() -> None:
    raise NotImplementedError


def test_forward_uses_triton_hash_kernel() -> None:
    raise NotImplementedError


def test_oracle_expectation_per_req_per_step_can_be_pinned() -> None:
    raise NotImplementedError
