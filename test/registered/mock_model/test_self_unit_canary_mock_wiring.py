import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="phase-2; awaits mock_mode/pseudo_mode subsystem reimplementation"
)


def test_mock_expected_tokens_filled_from_oracle() -> None:
    raise NotImplementedError


def test_mock_expected_positions_filled_from_oracle() -> None:
    raise NotImplementedError


def test_mock_mode_on_enables_write_step_comparison() -> None:
    raise NotImplementedError


def test_mock_mode_off_dummy_tensors_safe() -> None:
    raise NotImplementedError


def test_mock_input_perturb_env_detects_violation() -> None:
    raise NotImplementedError


def test_violation_row_carries_req_id_and_position() -> None:
    raise NotImplementedError


def test_violation_does_not_cascade_to_next_verify() -> None:
    raise NotImplementedError
