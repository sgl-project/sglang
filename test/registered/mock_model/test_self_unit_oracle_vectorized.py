import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="phase-2; awaits mock_mode/pseudo_mode subsystem reimplementation"
)


def test_predict_input_tokens_for_plan_batch() -> None:
    raise NotImplementedError


def test_predict_input_tokens_for_plan_matches_scalar() -> None:
    raise NotImplementedError


def test_predict_next_tokens_for_active_batch_batch() -> None:
    raise NotImplementedError


def test_predict_next_tokens_matches_scalar() -> None:
    raise NotImplementedError


def test_vectorized_handles_chunked_offsets() -> None:
    raise NotImplementedError
