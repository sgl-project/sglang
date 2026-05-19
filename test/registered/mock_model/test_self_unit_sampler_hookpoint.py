import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="phase-2; awaits mock_mode/pseudo_mode subsystem reimplementation"
)


def test_sampler_override_forces_oracle_token() -> None:
    raise NotImplementedError


def test_sampler_path_still_executes() -> None:
    raise NotImplementedError


def test_top_p_intermediate_results_not_asserted() -> None:
    raise NotImplementedError


def test_eos_token_propagates_to_req() -> None:
    raise NotImplementedError
