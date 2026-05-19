import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_pp2_canary_clean() -> None:
    """PP=2 with canary fully enabled runs 60s and reports no violations."""
    assert False, "phase-2 placeholder"


def test_pp2_layer_split_real_kv_source_layout() -> None:
    """PP=2 layer-split pool exercises RealKvSource access invariants on the correct bytes."""
    assert False, "phase-2 placeholder"
