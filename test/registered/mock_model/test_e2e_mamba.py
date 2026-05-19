import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_mamba_extra_buffer_overlap_canary_clean() -> None:
    """Mamba extra_buffer overlap path runs 60s with canary full and reports no violations (#24954 race scenario)."""
    assert False, "phase-2 placeholder"


def test_mamba_state_kv_layout_real_data_all() -> None:
    """Mamba custom state-pool layout under --kv-cache-canary-real-data=all produces no false positives."""
    assert False, "phase-2 placeholder"
