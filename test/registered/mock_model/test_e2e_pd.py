import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_pd_transfer_canary_clean() -> None:
    """Standard PD prefill-decode split with canary fully enabled runs 60s and reports no violations."""
    assert False, "phase-2 placeholder"


def test_pd_transfer_checksum_full_real_data() -> None:
    """PD path under --kv-cache-canary-real-data=all with full-sweep cadence detects any misplaced byte on the transfer link."""
    assert False, "phase-2 placeholder"


def test_pd_transfer_corrupted_byte_detected() -> None:
    """Injecting a single-byte corruption on the PD transfer makes the next sweep raise a REAL_KV violation."""
    assert False, "phase-2 placeholder"
