import os

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_eagle_v2_draft_verify_chain_canary_clean() -> None:
    """EAGLE spec v2 draft+verify chain runs 60s under canary full + real_data=all with no violations."""
    assert False, "phase-2 placeholder"


def test_eagle_v2_with_chunked_prefill() -> None:
    """EAGLE spec v2 combined with chunked prefill runs canary-clean."""
    assert False, "phase-2 placeholder"


def test_eagle_v2_with_radix_cache_hit() -> None:
    """EAGLE spec v2 with radix prefix-cache hit runs canary-clean."""
    assert False, "phase-2 placeholder"


def test_v2_env_var_explicitly_set() -> None:
    """Fixture-side check that the test harness explicitly set SGLANG_ENABLE_SPEC_V2=1 (guards against default-value drift)."""
    assert (
        os.environ.get("SGLANG_ENABLE_SPEC_V2") == "1"
    ), "fixture must explicitly set SGLANG_ENABLE_SPEC_V2=1 before launching the server"
