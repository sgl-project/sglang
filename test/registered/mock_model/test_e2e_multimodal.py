import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(reason="phase-2; awaits mock_mode subsystem")


def test_multimodal_image_embed_canary_clean() -> None:
    """Mixed text+image prompt runs 60s with canary fully enabled and reports no violations."""
    assert False, "phase-2 placeholder"


def test_multimodal_special_token_position_chain() -> None:
    """Image-embed placeholder tokens advance the canary chain position correctly instead of being treated as regular input_ids."""
    assert False, "phase-2 placeholder"
