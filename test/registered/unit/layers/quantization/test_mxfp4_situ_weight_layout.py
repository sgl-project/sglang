import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


@pytest.mark.parametrize(
    ("a8w4", "a4w4", "expected"),
    [
        (None, None, True),
        ("0", "1", False),
        ("1", "0", True),
        ("1", "1", True),
    ],
)
def test_aiter_situ_weight_layout_matches_activation_mode_precedence(
    monkeypatch, a8w4, a4w4, expected
):
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )

    monkeypatch.delenv("AITER_SITUV2_A8W4", raising=False)
    monkeypatch.delenv("AITER_SITUV2_A4W4", raising=False)
    if a8w4 is not None:
        monkeypatch.setenv("AITER_SITUV2_A8W4", a8w4)
    if a4w4 is not None:
        monkeypatch.setenv("AITER_SITUV2_A4W4", a4w4)

    assert _aiter_situ_uses_gu_interleaved_weights() is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
