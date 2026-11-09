import itertools
import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")

# "true" is off for AITER, which only honours "1"; SGLang must read it the same way.
_ENV_VALUES = (None, "0", "1", "true")


def _set_situ_env(monkeypatch, a8w4, a4w4):
    monkeypatch.delenv("AITER_SITUV2_A8W4", raising=False)
    monkeypatch.delenv("AITER_SITUV2_A4W4", raising=False)
    if a8w4 is not None:
        monkeypatch.setenv("AITER_SITUV2_A8W4", a8w4)
    if a4w4 is not None:
        monkeypatch.setenv("AITER_SITUV2_A4W4", a4w4)


# AITER's A16W4 SiTUv2 kernel reads the separated w13 layout, so only A8W4
# (gfx95) may get the gate/up-interleaved one. Giving A16W4 that layout made
# the MoE output nearly orthogonal to the reference on gfx942.
@pytest.mark.parametrize(
    ("gfx95", "a8w4", "a4w4", "expected"),
    [
        (False, None, None, False),
        (False, "0", "1", False),
        (False, "1", "0", False),
        (True, None, None, False),
        (True, "0", "1", False),
        (True, "1", "0", True),
        (True, "1", "1", True),
        (True, "true", None, False),
    ],
)
def test_aiter_situ_weight_layout_is_gu_interleaved_only_for_a8w4(
    monkeypatch, gfx95, a8w4, a4w4, expected
):
    from sglang.srt.layers.quantization import mxfp4

    monkeypatch.setattr(mxfp4, "is_gfx95_supported", lambda: gfx95)
    _set_situ_env(monkeypatch, a8w4, a4w4)

    assert mxfp4._aiter_situ_uses_gu_interleaved_weights() is expected


@pytest.mark.parametrize(
    ("gfx95", "a8w4", "a4w4", "expected"),
    [
        (True, None, None, False),
        (True, "0", "1", False),
        (True, "1", "0", True),
        (True, "1", "1", True),
        (False, "1", "0", False),
        (True, "true", None, False),
    ],
)
def test_aiter_situ_a8w4_route_matches_activation_mode_precedence(
    monkeypatch, gfx95, a8w4, a4w4, expected
):
    from sglang.srt.layers.quantization import mxfp4

    monkeypatch.setattr(mxfp4, "is_gfx95_supported", lambda: gfx95)
    _set_situ_env(monkeypatch, a8w4, a4w4)

    assert mxfp4._aiter_situ_uses_a8w4() is expected


@pytest.mark.parametrize(
    ("a8w4", "a4w4", "expected"),
    [
        (None, None, False),
        ("0", "1", True),
        ("1", "0", False),
        ("1", "1", False),
        (None, "true", False),
        ("true", "1", True),
    ],
)
def test_aiter_situ_a4w4_route_yields_to_a8w4(monkeypatch, a8w4, a4w4, expected):
    from sglang.srt.layers.quantization import mxfp4

    _set_situ_env(monkeypatch, a8w4, a4w4)

    assert mxfp4._aiter_situ_uses_a4w4() is expected


@pytest.mark.parametrize(
    ("gfx95", "a8w4", "a4w4"),
    list(itertools.product((False, True), _ENV_VALUES, _ENV_VALUES)),
)
def test_aiter_situ_weight_layout_follows_a8w4_route(monkeypatch, gfx95, a8w4, a4w4):
    """The layout selector and the A8W4 route must not drift apart again."""
    from sglang.srt.layers.quantization import mxfp4

    monkeypatch.setattr(mxfp4, "is_gfx95_supported", lambda: gfx95)
    _set_situ_env(monkeypatch, a8w4, a4w4)

    assert (
        mxfp4._aiter_situ_uses_gu_interleaved_weights() is mxfp4._aiter_situ_uses_a8w4()
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
