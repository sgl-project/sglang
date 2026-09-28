"""DFlash attention configuration compatibility tests."""

import pytest

from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_causal_override,
    get_dflash_attention_sliding_window_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_absent_causal_override_preserves_legacy_behavior():
    assert get_dflash_attention_causal_override({"dflash_config": {}}) is None


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        (False, False),
        (True, True),
        (0, False),
        (1, True),
        ("off", False),
        ("yes", True),
    ),
)
def test_explicit_causal_override_is_parsed(configured, expected):
    config = {"dflash_config": {"causal": configured}}
    assert get_dflash_attention_causal_override(config) is expected


def test_invalid_causal_override_is_rejected():
    with pytest.raises(ValueError, match="dflash_config.causal"):
        get_dflash_attention_causal_override({"dflash_config": {"causal": "sometimes"}})


def test_sliding_window_falls_back_to_dflash_swa_window_size():
    config = {
        "layer_types": ["sliding_attention"],
        "dflash_config": {"swa_window_size": 1024},
    }
    assert get_dflash_attention_sliding_window_size(config) == 1023


def test_top_level_sliding_window_keeps_precedence():
    config = {
        "layer_types": ["sliding_attention"],
        "sliding_window": 2048,
        "dflash_config": {"swa_window_size": 1024},
    }
    assert get_dflash_attention_sliding_window_size(config) == 2047
