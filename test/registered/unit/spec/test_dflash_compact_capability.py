from copy import deepcopy

import pytest

from sglang.srt.speculative.dflash_utils import (
    resolve_uniform_swa_dflash_compact_capability,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(
    architecture="DFlash2DraftModel",
    *,
    layers=5,
    window=2048,
    layer_types=None,
):
    return {
        "architectures": [architecture],
        "num_hidden_layers": layers,
        "layer_types": (
            ["sliding_attention"] * layers if layer_types is None else layer_types
        ),
        "sliding_window": window,
        "_name_or_path": "/sensitive/checkpoint/path",
    }


@pytest.mark.parametrize(
    ("architecture", "layers", "window", "block"),
    [
        ("DFlashDraftModel", 5, 2048, 16),
        ("DFlash2DraftModel", 5, 2048, 8),
        ("DFlashLagunaForCausalLM", 4, 4096, 16),
        ("MuseGlimmerAssistantModel", 5, 2048, 16),
    ],
)
def test_registered_architecture_fixtures_are_structurally_eligible(
    architecture, layers, window, block
):
    capability = resolve_uniform_swa_dflash_compact_capability(
        _config(architecture, layers=layers, window=window), window, block
    )

    assert capability.eligible
    assert capability.architecture == architecture
    assert capability.num_layers == layers
    assert capability.layer_types == ("sliding_attention",) * layers
    assert capability.checkpoint_window_tokens == window
    assert capability.attention_window_left == window - 1
    assert capability.block_size == block
    assert capability.rejection_reasons == ()


@pytest.mark.parametrize(
    ("mutate", "runtime_window", "block", "message"),
    [
        (lambda cfg: cfg.pop("layer_types"), 2048, 8, "explicitly present"),
        (
            lambda cfg: cfg.update(layer_types=["full_attention"] * 5),
            2048,
            8,
            "layer_types[0]='full_attention'",
        ),
        (
            lambda cfg: cfg.update(
                num_hidden_layers=6,
                layer_types=["sliding_attention"] * 5 + ["full_attention"],
            ),
            2048,
            8,
            "layer_types[5]='full_attention'",
        ),
        (
            lambda cfg: cfg.update(
                layer_types=["sliding_attention"] * 4 + ["mystery_attention"]
            ),
            2048,
            8,
            "layer_types[4]='mystery_attention'",
        ),
        (
            lambda cfg: cfg.update(layer_types=["sliding_attention"] * 4),
            2048,
            8,
            "length must match",
        ),
        (lambda cfg: cfg.pop("sliding_window"), 2048, 8, "sliding_window"),
        (lambda cfg: cfg.update(sliding_window=0), 2048, 8, "sliding_window"),
        (lambda cfg: cfg.update(sliding_window=-1), 2048, 8, "sliding_window"),
        (lambda cfg: None, 4096, 8, "checkpoint=2048, runtime=4096"),
        (lambda cfg: None, 2048, 4096, "block=4096, checkpoint=2048"),
        (
            lambda cfg: cfg.update(num_sink_tokens=4),
            2048,
            8,
            "num_sink_tokens=4",
        ),
    ],
)
def test_non_uniform_or_unbounded_checkpoint_fails_closed(
    mutate, runtime_window, block, message
):
    config = _config()
    mutate(config)
    capability = resolve_uniform_swa_dflash_compact_capability(
        config, runtime_window, block
    )

    assert not capability.eligible
    rendered = "; ".join(capability.rejection_reasons)
    assert message in rendered
    assert "/sensitive/checkpoint/path" not in rendered


@pytest.mark.parametrize("value", [None, True, 2048.0, "2048", 0, -1])
def test_runtime_window_requires_a_positive_integer(value):
    capability = resolve_uniform_swa_dflash_compact_capability(_config(), value, 8)
    assert not capability.eligible
    assert "runtime draft window" in "; ".join(capability.rejection_reasons)


def test_text_config_fields_are_resolved_without_architecture_allowlisting():
    config = {
        "architectures": ["FuturePureSwaDFlashModel"],
        "text_config": deepcopy(_config(architecture="ignored")),
    }
    config["text_config"].pop("architectures")
    capability = resolve_uniform_swa_dflash_compact_capability(config, 2048, 16)

    assert capability.eligible
    assert capability.architecture == "FuturePureSwaDFlashModel"
    assert capability.attention_window_left == 2047
