from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import (
    _resolve_dflash_draft_attention_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_UNSET = object()


def _resolve(*, layer_types: list[str], is_causal=_UNSET) -> str:
    fields = {"layer_types": layer_types, "num_hidden_layers": len(layer_types)}
    if is_causal is not _UNSET:
        fields["is_causal"] = is_causal
    draft_config = SimpleNamespace(text_config=SimpleNamespace(**fields))
    server_args = SimpleNamespace(
        speculative_draft_attention_backend="trtllm_mha",
        speculative_draft_model_path="draft",
        speculative_draft_model_revision="main",
        trust_remote_code=False,
        json_model_override_args="{}",
    )
    with (
        patch("sglang.srt.utils.is_hip", return_value=False),
        patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=draft_config,
        ),
    ):
        _resolve_dflash_draft_attention_backend(server_args)
    return server_args.speculative_draft_attention_backend


def test_trtllm_rejected_for_a_non_causal_sliding_layer():
    """trtllm's verify kernel is causal in-window, so a non-causal block is served
    by expanding it into single-query rows that share one `cache_seqlens` -- one
    window for the whole block, anchored at its last position. It is the pairing
    that breaks, so a mixed draft is rejected on the strength of its sliding
    layers alone."""
    for layer_types in (
        ["sliding_attention"] * 5,
        ["full_attention", "sliding_attention", "full_attention"],
    ):
        assert _resolve(layer_types=layer_types, is_causal=False) != "trtllm_mha"


def test_trtllm_kept_when_no_layer_is_both_sliding_and_non_causal():
    """A window is only mis-anchored if the layer carrying it is non-causal. A
    draft that leaves `is_causal` unset keeps its sliding layers on the causal
    layer default (models/dflash.py:72-77), and a non-causal draft with no
    sliding layer has no window for the expansion to get wrong."""
    for layer_types, is_causal in (
        (["sliding_attention"] * 5, True),
        (["sliding_attention"] * 5, _UNSET),
        (["full_attention", "sliding_attention", "full_attention"], _UNSET),
        (["full_attention"] * 5, False),
    ):
        assert _resolve(layer_types=layer_types, is_causal=is_causal) == "trtllm_mha"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
