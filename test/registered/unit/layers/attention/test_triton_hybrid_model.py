"""Focused tests for Triton attention on hybrid linear-attention models."""

from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_nemotron_h_mamba2_is_treated_as_hybrid_linear_attention():
    from sglang.srt.layers.attention import triton_backend

    model_config = object()
    with (
        patch.object(triton_backend, "hybrid_gdn_config", return_value=None),
        patch.object(triton_backend, "kimi_linear_config", return_value=None),
        patch.object(triton_backend, "linear_attn_model_spec", return_value=None),
        patch.object(triton_backend, "mamba2_config", return_value=object()),
    ):
        assert triton_backend._is_hybrid_linear_attention_model(model_config)
