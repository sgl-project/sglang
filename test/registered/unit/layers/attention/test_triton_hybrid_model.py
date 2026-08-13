"""Focused tests for Triton attention on hybrid linear-attention models."""

from types import SimpleNamespace

from sglang.srt.configs import NemotronHConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_nemotron_h_mamba2_is_treated_as_hybrid_linear_attention():
    from sglang.srt.layers.attention import triton_backend

    model_config = SimpleNamespace(
        hf_config=NemotronHConfig(),
        is_draft_model=False,
        linear_attn_registry_result=None,
    )

    assert triton_backend.mamba2_config(model_config) is model_config.hf_config
    assert triton_backend._is_hybrid_linear_attention_model(model_config)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
