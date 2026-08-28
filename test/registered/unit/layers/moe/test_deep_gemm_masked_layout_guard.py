"""Guard: clamped/swizzled MoE activations must not select the masked layout
when the DSV4 JIT masked kernel cannot serve their shape or quant group."""

import sys

import pytest
import torch

from sglang.srt.layers.moe.moe_runner import deep_gemm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    _should_use_masked_standard_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


@pytest.fixture(autouse=True)
def _layout_state(monkeypatch):
    monkeypatch.setenv("SGLANG_DEEPGEMM_STANDARD_LAYOUT", "auto")
    monkeypatch.setattr(
        deep_gemm, "_masked_standard_layout_memory_budget_bytes", 1 << 40
    )
    monkeypatch.setattr(deep_gemm, "_masked_activation_fallback_logged", False)


def _config(d, e, swiglu_limit=None):
    return MoeRunnerConfig(
        num_experts=e,
        num_local_experts=e,
        hidden_size=64,
        intermediate_size_per_partition=d,
        layer_id=0,
        top_k=8,
        swiglu_limit=swiglu_limit,
    )


def _quant_info(d, e, block_shape=None):
    hidden = 64
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty(e, 2 * d, hidden, dtype=torch.bfloat16),
        w2_weight=torch.empty(e, hidden, d, dtype=torch.bfloat16),
        use_fp8=False,
        block_shape=block_shape,
    )


def _hidden(rows=8, hidden=64):
    return torch.empty(rows, hidden, dtype=torch.bfloat16)


def test_clamp_with_d_div_8_below_experts_forces_compact(caplog):
    # Hy4 at TP8: D = 2048 / 8 = 256, E = 256 -> D // 8 = 32 < 256.
    with caplog.at_level("INFO", logger=deep_gemm.__name__):
        assert not _should_use_masked_standard_layout(
            _config(d=256, e=256, swiglu_limit=10.0),
            _quant_info(d=256, e=256, block_shape=[128, 128]),
            _hidden(),
        )
    assert "masked standard layout disabled" in caplog.text


def test_clamp_with_group32_mxfp8_forces_compact():
    # D // 8 >= E holds here; the group-32 static_assert alone must suffice.
    assert not _should_use_masked_standard_layout(
        _config(d=2048, e=256, swiglu_limit=10.0),
        _quant_info(d=2048, e=256, block_shape=[1, 32]),
        _hidden(),
    )


def test_clamp_with_supported_shape_keeps_masked():
    # DSV4-style TP1 shape: D // 8 = 256 >= E, group 128 -> guard must not fire.
    assert _should_use_masked_standard_layout(
        _config(d=2048, e=256, swiglu_limit=10.0),
        _quant_info(d=2048, e=256, block_shape=[128, 128]),
        _hidden(),
    )


def test_no_clamp_unaffected():
    assert _should_use_masked_standard_layout(
        _config(d=256, e=256),
        _quant_info(d=256, e=256, block_shape=[1, 32]),
        _hidden(),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
