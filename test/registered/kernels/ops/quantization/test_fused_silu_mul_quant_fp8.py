"""Correctness tests for fused_silu_mul_quant_fp8 kernel.

The reference is computed in pure PyTorch: silu(gate) * up followed by
per_token_group_quant_fp8. The fused kernel performs both steps in a single
Triton launch, so we verify that the output fp8 codes and scales match the
two-step baseline within FP8 quantization tolerance.

Test strategy:
  - scale_diff: max abs difference of per-group scales (should be ~0)
  - fp8_match: percentage of FP8 outputs that are bit-exact
  - cosine_sim: cosine similarity of dequantized outputs (should be > 0.9999)
  - rel_err: mean relative error of dequantized outputs (should be < 0.01)
"""

import sys

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_silu_mul_quant_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _can_use_fused_silu_mul_quant_fp8,
    fused_experts_impl,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

G = 128
EPS = 1e-10


# --------------------------------------------------------------------------- #
# Pure-torch reference (two-step baseline).
# --------------------------------------------------------------------------- #
def ref_silu_mul_quant(x: torch.Tensor, gs: int):
    """Two-step baseline: silu_and_mul + per_token_group_quant_fp8."""
    d = x.shape[-1] // 2
    result = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    fp8_out, scale = per_token_group_quant_fp8(result, gs)
    return fp8_out, scale


def ref_silu_mul_quant_swiglu_clamp(x: torch.Tensor, gs: int, limit: float):
    """Two-step baseline with DeepSeek-V4 SwiGLU clamp.

    gate is clamped to [-inf, limit]; up is clamped to [-limit, limit].
    Then silu(gate) * up is computed and quantized.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    result = (torch.nn.functional.silu(gate) * up).to(x.dtype)
    fp8_out, scale = per_token_group_quant_fp8(result, gs)
    return fp8_out, scale


def dequantize(fp8_codes: torch.Tensor, scales: torch.Tensor, gs: int):
    """Dequantize fp8 codes back to float32 using per-group scales."""
    return fp8_codes.float() * scales.repeat_interleave(gs, dim=1).float()


# --------------------------------------------------------------------------- #
# Parametrized test configurations.
# --------------------------------------------------------------------------- @@
SIZES = [
    # (num_tokens, hidden_dim, group_size)
    (1, 2048, 128),
    (16, 2048, 128),
    (256, 2048, 128),
    (4096, 2048, 128),
    (8192, 2048, 128),
    (256, 4096, 128),
    (256, 8192, 128),
    (256, 2048, 64),
    (256, 2048, 256),
]

DTYPES = [torch.bfloat16, torch.float16]


def test_fused_silu_mul_quant_fp8_auto_dispatch():
    """The fast path is automatic only when all caller contracts are satisfied."""
    compatible = dict(
        hidden_size=4096,
        hidden_dtype=torch.bfloat16,
        use_fp8_w8a8=True,
        block_shape=[128, 128],
        filter_expert=False,
        activation="silu",
        is_gated=True,
        gemm1_alpha=None,
        gemm1_limit=None,
        hooks=None,
        fuse_swiglu_interleaved=False,
    )
    assert _can_use_fused_silu_mul_quant_fp8(**compatible)

    incompatible_overrides = (
        {"hidden_size": 4100},
        {"hidden_dtype": torch.float32},
        {"use_fp8_w8a8": False},
        {"block_shape": None},
        {"block_shape": [128, 0]},
        {"filter_expert": True},
        {"activation": "gelu"},
        {"is_gated": False},
        {"gemm1_alpha": 1.0},
        {"gemm1_limit": 7.0},
        {"hooks": object()},
        {"fuse_swiglu_interleaved": True},
    )
    for override in incompatible_overrides:
        inputs = compatible | override
        assert not _can_use_fused_silu_mul_quant_fp8(**inputs), override


def test_fused_silu_mul_quant_fp8_auto_dispatch_moe_path(monkeypatch):
    """The compatible MoE path invokes the fusion and matches its fallback."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_single_process_dist(master_port=29676)

    torch.manual_seed(42)
    num_tokens, hidden_size, intermediate_size = 8, 256, 256
    num_experts, topk, group_size = 4, 2, 128
    block_shape = [group_size, group_size]

    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.02)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.02)
        .to(torch.float8_e4m3fn)
    )
    w1_scale = torch.ones(
        num_experts,
        2 * intermediate_size // group_size,
        hidden_size // group_size,
        device="cuda",
        dtype=torch.float32,
    )
    w2_scale = torch.ones(
        num_experts,
        hidden_size // group_size,
        intermediate_size // group_size,
        device="cuda",
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]] * 2, device="cuda", dtype=torch.int32
    )
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32
    )

    def run_moe():
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            filter_expert=False,
        )

    original_fused_kernel = fused_moe_module.fused_silu_mul_quant_fp8
    fused_kernel_calls = 0

    def record_fused_kernel(*args, **kwargs):
        nonlocal fused_kernel_calls
        fused_kernel_calls += 1
        return original_fused_kernel(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe_module, "fused_silu_mul_quant_fp8", record_fused_kernel
    )
    fused_output = run_moe()
    assert fused_kernel_calls == 1

    monkeypatch.setattr(
        fused_moe_module, "_can_use_fused_silu_mul_quant_fp8", lambda **_: False
    )
    fallback_output = run_moe()
    assert fused_kernel_calls == 1
    torch.testing.assert_close(fused_output, fallback_output, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("num_tokens, hidden_dim, group_size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_silu_mul_quant_fp8_correctness(
    num_tokens, hidden_dim, group_size, dtype
):
    """Verify fused kernel matches the two-step baseline."""
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * hidden_dim, device="cuda", dtype=dtype) * 3.0

    # Baseline
    ref_fp8, ref_scale = ref_silu_mul_quant(x, group_size)

    # Fused
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, group_size)

    # Scale difference
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, (
        f"scale_diff={scale_diff} exceeds tolerance for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # FP8 bit-exact match rate
    fp8_match = (ref_fp8 == fused_fp8).float().mean().item()
    assert fp8_match > 0.95, (
        f"fp8_match={fp8_match:.1%} below 95% for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # Dequantized cosine similarity
    ref_dq = dequantize(ref_fp8, ref_scale, group_size)
    fused_dq = dequantize(fused_fp8, fused_scale, group_size)
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_dq.flatten().unsqueeze(0), fused_dq.flatten().unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, (
        f"cosine_sim={cosine_sim} below 0.9999 for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )

    # Relative error
    denom = ref_dq.flatten().abs().clamp(min=1e-6)
    rel_err = ((ref_dq.flatten() - fused_dq.flatten()).abs() / denom).mean().item()
    assert rel_err < 0.01, (
        f"rel_err={rel_err} exceeds 0.01 for "
        f"shape=({num_tokens}, {hidden_dim}), gs={group_size}, dtype={dtype}"
    )


def test_fused_silu_mul_quant_fp8_zero_input():
    """All-zero input should produce all-zero output and near-zero scales."""
    x = torch.zeros(16, 2 * 2048, device="cuda", dtype=torch.bfloat16)
    fp8_out, scale = fused_silu_mul_quant_fp8(x, G)
    max_scale = scale.max().item()
    assert max_scale < 1e-5, f"Expected near-zero scale, got {max_scale}"
    assert fp8_out.float().abs().max().item() == 0.0, "Expected all-zero fp8 output"


def test_fused_silu_mul_quant_fp8_negative_input():
    """Negative input values should be handled correctly."""
    x = -torch.randn(16, 2 * 2048, device="cuda", dtype=torch.bfloat16) * 3.0
    ref_fp8, ref_scale = ref_silu_mul_quant(x, G)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, G)
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"Negative input scale_diff={scale_diff}"


@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_fused_silu_mul_quant_fp8_group_sizes(group_size):
    """Test various group sizes."""
    hidden_dim = group_size * 4  # ensure divisibility
    x = torch.randn(8, 2 * hidden_dim, device="cuda", dtype=torch.bfloat16)
    ref_fp8, ref_scale = ref_silu_mul_quant(x, group_size)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, group_size)
    assert ref_fp8.shape == fused_fp8.shape
    assert ref_scale.shape == fused_scale.shape
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"gs={group_size} scale_diff={scale_diff}"


# --------------------------------------------------------------------------- #
# P2: swiglu_limit clamp correctness.
#
# DeepSeek-V4 uses swiglu_limit=10.0 (see config.json "swiglu_limit": 10.0).
# The non-fused path asserts swiglu_limit == 10. We test primarily with the
# DSV4 value, plus a few edge cases.
# --------------------------------------------------------------------------- #
DSV4_SWIGLU_LIMIT = 10.0  # DeepSeek-V4 actual value from config.json


@pytest.mark.parametrize("swiglu_limit", [1.0, 5.0, DSV4_SWIGLU_LIMIT])
def test_fused_silu_mul_quant_fp8_swiglu_limit(swiglu_limit):
    """swiglu_limit must clamp gate to [-inf, L] and up to [-L, L] before silu*up.

    This exercises the DeepSeek-V4 activation contract. We use inputs with
    large magnitudes so the clamp is guaranteed to trigger.
    """
    torch.manual_seed(42)
    x = torch.randn(64, 2 * 2048, device="cuda", dtype=torch.bfloat16) * 20.0

    # Reference with clamp
    ref_fp8, ref_scale = ref_silu_mul_quant_swiglu_clamp(x, G, swiglu_limit)
    # Fused with clamp
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, G, swiglu_limit=swiglu_limit)

    # Scales should match closely
    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"swiglu_limit={swiglu_limit} scale_diff={scale_diff}"

    # FP8 codes should be mostly bit-exact
    fp8_match = (ref_fp8 == fused_fp8).float().mean().item()
    assert fp8_match > 0.95, f"swiglu_limit={swiglu_limit} fp8_match={fp8_match:.1%}"

    # Dequantized cosine similarity
    ref_dq = dequantize(ref_fp8, ref_scale, G)
    fused_dq = dequantize(fused_fp8, fused_scale, G)
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_dq.flatten().unsqueeze(0), fused_dq.flatten().unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, f"swiglu_limit={swiglu_limit} cosine_sim={cosine_sim}"


def test_fused_silu_mul_quant_fp8_swiglu_limit_dsv4():
    """DeepSeek-V4 exact configuration: swiglu_limit=10.0, hidden_dim=2048,
    group_size=128, bf16.

    DSV4 config.json: swiglu_limit=10.0, moe_intermediate_size=2048,
    weight_block_size=[128, 128]. This test uses the exact production values.
    """
    torch.manual_seed(42)
    # DSV4 expert weights are [2048, 2048] (w1) and [4096, 1024] (w2).
    # hidden_dim=4096, intermediate=2048 -> gate_up dim = 2*2048 = 4096.
    # But the fused kernel operates on the gate_up output (intermediate_cache1)
    # which has shape [num_tokens, 2 * intermediate_size].
    num_tokens = 256
    intermediate_size = 2048  # DSV4 moe_intermediate_size
    x = (
        torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=torch.bfloat16
        )
        * 20.0
    )

    ref_fp8, ref_scale = ref_silu_mul_quant_swiglu_clamp(x, G, DSV4_SWIGLU_LIMIT)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(
        x, G, swiglu_limit=DSV4_SWIGLU_LIMIT
    )

    scale_diff = (ref_scale.float() - fused_scale.float()).abs().max().item()
    assert scale_diff < 0.01, f"DSV4 config scale_diff={scale_diff}"

    fp8_match = (ref_fp8 == fused_fp8).float().mean().item()
    assert fp8_match > 0.95, f"DSV4 config fp8_match={fp8_match:.1%}"

    ref_dq = dequantize(ref_fp8, ref_scale, G)
    fused_dq = dequantize(fused_fp8, fused_scale, G)
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_dq.flatten().unsqueeze(0), fused_dq.flatten().unsqueeze(0)
    ).item()
    assert cosine_sim > 0.9999, f"DSV4 config cosine_sim={cosine_sim}"


def test_fused_silu_mul_quant_fp8_swiglu_limit_changes_output():
    """swiglu_limit=10.0 (DSV4 value) must produce different output than no clamp
    when inputs exceed the limit."""
    torch.manual_seed(42)
    x = torch.randn(64, 2 * 2048, device="cuda", dtype=torch.bfloat16) * 20.0

    no_clamp_fp8, no_clamp_scale = fused_silu_mul_quant_fp8(x, G, swiglu_limit=0.0)
    clamp_fp8, clamp_scale = fused_silu_mul_quant_fp8(
        x, G, swiglu_limit=DSV4_SWIGLU_LIMIT
    )

    # With large inputs and clamp=10.0, outputs must differ
    diff_rate = (no_clamp_fp8 != clamp_fp8).float().mean().item()
    assert diff_rate > 0.01, f"Expected clamp to change outputs, diff_rate={diff_rate}"

    # Clamped scales should be smaller (values are bounded by limit)
    assert clamp_scale.max().item() <= no_clamp_scale.max().item() + 1e-6


def test_fused_silu_mul_quant_fp8_swiglu_limit_zero_is_noop():
    """swiglu_limit=0.0 should behave identically to no clamp (default)."""
    torch.manual_seed(42)
    x = torch.randn(32, 2 * 2048, device="cuda", dtype=torch.bfloat16) * 5.0

    default_fp8, default_scale = fused_silu_mul_quant_fp8(x, G)
    zero_fp8, zero_scale = fused_silu_mul_quant_fp8(x, G, swiglu_limit=0.0)

    assert torch.equal(default_fp8, zero_fp8), "swiglu_limit=0 should be a no-op"
    assert torch.equal(default_scale, zero_scale), "scales should match"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
