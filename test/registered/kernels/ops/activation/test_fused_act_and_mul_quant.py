"""
Correctness tests for the fused activation + per-token-group FP8 quantization JIT kernel.

Validates against the unfused reference path:
    act_out = run_activation(op_name, input)
    ref_q, ref_scale = sglang_per_token_group_quant_fp8(act_out, group_size)

NOTE: The fused kernel keeps activation results in float32 registers before
quantizing, while the unfused path truncates to bf16/fp16 between activation
and quantization. This causes small numerical differences in scale (~1e-6)
and occasionally different FP8 rounding for borderline values.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.activation.activation import (
    SUPPORTED_ACTIVATIONS,
    run_activation,
    run_activation_quant,
)
from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=40, suite="nightly-kernel-1-gpu", nightly=True)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

OPS = list(SUPPORTED_ACTIVATIONS)  # ["silu", "gelu", "gelu_tanh"]
DTYPES = [torch.bfloat16, torch.float16]
GROUP_SIZE = 128

# Shapes: (num_tokens, hidden_dim * 2).
# hidden_dim must be divisible by GROUP_SIZE.
SHAPES = get_ci_test_range(
    full_range=[
        (1, 256),  # minimal: hidden=128, 1 group per token
        (7, 512),  # odd token count
        (64, 1024),
        (128, 2048),
        (256, 4096),
        (512, 8192),
        (1024, 12288),  # large hidden (Llama-65B intermediate)
        (4096, 2048),  # large token count
    ],
    ci_range=[
        (1, 256),
        (64, 1024),
        (256, 4096),
    ],
)

FILTER_SHAPES = get_ci_test_range(
    full_range=[(128, 1024), (256, 2048), (512, 4096), (1024, 8192)],
    ci_range=[(128, 1024), (256, 4096)],
)
EXPERT_STEPS = [1, 16]


# ---------------------------------------------------------------------------
# Tolerances: fused path keeps float32 precision through activation→quant,
# while unfused truncates to bf16/fp16 in between. Scale differs by ~1e-5,
# and FP8 values may differ by 1 ULP for borderline rounding.
# ---------------------------------------------------------------------------

SCALE_ATOL = 0.0
SCALE_RTOL = 0.0
# FP8 E4M3: allow 1 ULP difference due to rounding tie-breaking between
# different FP8 conversion paths (fp8_t(float) vs __nv_cvt_float2_to_fp8x2).
FP8_MAX_DIFF_ULPS = 1


def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert FP8 tensors are within FP8_MAX_DIFF_ULPS of each other."""
    diff = (actual.view(torch.uint8).int() - expected.view(torch.uint8).int()).abs()
    max_diff = diff.max().item()
    assert max_diff <= FP8_MAX_DIFF_ULPS, (
        f"FP8 mismatch: max ULP diff = {max_diff}, "
        f"mismatched elements = {(diff > FP8_MAX_DIFF_ULPS).sum().item()}/{diff.numel()}"
    )


def _reference_activation_quant(
    op_name: str, input: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused reference: activation then per-token-group FP8 quantization."""
    act_out = run_activation(op_name, input, None)
    ref_q, ref_scale = sglang_per_token_group_quant_fp8(act_out, group_size)
    return ref_q, ref_scale


# ---------------------------------------------------------------------------
# Basic correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_act_quant_correctness(
    op_name: str, dtype: torch.dtype, shape: tuple[int, int]
) -> None:
    """Fused kernel output must match unfused activation + quant."""
    num_tokens, dim = shape
    input = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")

    # Reference (unfused)
    ref_q, ref_scale = _reference_activation_quant(op_name, input, GROUP_SIZE)

    # Fused
    fused_q, fused_scale = run_activation_quant(op_name, input, group_size=GROUP_SIZE)

    # Quantized values should match exactly (both use same rounding)
    assert fused_q.dtype == torch.float8_e4m3fn
    assert fused_scale.dtype == torch.float32
    assert fused_q.shape == ref_q.shape
    assert fused_scale.shape == ref_scale.shape

    # Compare scales (small floating-point diff due to bf16 truncation in unfused path)
    torch.testing.assert_close(fused_scale, ref_scale, atol=SCALE_ATOL, rtol=SCALE_RTOL)
    # Compare quantized values (allow 1 ULP diff for borderline rounding)
    _assert_fp8_close(fused_q, ref_q)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_act_quant_preallocated_output(op_name: str, dtype: torch.dtype) -> None:
    """Test with pre-allocated output buffers."""
    num_tokens, dim = 64, 2048
    hidden = dim // 2
    input = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")

    output_q = torch.empty(num_tokens, hidden, dtype=torch.float8_e4m3fn, device="cuda")
    output_scale = torch.empty(
        num_tokens, hidden // GROUP_SIZE, dtype=torch.float32, device="cuda"
    )

    result_q, result_scale = run_activation_quant(
        op_name,
        input,
        output_q=output_q,
        output_scale=output_scale,
        group_size=GROUP_SIZE,
    )

    # Must return the same buffers
    assert result_q.data_ptr() == output_q.data_ptr()
    assert result_scale.data_ptr() == output_scale.data_ptr()

    # Compare with reference
    ref_q, ref_scale = _reference_activation_quant(op_name, input, GROUP_SIZE)
    torch.testing.assert_close(
        result_scale, ref_scale, atol=SCALE_ATOL, rtol=SCALE_RTOL
    )
    _assert_fp8_close(result_q, ref_q)


# ---------------------------------------------------------------------------
# Expert-filtered tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", FILTER_SHAPES)
@pytest.mark.parametrize("expert_step", EXPERT_STEPS)
def test_fused_act_quant_filter_expert(
    op_name: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    expert_step: int,
) -> None:
    """Rows with expert_ids == -1 must leave output untouched (sentinel preserved)."""
    num_tokens, dim = shape
    hidden = dim // 2
    input = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")

    # Pre-fill with sentinel
    sentinel_q = torch.zeros(
        num_tokens, hidden, dtype=torch.float8_e4m3fn, device="cuda"
    )
    sentinel_scale = torch.full(
        (num_tokens, hidden // GROUP_SIZE),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    output_q = sentinel_q.clone()
    output_scale = sentinel_scale.clone()

    # Create expert_ids with some -1 entries
    num_groups = (num_tokens + expert_step - 1) // expert_step
    expert_ids = torch.randint(
        low=0, high=8, size=(num_groups,), dtype=torch.int32, device="cuda"
    )
    skip_mask = torch.rand(num_groups, device="cuda") < 0.4
    expert_ids[skip_mask] = -1

    result_q, result_scale = run_activation_quant(
        op_name,
        input,
        output_q=output_q,
        output_scale=output_scale,
        expert_ids=expert_ids,
        expert_step=expert_step,
        group_size=GROUP_SIZE,
    )

    # Check that skipped rows are untouched
    token_skip = skip_mask[torch.arange(num_tokens, device="cuda") // expert_step]
    if token_skip.any():
        skipped_q = result_q[token_skip]
        skipped_scale = result_scale[token_skip]
        # quantized values should remain zero (sentinel)
        assert torch.equal(
            skipped_q.view(torch.uint8),
            sentinel_q[token_skip].view(torch.uint8),
        ), "fused kernel modified quantized output for skipped rows"
        # scales should remain NaN (sentinel)
        assert torch.isnan(
            skipped_scale
        ).all(), "fused kernel modified scale output for skipped rows"

    # Check that non-skipped rows match the unfused reference
    kept = ~token_skip
    if kept.any():
        # Run unfused reference on kept tokens only for comparison
        ref_q, ref_scale = _reference_activation_quant(op_name, input, GROUP_SIZE)
        torch.testing.assert_close(
            result_scale[kept], ref_scale[kept], atol=SCALE_ATOL, rtol=SCALE_RTOL
        )
        _assert_fp8_close(result_q[kept], ref_q[kept])


@pytest.mark.parametrize("op_name", OPS)
def test_fused_act_quant_all_skipped(op_name: str) -> None:
    """If every expert_id is -1, output must be entirely untouched."""
    num_tokens, dim = 32, 1024
    hidden = dim // 2
    input = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device="cuda")

    output_q = torch.zeros(num_tokens, hidden, dtype=torch.float8_e4m3fn, device="cuda")
    output_scale = torch.full(
        (num_tokens, hidden // GROUP_SIZE),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    orig_q = output_q.clone()

    expert_ids = torch.full((num_tokens,), -1, dtype=torch.int32, device="cuda")
    run_activation_quant(
        op_name,
        input,
        output_q=output_q,
        output_scale=output_scale,
        expert_ids=expert_ids,
        expert_step=1,
        group_size=GROUP_SIZE,
    )

    assert torch.equal(output_q.view(torch.uint8), orig_q.view(torch.uint8))
    assert torch.isnan(output_scale).all()


@pytest.mark.parametrize("op_name", OPS)
def test_fused_act_quant_none_skipped(op_name: str) -> None:
    """No -1 in expert_ids must yield identical output to unfiltered path."""
    num_tokens, dim = 64, 2048
    dtype = torch.bfloat16
    input = torch.randn(num_tokens, dim, dtype=dtype, device="cuda")

    expert_ids = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
    filtered_q, filtered_scale = run_activation_quant(
        op_name,
        input,
        expert_ids=expert_ids,
        expert_step=1,
        group_size=GROUP_SIZE,
    )
    unfiltered_q, unfiltered_scale = run_activation_quant(
        op_name,
        input,
        group_size=GROUP_SIZE,
    )

    torch.testing.assert_close(filtered_scale, unfiltered_scale, atol=0.0, rtol=0.0)
    assert torch.equal(filtered_q.view(torch.uint8), unfiltered_q.view(torch.uint8))


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", OPS)
def test_fused_act_quant_single_token(op_name: str) -> None:
    """Single-token input must work correctly."""
    input = torch.randn(1, 256, dtype=torch.bfloat16, device="cuda")
    fused_q, fused_scale = run_activation_quant(op_name, input, group_size=GROUP_SIZE)
    ref_q, ref_scale = _reference_activation_quant(op_name, input, GROUP_SIZE)

    torch.testing.assert_close(fused_scale, ref_scale, atol=SCALE_ATOL, rtol=SCALE_RTOL)
    _assert_fp8_close(fused_q, ref_q)


@pytest.mark.parametrize("op_name", OPS)
def test_fused_act_quant_zero_input(op_name: str) -> None:
    """All-zero input should produce all-zero quantized output."""
    input = torch.zeros(16, 512, dtype=torch.bfloat16, device="cuda")
    fused_q, fused_scale = run_activation_quant(op_name, input, group_size=GROUP_SIZE)
    # act(0) * 0 = 0 for all activations, so quantized output should be 0
    assert torch.equal(
        fused_q.view(torch.uint8),
        torch.zeros_like(fused_q).view(torch.uint8),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
