"""
Correctness tests for the moe_sum_reduce JIT kernel.

Validates against a pure-PyTorch reference and, when sgl_kernel is available,
cross-checks against the AOT implementation.
"""

import itertools
import os

import pytest
import torch

from sglang.jit_kernel.moe_sum_reduce import moe_sum_reduce

try:
    from sgl_kernel import moe_sum_reduce as moe_sum_reduce_aot

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# CI / full-range helpers
# ---------------------------------------------------------------------------

_is_ci = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

NUM_TOKENS_FULL = [1, 16, 64, 128, 256, 512, 1024]
NUM_TOKENS_CI = [1, 64, 512]

# topk values: 2, 4, 8, 9 hit static-dispatch; 3 exercises general fallback
TOPK_FULL = [2, 4, 8, 9, 3]
TOPK_CI = [2, 4, 3]

HIDDEN_DIM_FULL = [256, 1024, 4096]
HIDDEN_DIM_CI = [256, 4096]

DTYPES_FULL = [torch.float32, torch.float16, torch.bfloat16]
DTYPES_CI = [torch.float32, torch.bfloat16]

NUM_TOKENS = NUM_TOKENS_CI if _is_ci else NUM_TOKENS_FULL
TOPK_LIST = TOPK_CI if _is_ci else TOPK_FULL
HIDDEN_DIMS = HIDDEN_DIM_CI if _is_ci else HIDDEN_DIM_FULL
DTYPES = DTYPES_CI if _is_ci else DTYPES_FULL


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------


def moe_sum_reduce_ref(
    x: torch.Tensor,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Sum over the topk dim (dim=1) and multiply by scale."""
    return (x.float().sum(dim=1) * routed_scaling_factor).to(x.dtype)


# ---------------------------------------------------------------------------
# Correctness: JIT vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, topk, hidden_dim",
    list(itertools.product(NUM_TOKENS, TOPK_LIST, HIDDEN_DIMS)),
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 0.5])
def test_moe_sum_reduce_vs_ref(
    num_tokens, topk, hidden_dim, dtype, routed_scaling_factor
):
    torch.manual_seed(num_tokens * topk * hidden_dim)
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device="cuda")
    out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    moe_sum_reduce(x, out, routed_scaling_factor)
    ref = moe_sum_reduce_ref(x, routed_scaling_factor)

    # fp16/bf16 accumulation tolerance
    atol = 1e-2 if dtype != torch.float32 else 1e-5
    assert torch.allclose(out, ref, atol=atol, rtol=1e-3), (
        f"Mismatch (dtype={dtype}, tokens={num_tokens}, topk={topk}, "
        f"hidden={hidden_dim}, scale={routed_scaling_factor})"
    )


# ---------------------------------------------------------------------------
# Correctness: scale=0 → all-zeros output
# ---------------------------------------------------------------------------


def test_zero_scale():
    x = torch.randn((64, 4, 256), dtype=torch.float32, device="cuda")
    out = torch.empty((64, 256), dtype=torch.float32, device="cuda")
    moe_sum_reduce(x, out, 0.0)
    assert torch.all(out == 0.0), "scale=0 should produce all-zeros output"


# ---------------------------------------------------------------------------
# Correctness: BF16 vectorized fast path (token_num > 256, hidden_dim % 8 == 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_dim", [256, 4096, 8192])
@pytest.mark.parametrize("topk", [2, 4, 8])
def test_bf16_vec_fast_path(hidden_dim, topk):
    """token_num=512 > 256 triggers the vectorized BF16 path."""
    num_tokens = 512
    torch.manual_seed(42)
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
    moe_sum_reduce(x, out, 1.0)
    ref = moe_sum_reduce_ref(x, 1.0)
    assert torch.allclose(
        out, ref, atol=1e-2, rtol=1e-3
    ), f"BF16 vec path mismatch (hidden={hidden_dim}, topk={topk})"


# ---------------------------------------------------------------------------
# Output shape and dtype preserved
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    num_tokens, topk, hidden_dim = 64, 4, 1024
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device="cuda")
        out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")
        moe_sum_reduce(x, out, 1.0)
        assert out.shape == (num_tokens, hidden_dim)
        assert out.dtype == dtype


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize(
    "num_tokens, topk, hidden_dim",
    list(itertools.product([1, 128, 512], [2, 4, 8], [256, 4096])),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_sum_reduce_vs_aot(num_tokens, topk, hidden_dim, dtype):
    torch.manual_seed(42)
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device="cuda")

    out_jit = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    out_aot = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    moe_sum_reduce(x, out_jit, 1.0)
    moe_sum_reduce_aot(x, out_aot, 1.0)

    assert torch.allclose(
        out_jit, out_aot, atol=1e-5, rtol=1e-5
    ), f"JIT vs AOT mismatch (dtype={dtype}, tokens={num_tokens}, topk={topk}, hidden={hidden_dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
