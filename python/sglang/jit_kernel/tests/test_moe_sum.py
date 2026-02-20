"""
Correctness tests for the moe_sum JIT kernel.

Validates against a pure-PyTorch reference and, when sgl_kernel is available,
cross-checks against the AOT implementation.
"""

import itertools
import os

import pytest
import torch

from sglang.jit_kernel.moe_sum import moe_sum

try:
    from sgl_kernel import moe_sum as moe_sum_aot

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

NUM_TOKENS_FULL = [1, 16, 128, 512, 1024]
NUM_TOKENS_CI = [1, 128, 1024]

# topk 1-4 hit static dispatch; 6 exercises general fallback
TOPK_FULL = [1, 2, 3, 4, 6]
TOPK_CI = [2, 4, 6]

HIDDEN_DIM_FULL = [64, 1024, 4096]
HIDDEN_DIM_CI = [64, 4096]

DTYPES_FULL = [torch.float32, torch.float16, torch.bfloat16]
DTYPES_CI = [torch.float32, torch.bfloat16]

NUM_TOKENS = NUM_TOKENS_CI if _is_ci else NUM_TOKENS_FULL
TOPK_LIST = TOPK_CI if _is_ci else TOPK_FULL
HIDDEN_DIMS = HIDDEN_DIM_CI if _is_ci else HIDDEN_DIM_FULL
DTYPES = DTYPES_CI if _is_ci else DTYPES_FULL


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------


def moe_sum_ref(x: torch.Tensor) -> torch.Tensor:
    """Sum over the topk dim (dim=1)."""
    return x.sum(dim=1)


# ---------------------------------------------------------------------------
# Correctness: JIT vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, topk, hidden_dim",
    list(itertools.product(NUM_TOKENS, TOPK_LIST, HIDDEN_DIMS)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_sum_vs_ref(num_tokens, topk, hidden_dim, dtype):
    torch.manual_seed(num_tokens * topk * hidden_dim)
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device="cuda")
    out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    moe_sum(x, out)
    ref = moe_sum_ref(x)

    # Kernel accumulates in native dtype; PyTorch ref upcasts to float.
    # Allow wider tolerance for fp16/bf16.
    atol = 0.05 if dtype != torch.float32 else 1e-5
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(
        out, ref, atol=atol, rtol=rtol
    ), f"Mismatch (dtype={dtype}, tokens={num_tokens}, topk={topk}, hidden={hidden_dim})"


# ---------------------------------------------------------------------------
# Output shape and dtype preserved
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype():
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn((64, 4, 1024), dtype=dtype, device="cuda")
        out = torch.empty((64, 1024), dtype=dtype, device="cuda")
        moe_sum(x, out)
        assert out.shape == (64, 1024)
        assert out.dtype == dtype


# ---------------------------------------------------------------------------
# General fallback (topk not in 1-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topk", [5, 6, 8])
def test_general_fallback(topk):
    num_tokens, hidden_dim = 64, 512
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=torch.float32, device="cuda")
    out = torch.empty((num_tokens, hidden_dim), dtype=torch.float32, device="cuda")
    moe_sum(x, out)
    ref = moe_sum_ref(x)
    assert torch.allclose(
        out, ref, atol=1e-5
    ), f"General fallback mismatch (topk={topk})"


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize(
    "num_tokens, topk, hidden_dim",
    list(itertools.product([1, 128, 1024], [2, 3, 4], [256, 4096])),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_sum_vs_aot(num_tokens, topk, hidden_dim, dtype):
    torch.manual_seed(42)
    x = torch.randn((num_tokens, topk, hidden_dim), dtype=dtype, device="cuda")

    out_jit = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    out_aot = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")

    moe_sum(x, out_jit)
    moe_sum_aot(x, out_aot)

    assert torch.allclose(
        out_jit, out_aot, atol=1e-5, rtol=1e-5
    ), f"JIT vs AOT mismatch (dtype={dtype}, tokens={num_tokens}, topk={topk}, hidden={hidden_dim})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
