"""
Correctness tests for the moe_topk_sigmoid JIT kernel.

Validates against a pure-PyTorch reference and, when sgl_kernel is available,
cross-checks against the AOT implementation.
"""

import itertools
import os

import pytest
import torch

from sglang.jit_kernel.moe_topk_sigmoid import topk_sigmoid

try:
    from sgl_kernel import topk_sigmoid as topk_sigmoid_aot

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

# Power-of-2 configs covered by static dispatch (num_experts 1–256)
# Plus 48 (non-power-of-2) to exercise the fallback path
NUM_TOKENS_FULL = [1, 16, 128, 512, 1024, 2048]
NUM_TOKENS_CI = [1, 128, 1024]

NUM_EXPERTS_FULL = [4, 8, 16, 32, 64, 128, 256, 48]  # 48 = fallback path
NUM_EXPERTS_CI = [8, 64, 48]

TOPK_FULL = [1, 2, 4]
TOPK_CI = [1, 4]

DTYPES_FULL = [torch.float32, torch.float16, torch.bfloat16]
DTYPES_CI = [torch.float32, torch.bfloat16]

NUM_TOKENS = NUM_TOKENS_CI if _is_ci else NUM_TOKENS_FULL
NUM_EXPERTS = NUM_EXPERTS_CI if _is_ci else NUM_EXPERTS_FULL
TOPK_LIST = TOPK_CI if _is_ci else TOPK_FULL
DTYPES = DTYPES_CI if _is_ci else DTYPES_FULL


# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------


def topk_sigmoid_ref(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference: sigmoid → (add bias) → topk → (renormalize).
    Indices are selected on biased scores; weights are the unbiased sigmoid values.
    """
    scores = gating_output.float().sigmoid()
    biased = scores if correction_bias is None else scores + correction_bias.float()
    _, ref_ids = torch.topk(biased, k=topk, dim=-1)
    ref_weights = scores.gather(1, ref_ids)
    if renormalize:
        ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    return ref_weights.float(), ref_ids.int()


# ---------------------------------------------------------------------------
# Correctness: JIT vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(itertools.product(NUM_TOKENS, NUM_EXPERTS, TOPK_LIST)),
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_sigmoid_vs_ref(num_tokens, num_experts, topk, dtype, renormalize):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(num_tokens * num_experts)
    gating = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w, topk_i, gating, renormalize=renormalize)

    ref_w, ref_i = topk_sigmoid_ref(gating, topk, renormalize, correction_bias=None)

    # Compare sorted weights (indices may differ for ties when dtype != float32)
    assert torch.allclose(
        topk_w.sort(dim=-1)[0],
        ref_w.sort(dim=-1)[0],
        atol=1e-3,
        rtol=1e-3,
    ), f"Weight mismatch (dtype={dtype}, n_exp={num_experts}, topk={topk}, renorm={renormalize})"
    # Exact index match is only reliable for float32 (fp16/bf16 tie-breaking may differ)
    if dtype == torch.float32:
        assert torch.equal(
            topk_i, ref_i
        ), f"Index mismatch (dtype={dtype}, n_exp={num_experts}, topk={topk})"


# ---------------------------------------------------------------------------
# Correctness: with correction_bias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(itertools.product(NUM_TOKENS, NUM_EXPERTS, TOPK_LIST)),
)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_sigmoid_with_correction_bias(num_tokens, num_experts, topk, renormalize):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(num_tokens + num_experts)
    gating = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")

    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w, topk_i, gating, renormalize=renormalize, correction_bias=bias)

    ref_w, ref_i = topk_sigmoid_ref(gating, topk, renormalize, correction_bias=bias)

    assert torch.allclose(
        topk_w, ref_w, atol=1e-3, rtol=1e-3
    ), f"Weight mismatch with bias (n_exp={num_experts}, topk={topk}, renorm={renormalize})"
    assert torch.equal(
        topk_i, ref_i
    ), f"Index mismatch with bias (n_exp={num_experts}, topk={topk})"


# ---------------------------------------------------------------------------
# Renormalization: weights should sum to 1 per row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens, num_experts, topk", [(128, 64, 4), (1, 8, 2)])
def test_renormalize_sums_to_one(num_tokens, num_experts, topk):
    gating = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w, topk_i, gating, renormalize=True)
    row_sums = topk_w.sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones(num_tokens, device="cuda"), rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shapes_and_dtypes():
    num_tokens, num_experts, topk = 64, 128, 4
    gating = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w, topk_i, gating)

    assert topk_w.shape == (num_tokens, topk)
    assert topk_i.shape == (num_tokens, topk)
    assert topk_w.dtype == torch.float32
    assert topk_i.dtype == torch.int32


# ---------------------------------------------------------------------------
# Fallback path (non-power-of-2 experts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts", [48, 96])
def test_fallback_non_power_of_two(num_experts):
    num_tokens, topk = 64, 2
    gating = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w, topk_i, gating, renormalize=True)

    # Weights should be positive and sum to 1
    assert torch.all(topk_w > 0)
    torch.testing.assert_close(
        topk_w.sum(dim=-1), torch.ones(num_tokens, device="cuda"), rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk",
    list(itertools.product([1, 128, 1024], [8, 64, 128], [1, 4])),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_sigmoid_vs_aot(num_tokens, num_experts, topk, dtype, renormalize):
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(42)
    gating = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")

    topk_w_jit = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i_jit = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid(topk_w_jit, topk_i_jit, gating, renormalize=renormalize)

    topk_w_aot = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_i_aot = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    topk_sigmoid_aot(topk_w_aot, topk_i_aot, gating, renormalize=renormalize)

    assert torch.allclose(
        topk_w_jit, topk_w_aot, atol=1e-3, rtol=1e-3
    ), f"JIT vs AOT weight mismatch (dtype={dtype}, n_exp={num_experts}, topk={topk})"
    assert torch.equal(
        topk_i_jit, topk_i_aot
    ), f"JIT vs AOT index mismatch (dtype={dtype}, n_exp={num_experts}, topk={topk})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
