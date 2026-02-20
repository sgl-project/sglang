"""
Correctness tests for the kimi_k2_moe_fused_gate JIT kernel.

Validates against a pure-PyTorch reference (kimi_k2_biased_topk_ref) and,
when sgl_kernel is available, also cross-checks against the AOT implementation.
"""

import os

import pytest
import torch

from sglang.jit_kernel.kimi_k2_moe_fused_gate import kimi_k2_moe_fused_gate

try:
    from sgl_kernel import kimi_k2_moe_fused_gate as kimi_k2_moe_fused_gate_aot

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure-PyTorch reference
# ---------------------------------------------------------------------------

NUM_EXPERTS = 384  # Kimi K2 is hard-coded to 384 experts


def kimi_k2_biased_topk_ref(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch reference for kimi_k2_moe_fused_gate.

    Algorithm (matches kernel):
      1. sigmoid(input) → scores
      2. scores + bias  → biased_scores
      3. topk on biased_scores → indices
      4. gather scores[indices] → weights
      5. renormalize and optionally scale
    """
    scores = input.sigmoid()
    biased_scores = scores + bias.unsqueeze(0)

    _, topk_ids = torch.topk(biased_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.float(), topk_ids.int()


# ---------------------------------------------------------------------------
# CI / full-range selection
# ---------------------------------------------------------------------------

_is_ci = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

SEQ_LENGTHS_FULL = list(range(1, 10)) + [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]
SEQ_LENGTHS_CI = [1, 3, 7, 64, 512, 2048]
SEQ_LENGTHS = SEQ_LENGTHS_CI if _is_ci else SEQ_LENGTHS_FULL

# ---------------------------------------------------------------------------
# Main correctness test: JIT vs PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("topk", [6])  # Kimi K2 uses topk=6
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [True, False])
def test_kimi_k2_moe_fused_gate_vs_ref(
    seq_length,
    topk,
    renormalize,
    apply_routed_scaling_factor_on_output,
):
    routed_scaling_factor = 2.872  # Kimi K2's scaling factor

    torch.manual_seed(seq_length)
    inp = torch.rand((seq_length, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    jit_weights, jit_ids = kimi_k2_moe_fused_gate(
        inp,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    ref_weights, ref_ids = kimi_k2_biased_topk_ref(
        inp,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Compare sorted weights (indices may differ for ties, weights must agree)
    assert torch.allclose(
        jit_weights.sort(dim=-1)[0],
        ref_weights.sort(dim=-1)[0],
        rtol=1e-3,
        atol=1e-4,
    ), (
        f"Weight mismatch: seq={seq_length}, topk={topk}, renorm={renormalize}, "
        f"scale_out={apply_routed_scaling_factor_on_output}"
    )

    # Verify that selected indices are valid expert IDs
    assert torch.all((jit_ids >= 0) & (jit_ids < NUM_EXPERTS)), "Expert indices out of range"


# ---------------------------------------------------------------------------
# Output shape and dtype checks
# ---------------------------------------------------------------------------


def test_output_shapes_and_dtypes():
    seq, topk = 64, 6
    inp = torch.rand((seq, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    weights, ids = kimi_k2_moe_fused_gate(inp, bias, topk=topk)

    assert weights.shape == (seq, topk), f"weights shape: {weights.shape}"
    assert ids.shape == (seq, topk), f"ids shape: {ids.shape}"
    assert weights.dtype == torch.float32, f"weights dtype: {weights.dtype}"
    assert ids.dtype == torch.int32, f"ids dtype: {ids.dtype}"


# ---------------------------------------------------------------------------
# Renormalization check: weights must sum to 1 per row when renormalize=True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", [1, 64, 512, 1024])
def test_weights_sum_to_one_when_renormalize(seq_length):
    topk = 6
    inp = torch.rand((seq_length, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    weights, _ = kimi_k2_moe_fused_gate(inp, bias, topk=topk, renormalize=True)
    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones(seq_length, device="cuda"), rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Kernel boundary: test both sides of the small/large-token threshold (512)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", [511, 512, 513, 1024])
def test_small_large_token_boundary(seq_length):
    """Verify consistent results across the small/large-token kernel boundary."""
    topk = 6
    torch.manual_seed(seq_length)
    inp = torch.rand((seq_length, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    weights, ids = kimi_k2_moe_fused_gate(inp, bias, topk=topk, renormalize=True)
    ref_weights, _ = kimi_k2_biased_topk_ref(
        inp,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
    )

    assert torch.allclose(
        weights.sort(dim=-1)[0],
        ref_weights.sort(dim=-1)[0],
        rtol=1e-3,
        atol=1e-4,
    ), f"Boundary mismatch at seq_length={seq_length}"


# ---------------------------------------------------------------------------
# Routed scaling factor check
# ---------------------------------------------------------------------------


def test_routed_scaling_factor_applied():
    seq, topk = 32, 6
    scale = 2.872
    inp = torch.rand((seq, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    weights_no_scale, _ = kimi_k2_moe_fused_gate(
        inp,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=False,
    )
    weights_scaled, _ = kimi_k2_moe_fused_gate(
        inp,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=True,
    )

    # scaled weights should be ≈ scale × unscaled weights
    torch.testing.assert_close(
        weights_scaled,
        weights_no_scale * scale,
        rtol=1e-4,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Invalid input checks
# ---------------------------------------------------------------------------


def test_invalid_num_experts():
    inp = torch.rand((16, 128), dtype=torch.float32, device="cuda")  # wrong: not 384
    bias = torch.rand(128, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):
        kimi_k2_moe_fused_gate(inp, bias, topk=6)


def test_invalid_dtype():
    inp = torch.rand((16, NUM_EXPERTS), dtype=torch.float16, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float16, device="cuda")
    with pytest.raises(Exception):
        kimi_k2_moe_fused_gate(inp, bias, topk=6)


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel (skipped when not available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("seq_length", [1, 64, 512, 1024, 4096])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_kimi_k2_moe_fused_gate_vs_aot(seq_length, apply_routed_scaling_factor_on_output):
    topk = 6
    routed_scaling_factor = 2.872
    torch.manual_seed(seq_length)
    inp = torch.rand((seq_length, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.rand(NUM_EXPERTS, dtype=torch.float32, device="cuda")

    kwargs = dict(
        topk=topk,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    jit_weights, jit_ids = kimi_k2_moe_fused_gate(inp, bias, **kwargs)
    aot_weights, aot_ids = kimi_k2_moe_fused_gate_aot(inp, bias, **kwargs)

    # Compare sorted (indices may differ for ties)
    assert torch.allclose(
        jit_weights.sort(dim=-1)[0],
        aot_weights.sort(dim=-1)[0].float(),
        rtol=1e-3,
        atol=1e-4,
    ), f"JIT vs AOT weight mismatch at seq={seq_length}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
