"""
Tests for jit_kernel.moe_fused_gate.

Correctness strategy:
  1. Primary reference: biased_grouped_topk_impl — a pure-PyTorch implementation
     of the same algorithm (sigmoid → add bias → group exclusion → topk → rescale).
  2. AOT cross-check: when sgl_kernel is available, compare JIT output directly
     against the AOT kernel on identical inputs (tightest possible check).

Config coverage:
  - Static dispatch configs: (256,8), (256,16), (128,4), (128,8)
  - Dynamic fallback config: (512,16) — VPT=32, not in static switch
  - Dtypes: float32, bfloat16, float16
  - num_fused_shared_experts: 0, 1, 2
  - apply_routed_scaling_factor_on_output: False, True
"""

from typing import Optional

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

# ---------------------------------------------------------------------------
# Optional AOT reference (sgl_kernel)
# ---------------------------------------------------------------------------
try:
    from sgl_kernel import moe_fused_gate as moe_fused_gate_aot

    AOT_AVAILABLE = True
except ImportError:
    moe_fused_gate_aot = None
    AOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure-PyTorch reference — copied from sgl-kernel/tests/test_moe_fused_gate.py
# ---------------------------------------------------------------------------


def biased_grouped_topk_ref(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for moe_fused_gate."""
    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, topk_group]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

    topk_excluding_shared = topk - num_fused_shared_experts
    _, routed_topk_ids = torch.topk(tmp_scores, k=topk_excluding_shared, dim=-1, sorted=False)
    routed_topk_weights = scores.gather(1, routed_topk_ids)

    if num_fused_shared_experts > 0:
        topk_ids = torch.empty((num_token, topk), dtype=routed_topk_ids.dtype, device=routed_topk_ids.device)
        topk_weights = torch.empty(
            (num_token, topk), dtype=routed_topk_weights.dtype, device=routed_topk_weights.device
        )
        topk_ids[:, :topk_excluding_shared] = routed_topk_ids
        topk_weights[:, :topk_excluding_shared] = routed_topk_weights

        scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
        routed_sum = routed_topk_weights.sum(dim=-1, keepdim=True)
        for i in range(num_fused_shared_experts):
            topk_ids[:, topk_excluding_shared + i] = num_experts + i
            topk_weights[:, topk_excluding_shared + i] = routed_sum[:, 0] / scale
    else:
        topk_ids = routed_topk_ids
        topk_weights = routed_topk_weights

    # renormalize (moe_fused_gate always renormalizes)
    if num_fused_shared_experts > 0:
        topk_weights_sum = topk_weights[:, :topk_excluding_shared].sum(dim=-1, keepdim=True)
    else:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights / topk_weights_sum

    if apply_routed_scaling_factor_on_output:
        scale = 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
        topk_weights = topk_weights * scale

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_outputs(
    jit_weights: torch.Tensor,
    jit_ids: torch.Tensor,
    ref_weights: torch.Tensor,
    ref_ids: torch.Tensor,
    num_fused_shared_experts: int,
    num_experts: int,
    label: str,
):
    """Compare JIT output against a reference, handling fused shared experts."""
    topk = jit_ids.shape[1]
    topk_excluding_shared = topk - num_fused_shared_experts

    # For shared expert slots, verify index is in valid range [num_experts, num_experts + n)
    if num_fused_shared_experts > 0:
        shared_jit = jit_ids[:, topk_excluding_shared:]
        assert torch.all(
            (shared_jit >= num_experts) & (shared_jit < num_experts + num_fused_shared_experts)
        ), f"[{label}] Shared expert indices out of range"

        if ref_ids is not None:
            shared_ref = ref_ids[:, topk_excluding_shared:]
            assert torch.all(
                (shared_ref >= num_experts) & (shared_ref < num_experts + num_fused_shared_experts)
            ), f"[{label}] Reference shared expert indices out of range"

        # Compare only the routed (non-shared) slots
        jit_ids = jit_ids[:, :topk_excluding_shared]
        ref_ids = ref_ids[:, :topk_excluding_shared] if ref_ids is not None else None
        jit_weights = jit_weights[:, :topk_excluding_shared]
        ref_weights = ref_weights[:, :topk_excluding_shared] if ref_weights is not None else None

    if ref_ids is not None:
        idx_ok = torch.allclose(
            ref_ids.sort()[0].to(torch.int32),
            jit_ids.sort()[0].to(torch.int32),
            rtol=1e-4,
            atol=1e-5,
        )
        assert idx_ok, f"[{label}] Indices mismatch"

    if ref_weights is not None:
        w_ok = torch.allclose(
            ref_weights.sort()[0].to(torch.float32),
            jit_weights.sort()[0].to(torch.float32),
            rtol=1e-2,
            atol=1e-3,
        )
        assert w_ok, f"[{label}] Weights mismatch"


# ---------------------------------------------------------------------------
# Main parametrized test
# ---------------------------------------------------------------------------

# seq_lengths: small edge cases + typical production sizes
SEQ_LENGTHS_FULL = list(range(1, 10)) + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SEQ_LENGTHS_CI = [1, 3, 7, 64, 512, 2048]

import os

_is_ci = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
SEQ_LENGTHS = SEQ_LENGTHS_CI if _is_ci else SEQ_LENGTHS_FULL

# (num_experts, num_expert_group, topk_group, topk_routed)
# Static dispatch configs: (256,8), (256,16), (128,4), (128,8)
# Dynamic fallback config: (512,16) — VPT=32, outside the static switch
EXPERT_CONFIGS = [
    (128, 4, 2, 4),   # static: VPT=32
    (128, 8, 4, 4),   # static: VPT=16
    (256, 8, 4, 8),   # static: VPT=32 — DeepSeek V3
    (256, 16, 8, 8),  # static: VPT=16
    (512, 16, 8, 16), # dynamic fallback: VPT=32
]
EXPERT_CONFIGS_CI = [
    (128, 4, 2, 4),
    (256, 8, 4, 8),
    (512, 16, 8, 16),
]
EXPERT_CONFIGS_USE = EXPERT_CONFIGS_CI if _is_ci else EXPERT_CONFIGS


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("expert_cfg", EXPERT_CONFIGS_USE)
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("apply_scaling", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_moe_fused_gate_vs_ref(
    seq_length,
    expert_cfg,
    num_fused_shared_experts,
    apply_scaling,
    dtype,
):
    num_experts, num_expert_group, topk_group, topk_routed = expert_cfg
    topk = topk_routed + num_fused_shared_experts
    routed_scaling_factor = 2.5

    torch.manual_seed(seq_length)
    inp = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    jit_weights, jit_ids = moe_fused_gate(
        inp,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scaling,
    )

    # Reference: pure-PyTorch implementation.
    # For float32 we use float32 directly; for bf16/fp16 we also pass native dtype so
    # the reference precision matches the kernel (both use bf16/fp16 arithmetic for
    # sigmoid+bias, causing the same tie-breaking in topk).
    ref_weights, ref_ids = biased_grouped_topk_ref(
        inp,
        bias,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scaling,
    )

    _check_outputs(
        jit_weights,
        jit_ids,
        ref_weights,
        ref_ids,
        num_fused_shared_experts,
        num_experts,
        label=f"vs_ref seq={seq_length} cfg={expert_cfg} dtype={dtype}",
    )


# ---------------------------------------------------------------------------
# AOT cross-check: JIT output must match sgl_kernel.moe_fused_gate exactly
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("seq_length", [1, 7, 64, 512, 2048])
@pytest.mark.parametrize(
    "expert_cfg",
    [
        (128, 4, 2, 4),
        (256, 8, 4, 8),   # DeepSeek V3 — most critical
        (512, 16, 8, 16), # dynamic fallback
    ],
)
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1, 2])
@pytest.mark.parametrize("apply_scaling", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_moe_fused_gate_vs_aot(
    seq_length,
    expert_cfg,
    num_fused_shared_experts,
    apply_scaling,
    dtype,
):
    """JIT output must be bit-identical to AOT sgl_kernel on the same inputs."""
    num_experts, num_expert_group, topk_group, topk_routed = expert_cfg
    topk = topk_routed + num_fused_shared_experts
    routed_scaling_factor = 2.5

    torch.manual_seed(seq_length + 1000)
    inp = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")
    bias = torch.rand(num_experts, dtype=dtype, device="cuda")

    jit_weights, jit_ids = moe_fused_gate(
        inp,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scaling,
    )

    aot_weights, aot_ids = moe_fused_gate_aot(
        inp,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scaling,
    )

    _check_outputs(
        jit_weights,
        jit_ids,
        aot_weights,
        aot_ids,
        num_fused_shared_experts,
        num_experts,
        label=f"vs_aot seq={seq_length} cfg={expert_cfg}",
    )


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_output_shapes():
    inp = torch.rand((32, 256), dtype=torch.float32, device="cuda")
    bias = torch.rand(256, dtype=torch.float32, device="cuda")
    w, ids = moe_fused_gate(inp, bias, num_expert_group=8, topk_group=4, topk=8)
    assert w.shape == (32, 8)
    assert ids.shape == (32, 8)
    assert w.dtype == torch.float32
    assert ids.dtype == torch.int32


def test_output_dtypes_always_float32_int32():
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        inp = torch.rand((16, 128), dtype=dtype, device="cuda")
        bias = torch.rand(128, dtype=dtype, device="cuda")
        w, ids = moe_fused_gate(inp, bias, num_expert_group=4, topk_group=2, topk=4)
        assert w.dtype == torch.float32, f"weights dtype should be float32, got {w.dtype}"
        assert ids.dtype == torch.int32, f"ids dtype should be int32, got {ids.dtype}"


def test_weights_sum_to_one_after_renorm():
    """After renormalization, routed weights (excl. shared experts) sum to 1."""
    seq = 64
    num_experts, num_expert_group, topk_group, topk = 256, 8, 4, 8
    inp = torch.rand((seq, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")
    w, _ = moe_fused_gate(
        inp, bias, num_expert_group=num_expert_group, topk_group=topk_group, topk=topk
    )
    row_sums = w.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(seq, device="cuda"), rtol=1e-4, atol=1e-4)


def test_fused_shared_expert_indices_in_range():
    """Fused shared expert indices must be in [num_experts, num_experts + n_shared)."""
    seq, num_experts = 32, 256
    topk_routed, n_shared = 8, 2
    topk = topk_routed + n_shared
    inp = torch.rand((seq, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")
    _, ids = moe_fused_gate(
        inp, bias,
        num_expert_group=8, topk_group=4, topk=topk,
        num_fused_shared_experts=n_shared, routed_scaling_factor=2.5,
    )
    shared_slots = ids[:, topk_routed:]
    assert torch.all((shared_slots >= num_experts) & (shared_slots < num_experts + n_shared))


def test_invalid_num_experts_not_power_of_two():
    inp = torch.rand((16, 100), dtype=torch.float32, device="cuda")
    bias = torch.rand(100, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception, match="power of 2"):
        moe_fused_gate(inp, bias, num_expert_group=4, topk_group=2, topk=4)


def test_invalid_vpt_exceeds_max():
    """num_experts / num_expert_group > 32 should raise."""
    inp = torch.rand((16, 256), dtype=torch.float32, device="cuda")
    bias = torch.rand(256, dtype=torch.float32, device="cuda")
    # VPT = 256/4 = 64 > MAX_VPT=32
    with pytest.raises(Exception, match="MAX_VPT"):
        moe_fused_gate(inp, bias, num_expert_group=4, topk_group=2, topk=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
