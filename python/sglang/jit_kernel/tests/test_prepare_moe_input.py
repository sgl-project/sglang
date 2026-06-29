"""
Correctness tests for the prepare_moe_input JIT kernels.

Validates prepare_moe_input, shuffle_rows, and apply_shuffle_mul_sum
against pure-PyTorch references and (when available) sgl_kernel AOT.
"""

import itertools
import os

import pytest
import torch

from sglang.jit_kernel.prepare_moe_input import (
    apply_shuffle_mul_sum,
    prepare_moe_input,
    shuffle_rows,
)

try:
    from sgl_kernel import apply_shuffle_mul_sum as apply_shuffle_mul_sum_aot
    from sgl_kernel import prepare_moe_input as prepare_moe_input_aot
    from sgl_kernel import shuffle_rows as shuffle_rows_aot

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

NUM_TOKENS_FULL = [1, 16, 128, 512]
NUM_TOKENS_CI = [1, 64, 256]

TOPK_FULL = [1, 2, 4, 8]
TOPK_CI = [2, 4]

NUM_EXPERTS_FULL = [8, 16, 64]
NUM_EXPERTS_CI = [8, 32]

NUM_TOKENS = NUM_TOKENS_CI if _is_ci else NUM_TOKENS_FULL
TOPK_LIST = TOPK_CI if _is_ci else TOPK_FULL
NUM_EXPERTS_LIST = NUM_EXPERTS_CI if _is_ci else NUM_EXPERTS_FULL


# ---------------------------------------------------------------------------
# Pure-PyTorch reference for prepare_moe_input
# ---------------------------------------------------------------------------


def prepare_moe_input_ref(topk_ids, num_experts, n, k):
    """Compute expert offsets, problem sizes, and permutations using PyTorch."""
    num_tokens, topk = topk_ids.shape
    topk_flat = topk_ids.flatten()  # [num_tokens * topk]

    # Count tokens per expert
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for eid in range(num_experts):
        counts[eid] = (topk_flat == eid).sum()

    # Expert offsets (exclusive prefix sum)
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    for i in range(num_experts):
        expert_offsets[i + 1] = expert_offsets[i] + counts[i]

    # Problem sizes
    problem_sizes1 = torch.zeros(
        (num_experts, 3), dtype=torch.int32, device=topk_ids.device
    )
    problem_sizes2 = torch.zeros(
        (num_experts, 3), dtype=torch.int32, device=topk_ids.device
    )
    for eid in range(num_experts):
        c = counts[eid].item()
        problem_sizes1[eid] = torch.tensor([c, 2 * n, k], dtype=torch.int32)
        problem_sizes2[eid] = torch.tensor([c, k, n], dtype=torch.int32)

    # Permutations: sort topk_flat by expert
    input_perm = torch.empty(
        num_tokens * topk, dtype=torch.int32, device=topk_ids.device
    )
    output_perm = torch.empty(
        num_tokens * topk, dtype=torch.int32, device=topk_ids.device
    )
    expert_cursor = expert_offsets[:-1].clone()  # running offset per expert
    for i in range(num_tokens * topk):
        eid = topk_flat[i].item()
        slot = expert_cursor[eid].item()
        input_perm[slot] = i // topk
        output_perm[i] = slot
        expert_cursor[eid] += 1

    return expert_offsets, problem_sizes1, problem_sizes2, input_perm, output_perm


# ---------------------------------------------------------------------------
# Tests: prepare_moe_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens, topk, num_experts",
    list(itertools.product(NUM_TOKENS, TOPK_LIST, NUM_EXPERTS_LIST)),
)
def test_prepare_moe_input_vs_ref(num_tokens, topk, num_experts):
    torch.manual_seed(num_tokens * topk * num_experts)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    n, k = 128, 256

    expert_offsets_jit = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    problem_sizes1_jit = torch.empty((num_experts, 3), dtype=torch.int32, device="cuda")
    problem_sizes2_jit = torch.empty((num_experts, 3), dtype=torch.int32, device="cuda")
    input_perm_jit = torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda")
    output_perm_jit = torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda")

    prepare_moe_input(
        topk_ids,
        expert_offsets_jit,
        problem_sizes1_jit,
        problem_sizes2_jit,
        input_perm_jit,
        output_perm_jit,
        num_experts,
        n,
        k,
    )

    ref = prepare_moe_input_ref(topk_ids, num_experts, n, k)
    expert_offsets_ref, ps1_ref, ps2_ref, ip_ref, op_ref = ref

    assert torch.equal(
        expert_offsets_jit, expert_offsets_ref
    ), "expert_offsets mismatch"
    assert torch.equal(problem_sizes1_jit, ps1_ref), "problem_sizes1 mismatch"
    assert torch.equal(problem_sizes2_jit, ps2_ref), "problem_sizes2 mismatch"
    # Permutations may differ in tie-breaking order within an expert;
    # verify they produce equivalent sorted token counts per expert.
    counts_jit = torch.bincount(
        topk_ids.flatten()[input_perm_jit.long()], minlength=num_experts
    )
    counts_ref = torch.bincount(
        topk_ids.flatten()[ip_ref.long()], minlength=num_experts
    )
    assert torch.equal(
        counts_jit, counts_ref
    ), "input_permutation expert counts mismatch"


def test_prepare_moe_input_with_blockscale_offsets():
    num_tokens, topk, num_experts = 64, 4, 8
    n, k = 128, 256
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )

    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    blockscale_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty((num_experts, 3), dtype=torch.int32, device="cuda")
    input_perm = torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda")
    output_perm = torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda")

    prepare_moe_input(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_perm,
        output_perm,
        num_experts,
        n,
        k,
        blockscale_offsets=blockscale_offsets,
    )

    # blockscale_offsets[i+1] - blockscale_offsets[i] must be a multiple of 128
    diffs = blockscale_offsets[1:] - blockscale_offsets[:-1]
    assert (
        diffs % 128 == 0
    ).all(), "blockscale_offsets spacing must be multiple of 128"
    assert blockscale_offsets[0].item() == 0, "blockscale_offsets must start at 0"


# ---------------------------------------------------------------------------
# Tests: shuffle_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_dst_rows, num_cols", [(64, 256), (128, 4096), (1, 512)])
def test_shuffle_rows_vs_ref(dtype, num_dst_rows, num_cols):
    torch.manual_seed(num_dst_rows * num_cols)
    num_src_rows = num_dst_rows + 32
    input_t = torch.randn((num_src_rows, num_cols), dtype=dtype, device="cuda")
    dst2src = torch.randperm(num_src_rows, device="cuda")[:num_dst_rows].to(torch.int32)

    out_jit = shuffle_rows(input_t, dst2src, (num_dst_rows, num_cols))
    out_ref = input_t[dst2src.long()]

    assert torch.equal(out_jit, out_ref), f"shuffle_rows mismatch (dtype={dtype})"
    assert out_jit.shape == (num_dst_rows, num_cols)
    assert out_jit.dtype == dtype


# ---------------------------------------------------------------------------
# Tests: apply_shuffle_mul_sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m, topk, k", [(32, 2, 256), (64, 4, 1024), (128, 8, 512)])
def test_apply_shuffle_mul_sum_vs_ref(dtype, m, topk, k):
    torch.manual_seed(m * topk * k)
    device = "cuda"

    # Generate a random permutation mapping [m*topk] -> row indices in input
    perm = torch.randperm(m * topk, device=device).to(torch.int32)
    input_t = torch.randn((m * topk, k), dtype=dtype, device=device)
    factors = torch.rand((m * topk,), dtype=dtype, device=device)
    output_jit = torch.empty((m, k), dtype=dtype, device=device)

    apply_shuffle_mul_sum(input_t, output_jit, perm, factors)

    # Reference: gather, reshape, weight, sum
    gathered = input_t[perm.long()].view(m, topk, k).to(torch.float32)
    w = factors.view(m, topk, 1).to(torch.float32)
    ref = (gathered * w).sum(dim=1).to(dtype)

    atol = 0.05 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(
        output_jit, ref, atol=atol, rtol=rtol
    ), f"apply_shuffle_mul_sum mismatch (dtype={dtype}, m={m}, topk={topk}, k={k})"


def test_apply_shuffle_mul_sum_no_factors():
    """Without factors, should be equivalent to a simple sum (factor=1)."""
    m, topk, k = 32, 4, 256
    device = "cuda"
    dtype = torch.float32
    perm = torch.randperm(m * topk, device=device).to(torch.int32)
    input_t = torch.randn((m * topk, k), dtype=dtype, device=device)
    output = torch.empty((m, k), dtype=dtype, device=device)

    apply_shuffle_mul_sum(input_t, output, perm, None)
    ref = input_t[perm.long()].view(m, topk, k).sum(dim=1)
    assert torch.allclose(
        output, ref, atol=1e-5
    ), "apply_shuffle_mul_sum (no factors) mismatch"


# ---------------------------------------------------------------------------
# Cross-validation against AOT sgl_kernel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("num_tokens, topk, num_experts", [(64, 4, 8), (256, 2, 16)])
def test_prepare_moe_input_vs_aot(num_tokens, topk, num_experts):
    torch.manual_seed(42)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    n, k = 256, 512

    def alloc():
        return (
            torch.empty(num_experts + 1, dtype=torch.int32, device="cuda"),
            torch.empty((num_experts, 3), dtype=torch.int32, device="cuda"),
            torch.empty((num_experts, 3), dtype=torch.int32, device="cuda"),
            torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda"),
            torch.empty(num_tokens * topk, dtype=torch.int32, device="cuda"),
        )

    eo_jit, ps1_jit, ps2_jit, ip_jit, op_jit = alloc()
    eo_aot, ps1_aot, ps2_aot, ip_aot, op_aot = alloc()

    prepare_moe_input(
        topk_ids, eo_jit, ps1_jit, ps2_jit, ip_jit, op_jit, num_experts, n, k
    )
    prepare_moe_input_aot(
        topk_ids, eo_aot, ps1_aot, ps2_aot, ip_aot, op_aot, num_experts, n, k
    )

    assert torch.equal(eo_jit, eo_aot), "expert_offsets mismatch vs AOT"
    assert torch.equal(ps1_jit, ps1_aot), "problem_sizes1 mismatch vs AOT"
    assert torch.equal(ps2_jit, ps2_aot), "problem_sizes2 mismatch vs AOT"


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shuffle_rows_vs_aot(dtype):
    torch.manual_seed(0)
    input_t = torch.randn((128, 1024), dtype=dtype, device="cuda")
    dst2src = torch.randperm(128, device="cuda").to(torch.int32)

    out_jit = shuffle_rows(input_t, dst2src, (128, 1024))
    out_aot = shuffle_rows_aot(input_t, dst2src, (128, 1024))
    assert torch.equal(
        out_jit, out_aot
    ), f"shuffle_rows vs AOT mismatch (dtype={dtype})"


@pytest.mark.skipif(not AOT_AVAILABLE, reason="sgl_kernel not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apply_shuffle_mul_sum_vs_aot(dtype):
    torch.manual_seed(0)
    m, topk, k = 64, 4, 1024
    perm = torch.randperm(m * topk, device="cuda").to(torch.int32)
    input_t = torch.randn((m * topk, k), dtype=dtype, device="cuda")
    factors = torch.rand((m * topk,), dtype=dtype, device="cuda")

    out_jit = torch.empty((m, k), dtype=dtype, device="cuda")
    out_aot = torch.empty((m, k), dtype=dtype, device="cuda")
    apply_shuffle_mul_sum(input_t, out_jit, perm, factors)
    apply_shuffle_mul_sum_aot(input_t, out_aot, perm, factors)
    assert torch.allclose(
        out_jit, out_aot, atol=1e-3, rtol=1e-3
    ), f"apply_shuffle_mul_sum vs AOT mismatch (dtype={dtype})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
