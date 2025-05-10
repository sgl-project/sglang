""" CUTLASS based Fused MoE kernels."""
from typing import Optional

import torch







FLOAT4_E2M1_MAX = 6.0 
FLOAT8_E4M3_MAX = 448.0
MAX_TOKENS_PER_EXPERT = 65536

def cutlass_moe_fp4(a: torch.Tensor, a1_gscale: torch.Tensor,
                    w1_fp4: torch.Tensor, w1_blockscale: torch.Tensor,
                    w1_alphas: torch.Tensor, a2_gscale: torch.Tensor,
                    w2_fp4: torch.Tensor, w2_blockscale: torch.Tensor,
                    w2_alphas: torch.Tensor, topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor, m: int, n: int, k: int, e: int,
                    device: torch.device):
    """
    MoE implementation for FP4 Inputs
    
    # Gemm 1
    a: Input tensor: [m, k] (half/bfloat16)
    a1_gscale: Activation scale per expert: [e]  (float32)
    w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
    w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
    (Note: `n` is the up projection output dim, `k` is the input dim in
     full precision)
    w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                   (Block size = 16 for NVFP4)
    
    # Gemm 2
    a2_gscale: Activation scale per expert: [e]
    w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
    w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
    w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3
    
    topk_weights: [m, topk] dtype: float8
    topk_ids: [m, topk] dtype: float8
    
    m, n, k: Unquantized weight shapes, dtype: int
    e: number of experts, dtype: int
    assumes that topk < k < n to satisfy - up/down projection expectations.
    """
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
    assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
    assert (w1_fp4.ndim == 3 and w2_fp4.ndim == 3 and w1_blockscale.ndim == 3
            and w2_blockscale.ndim
            == 3), ("All Weights must be of rank 3 for cutlass_moe_fp4")
    m_a, k_a = a.shape
    e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
    e_w2, k_w2, half_n_w2 = w2_fp4.shape

    assert (e_w1 == e_w2 and e_w1 == e), ("Number of experts must match",
                                          " between weights.")
    assert (k_a // 2 == half_k_w1
            and k == k_w2), ("Hidden size mismatch between a, w1 and w2")
    assert (nx2_w1 == n * 2 and half_n_w2 == n // 2), ("mismatch in "
                                                       "expected `n`")
    assert (m == m_a), "input shape mismatch"
    assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
    assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"
    assert (topk_weights.shape[0] == m and topk_ids.shape[0]
            == m), ("topk must be provided for each row of a")
    assert (m <= MAX_TOKENS_PER_EXPERT), (
        f"m must be less than MAX_TOKENS_PER_EXPERT({MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m = {m}")
    out_dtype = a.dtype
    num_topk = topk_ids.shape[1]

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,2n,k))
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    # Problem size:  (num_experts, (m,n,k))
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # problem shapes should have [m, n, k]
    # Note that problem sizes are based on logical number of elements.
    # To Do: Add @elfiegg's kernel here
    get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, e, n, k)

    tokens_per_expert = problem_sizes1[:, 0]
    rounded_tokens_per_expert = (tokens_per_expert + (128 - 1)) // 128 * 128
    blockscale_offsets = torch.zeros(e + 1, dtype=torch.int32, device=device)
    blockscale_offsets[1:] = torch.cumsum(rounded_tokens_per_expert, dim=0)

    rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
        a,
        a1_gscale,
        expert_offsets,
        blockscale_offsets,
        num_topk,
        expert_map=a_map)

    c1 = ops.cutlass_fp4_moe_mm(rep_a_fp4, w1_fp4, rep_a_blockscale,
                                w1_blockscale, w1_alphas, problem_sizes1,
                                expert_offsets[:-1], blockscale_offsets[:-1],
                                out_dtype, device)
    del rep_a_fp4, rep_a_blockscale
    # hidden size dimension is split to one halfpytho sized tensor.
    intermediate = torch.empty((m * num_topk, w1_fp4.shape[1] // 2),
                               device=device,
                               dtype=out_dtype)

    torch.ops._C.silu_and_mul(intermediate, c1)

    int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
        intermediate, a2_gscale, expert_offsets, blockscale_offsets, num_topk)
    c2 = ops.cutlass_fp4_moe_mm(int_fp4, w2_fp4, int_blockscale, w2_blockscale,
                                w2_alphas, problem_sizes2, expert_offsets[:-1],
                                blockscale_offsets[:-1], out_dtype, device)
    del int_fp4, int_blockscale
    out = (c2[c_map].view(m, num_topk, k) *
           topk_weights.view(m, num_topk, 1).half()).sum(dim=1)
    return out.to(dtype=out_dtype)