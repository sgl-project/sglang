import torch


def gptq_marlin_repack(
    b_q_weight,
    perm,
    size_k,
    size_n,
    num_bits,
):
    return torch.ops.sgl_kernel.gptq_marlin_repack.default(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
    )


def awq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)

def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops.sgl_kernel.gptq_marlin_repack(b_q_weight[e], perm[e],
                                                    size_k, size_n, num_bits)
    return output

def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = torch.ops.sgl_kernel.awq_marlin_repack(
            b_q_weight[e], size_k, size_n, num_bits
        )
    return output

def gptq_marlin_gemm(a: torch.Tensor,
                     c: Optional[torch.Tensor],
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     global_scale: Optional[torch.Tensor],
                     b_zeros: Optional[torch.Tensor],
                     g_idx: Optional[torch.Tensor],
                     perm: Optional[torch.Tensor],
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool = True,
                     use_atomic_add: bool = False,
                     use_fp32_reduce: bool = False,
                     is_zp_float: bool = False) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_marlin_gemm(a, c, b_q_weight, b_scales,
                                                 global_scale, b_zeros, g_idx, perm,
                                                 workspace, b_q_type.id, size_m,
                                                 size_n, size_k, is_k_full,
                                                 use_atomic_add, use_fp32_reduce,
                                                 is_zp_float)
