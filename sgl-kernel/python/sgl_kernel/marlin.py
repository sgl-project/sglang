import torch


def gptq_marlin_repack(
    b_q_weight,
    perm,
    size_k,
    size_n,
    num_bits,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_marlin_repack(
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
