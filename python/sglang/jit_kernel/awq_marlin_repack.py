from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_awq_marlin_repack_module() -> Module:
    return load_jit(
        "awq_marlin_repack",
        cuda_files=["gemm/marlin/awq_marlin_repack.cuh"],
        cuda_wrappers=[("awq_marlin_repack", "awq_marlin_repack")],
    )


def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    tile_size = 16
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // tile_size, size_n * tile_size // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    module = _jit_awq_marlin_repack_module()
    module.awq_marlin_repack(out, b_q_weight, size_k, size_n, num_bits)
    return out


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
        output[e] = awq_marlin_repack(b_q_weight[e], size_k, size_n, num_bits)
    return output
