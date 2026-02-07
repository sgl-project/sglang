from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Constants matching device::marlin:: in marlin.cuh
_TILE_SIZE = 16


@cache_once
def _jit_gptq_marlin_repack_module() -> Module:
    return load_jit(
        "gptq_marlin_repack",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cuda_wrappers=[("gptq_marlin_repack", "gptq_marlin_repack")],
    )


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits

    # Allocate output tensor
    out = torch.empty(
        (size_k // _TILE_SIZE, size_n * _TILE_SIZE // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )

    module = _jit_gptq_marlin_repack_module()
    module.gptq_marlin_repack(b_q_weight, perm, out, size_k, size_n, num_bits)
    return out
