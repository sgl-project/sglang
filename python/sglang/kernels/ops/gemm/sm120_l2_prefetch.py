from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Fire-and-forget L2 warmer (see csrc/gemm/sm120_l2_prefetch.cuh): streams a
# static weight blob through L2 on a side stream during the DRAM-quiet
# compute phases of the mamba layers so the following MoE layer's shared
# path hits L2 instead of contending with the routed experts for DRAM.


@cache_once
def _jit_l2_prefetch() -> Module:
    return load_jit(
        'sm120_l2_prefetch',
        cuda_files=['gemm/sm120_l2_prefetch.cuh'],
        cuda_wrappers=[('run', 'sglang::Sm120L2PrefetchKernel::run')],
        extra_cuda_cflags=['-O3'],
    )


def l2_prefetch(flat_u8: torch.Tensor, num_blocks: int = 48) -> None:
    _jit_l2_prefetch().run(flat_u8, num_blocks)
