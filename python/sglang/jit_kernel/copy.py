from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_copy_to_gpu_no_ce_module(n: int) -> Module:
    args = make_cpp_args(n)
    return load_jit(
        "copy_to_gpu_no_ce",
        *args,
        cuda_files=["elementwise/copy.cuh"],
        cuda_wrappers=[("copy_to_gpu_no_ce", f"copy_to_gpu_no_ce<{args}>")],
    )


def copy_to_gpu_no_ce(input: torch.Tensor, output: torch.Tensor) -> None:
    """Copy a small int32 CPU tensor to a pre-allocated GPU tensor without
    using the copy engine (CE).  The data is packed into kernel-launch
    arguments so the transfer bypasses DMA and goes through the kernel path.

    Args:
        input:  1-D int32 CPU tensor
        output: 1-D int32 CUDA tensor of the same length, pre-allocated
    """
    n = int(input.numel())
    module = _jit_copy_to_gpu_no_ce_module(n)
    module.copy_to_gpu_no_ce(input, output)
