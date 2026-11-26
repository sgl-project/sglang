"""The pair stages of CustomAllReduceV2's ``hier_2shot_push`` (ROCm gfx1250 DPX).

Each SoC's two partitions reduce-scatter between themselves, every partition
all-reduces its half with its peers on the other SoCs (a regular 2shot_push), and
the pair all-gathers the result. These are the pair halves, over a 2-rank push
plane of their own (``csrc/distributed/dpx_pair_push.cuh``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "dpx_pair_push",
        *args,
        cuda_files=["distributed/dpx_pair_push.cuh"],
        cuda_wrappers=[("run", f"DpxPairPushKernel<{args}>::run")],
    )


@debug_kernel_api
def custom_pair_stage(
    comm: Communicator, input: torch.Tensor, *, all_gather: bool
) -> torch.Tensor:
    """One half of a 2-rank 2shot_push over a pair plane.

    Reduce-scatter takes the full tensor and returns this rank's reduced half
    (half ``comm.rank``); all-gather takes that half and returns the full
    tensor, flattened.
    """
    result = _jit_module(input.dtype).run(comm, input, all_gather)
    if isinstance(result, torch.Tensor):
        return result
    return torch.from_dlpack(result)
