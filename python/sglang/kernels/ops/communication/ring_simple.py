"""CustomAllReduceV2's ring all-reduce, ``ring_simple`` (ROCm gfx1250).

Each chunk goes to the next rank only, through a fixed-size FIFO in its memory,
in ``2 * (world_size - 1)`` hops; one ring per block. It runs over the
communicator's ring plane (``csrc/distributed/ring_simple.cuh``).
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
def get_ring_simple_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "ring_simple",
        *args,
        cuda_files=["distributed/ring_simple.cuh"],
        cuda_wrappers=[("run", f"RingSimpleKernel<{args}>::run")],
    )


@debug_kernel_api
def custom_ring_simple(comm: Communicator, input: torch.Tensor) -> torch.Tensor:
    """All-reduce ``input`` over ``comm``'s ring plane; returns a new tensor."""
    result = get_ring_simple_module(input.dtype, comm.world_size).run(comm, input)
    if isinstance(result, torch.Tensor):
        return result
    return torch.from_dlpack(result)
