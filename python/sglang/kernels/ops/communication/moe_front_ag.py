"""K3 TP-sharded MoE front: fp32 multicast all-gather of the router + latent.

Each rank multicast-stores its ``[M, N / world]`` block into every peer's
CustomAllReduceV2 push workspace, where a Lamport spin assembles the full
``[M, N]`` rows. Needs
:func:`sglang.kernels.ops.communication.all_reduce_residual.register_comm` once
beforehand (the same registration the push all-reduce uses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.communication.all_reduce_residual import _COMM_MAP
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# Threads per block for a gathered output up to this many bytes; wider blocks
# beyond. The grid is capped at one block per push counter, so wider blocks are
# what keeps enough 16B spin loads in flight at large batches.
_NARROW_BLOCK_MAX_BYTES = 256 * 1024
_NARROW_BLOCK, _WIDE_BLOCK = 128, 512


@cache_once
def _jit_module(n_local: int, world_size: int, block: int) -> Module:
    args = make_cpp_args(n_local, world_size, block, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_moe_front_ag",
        *args,
        cuda_files=["kimi_k3/comm/moe_front_ag.cuh"],
        cuda_wrappers=[("run", f"K3FrontAllGather<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@register_custom_op(mutates_args=["out"])
def _moe_front_all_gather_op(
    world_size: int, local: torch.Tensor, out: torch.Tensor, block: int
) -> None:
    _jit_module(local.shape[1], world_size, block).run(
        _COMM_MAP[world_size], local, out
    )


def moe_front_all_gather(
    world_size: int,
    local: torch.Tensor,
    out: torch.Tensor,
    block: Optional[int] = None,
) -> torch.Tensor:
    """``out[:, r * n : (r + 1) * n] = local`` of rank ``r``, for every rank.

    ``local`` is this rank's fp32 ``[M, n]`` block, ``n % 4 == 0``, 16B-aligned
    rows that may be strided (a column slice of a wider GEMM output);
    ``out`` is ``[M, world_size * n]`` fp32 (output-only). ``M * n * 4`` bytes
    must fit one push slot. ``block`` (threads per block) defaults by size."""
    if block is None:
        narrow = out.numel() * out.element_size() <= _NARROW_BLOCK_MAX_BYTES
        block = _NARROW_BLOCK if narrow else _WIDE_BLOCK
    _moe_front_all_gather_op(world_size, local, out, block)
    return out
