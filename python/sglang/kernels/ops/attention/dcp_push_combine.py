"""DCP decode reduce as one push + combine kernel over a CustomAllReduceV2 push plane.

Every DCP rank holds the attention partials of all heads over its KV shard and
owns ``H / N`` heads of the result: the kernel pushes each peer's heads into its
push slot over NVLink and combines its own heads by LSE weight in place of the
pack + all-to-all + combine sequence. See ``csrc/dsa/dcp_push_combine.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.communication.comm_registry import CommRegistry
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

HEAD_DIM = 128
# Push slot bytes per (row, local head): the bf16 payload and one 16 B LSE word.
SLOT_BYTES_PER_ROW = HEAD_DIM * 2 + 16


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    return load_jit(
        "dcp_push_combine",
        *args,
        cuda_files=["dsa/dcp_push_combine.cuh"],
        cuda_wrappers=[("run", f"DcpPushCombineKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


_COMMS = CommRegistry("dcp_push_combine")


def register_comm(provider: Callable[[], Communicator], *, world_size: int) -> None:
    """Register the provider of the communicator these ops launch over; it runs on
    first launch (see ``CommRegistry``).
    """
    _COMMS.register(provider, world_size=world_size)


def is_registered(world_size: int) -> bool:
    return world_size in _COMMS


@register_custom_op(mutates_args=["out"])
def _push_combine_op(
    world_size: int,
    v: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    lse_base_e: bool,
) -> None:
    _jit_module(world_size).run(_COMMS.get(world_size), v, lse, out, lse_base_e)


def dcp_push_combine(
    world_size: int,
    v: torch.Tensor,
    lse: torch.Tensor,
    *,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Combine DCP partials ``v`` [B, H, 128] bf16 with LSE ``lse`` [B, H] fp32
    across the ``world_size`` ranks of the registered communicator; returns this
    rank's heads as [B, H / world_size * 128] bf16."""
    batch, num_heads, head_dim = v.shape
    assert head_dim == HEAD_DIM and num_heads % world_size == 0
    out = v.new_empty((batch, num_heads // world_size * HEAD_DIM))
    _push_combine_op(world_size, v, lse, out, is_lse_base_on_e)
    return out
