"""Kimi-K3 attention-residual aggregation fused with an NVLS pull reduce-scatter
or a direct multicast all-gather (the collective variants of attn_res_fused_tma)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.kernels.ops.attention.attn_res import _jit_fused_tma_module, _tuning
from sglang.kernels.ops.communication.comm_registry import CommRegistry
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator

_COMMS = CommRegistry("attn_res_collective")


def register_comm(provider: Callable[[], Communicator], *, world_size: int) -> None:
    """Register the provider of the communicator these ops launch over; it runs on
    first launch (see ``CommRegistry``).
    """
    _COMMS.register(provider, world_size=world_size)


@register_custom_op(mutates_args=["out", "prefix_out"])
def _attn_res_fused_pull_rs_op(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    input_mc_ptr: int,
    max_blocks: int,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_pull_rs(
        _COMMS.get(world_size),
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,
    )


@register_custom_op(mutates_args=["bank", "out"])
def _attn_res_fused_direct_ag_op(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    output_mc_ptr: int,
    max_blocks: int,
    write_prefix: bool,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_direct_ag(
        _COMMS.get(world_size),
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
    )


def attn_res_fused_direct_ag(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    output_mc_ptr: int,
    max_blocks: int = 128,
    write_prefix: bool = False,
) -> torch.Tensor:
    """Fuse local attention-residual aggregation with direct multicast AG.

    `out` is the full symmetric [world * local_tokens, H] output. Each rank
    aggregates its local token shard and multicast-stores normalized vectors
    directly from consumer registers into that rank's slice on every peer.
    """
    _attn_res_fused_direct_ag_op(
        world_size,
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
        *_tuning(nvb, prefix_sum.shape[0]),
    )
    return out


def attn_res_fused_pull_rs(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    input_mc_ptr: int,
    max_blocks: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused NVLS pull RS + local residual + TMA attention aggregation.

    `input` is the full TP-partial o_proj tensor in multicast symmetric
    memory. Every rank reduces only its contiguous token shard, optionally
    adds its already-local residual, materializes that prefix in `prefix_out`,
    and feeds it directly to the fused attention-residual aggregation/norm.
    """
    _attn_res_fused_pull_rs_op(
        world_size,
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,
        # prefix_out, not input: the shard this rank actually aggregates.
        *_tuning(nvb, prefix_out.shape[0]),
    )
    return out, prefix_out
