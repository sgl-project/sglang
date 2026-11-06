"""Kimi-K3's o_proj on row shards (gfx950 prefill, SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_O_PROJ).

For a batch the attention-residual bank reads on row shards (the row-sharded MoE tail),
o_proj keeps its TP-local input; the bank's FFN read exchanges it to row shards
(all-to-all, 1/4.7 of the output's bytes) and applies the full weight there instead of
reduce-scattering the TP-partial output. Eager batches only: a graph keeps the
reduce-scatter.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel


def pad_o_proj_rows(
    o_proj: RowParallelLinear, *, tp_size: int, row_shards: Callable[[int], bool]
) -> None:
    """For a batch whose attention output is reduce-scattered onto row shards,
    o_proj writes into rows padded to a multiple of TP, which the
    reduce-scatter then reads in place instead of copying into a padded buffer."""
    inner_forward = o_proj.forward

    def _row_padded_o_proj_forward(x, *args, **kwargs):
        rows = x.shape[0]
        pad = -rows % tp_size
        if pad == 0 or kwargs.get("output_tensor") is not None or not row_shards(rows):
            return inner_forward(x, *args, **kwargs)
        output = x.new_empty(rows + pad, o_proj.weight.shape[0])
        return inner_forward(x, *args, output_tensor=output[:rows], **kwargs)

    o_proj.forward = _row_padded_o_proj_forward


def row_shard_o_proj(o_proj: RowParallelLinear) -> None:
    """For a batch the bank reads on row shards, o_proj keeps its input instead
    of multiplying it (o_proj_on_row_shard applies it). The decoder layer sets
    the batch predicate (row_shards); post-load sets the gathered full_weight."""
    inner_forward = o_proj.forward
    o_proj.row_shards = None
    o_proj.full_weight = None
    o_proj.row_shard_input = None

    def _row_shard_o_proj_forward(x, *args, **kwargs):
        if (
            o_proj.row_shards is not None
            and o_proj.full_weight is not None
            and kwargs.get("output_tensor") is None
            and not torch.cuda.is_current_stream_capturing()
            and not is_in_breakable_cuda_graph()
            and o_proj.row_shards(x.shape[0])
        ):
            o_proj.row_shard_input = x
            # rows the read never touches; it takes row_shard_input instead
            return x.new_empty(1, 1).expand(x.shape[0], o_proj.weight.shape[0]), None
        return inner_forward(x, *args, **kwargs)

    o_proj.forward = _row_shard_o_proj_forward


def gather_full_o_proj(o_proj: RowParallelLinear) -> None:
    """Every rank gathers o_proj's full [H, heads x head_dim] weight from the
    TP shards (rank-major input columns), for o_proj_on_row_shard (+176 MB a layer)."""
    weight = o_proj.weight
    parallel = get_parallel()
    full_t = weight.new_empty(parallel.tp_size * weight.shape[1], weight.shape[0])
    torch.distributed.all_gather_into_tensor(
        full_t, weight.t().contiguous(), group=parallel.tp_group.device_group
    )
    o_proj.full_weight = full_t.t().contiguous()
    del full_t
    torch.cuda.empty_cache()


def take_o_proj_row_shard(
    o_proj: RowParallelLinear, num_tokens: int
) -> Optional[torch.Tensor]:
    """The attention output's complete sum on this rank's padded row shard when
    o_proj kept its input for it (row_shard_o_proj), else None."""
    x = o_proj.row_shard_input
    if x is None:
        return None
    o_proj.row_shard_input = None
    return _o_proj_on_row_shard(
        x, full_weight=o_proj.full_weight, num_tokens=num_tokens
    )


def _o_proj_on_row_shard(
    x: torch.Tensor, *, full_weight: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """All-to-all of the TP-local input (rank j sends its heads of rank i's rows
    to rank i), the heads concatenated in rank order, then the full-K GEMM."""
    from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

    group = get_parallel().tp_group
    world = group.world_size
    rows = -(-num_tokens // world)
    if x.shape[0] < rows * world:
        x = torch.nn.functional.pad(x, (0, 0, 0, rows * world - x.shape[0]))
    exchanged = torch.empty_like(x)
    torch.distributed.all_to_all_single(
        exchanged, x.contiguous(), group=group.device_group
    )
    heads = exchanged.view(world, rows, -1).transpose(0, 1).reshape(rows, -1)
    return _k3_bf16_gemm(heads, full_weight)
