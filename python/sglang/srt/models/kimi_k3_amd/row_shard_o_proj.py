"""Kimi-K3's o_proj output rows padded for the row-sharded read's reduce-scatter (gfx950)."""

from __future__ import annotations

from typing import Callable

from sglang.srt.layers.linear import RowParallelLinear


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
