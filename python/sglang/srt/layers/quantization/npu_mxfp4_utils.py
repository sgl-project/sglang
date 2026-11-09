"""Shared MXFP4 config contracts, without NPU-only imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.linear import LinearBase

MXFP4_BLOCK_SIZE = 32


def local_input_size(layer: "LinearBase") -> int:
    """Resolve K before LinearBase calls create_weights and sets local sizes."""
    from sglang.srt.distributed.utils import get_group_rank_size
    from sglang.srt.layers.linear import RowParallelLinear

    if isinstance(layer, RowParallelLinear):
        return layer.input_size // get_group_rank_size(layer.tp_group)[1]
    return layer.input_size
