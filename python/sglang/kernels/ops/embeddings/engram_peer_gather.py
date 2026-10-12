from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

ENGRAM_HEAD_DIM = 256
ENGRAM_NUM_COLS = 24


@cache_once
def _jit_engram_gather_module(world_size: int) -> Module:
    if ENGRAM_NUM_COLS % world_size != 0:
        raise ValueError(
            f"engram shards cut on hash columns: {ENGRAM_NUM_COLS} columns over "
            f"{world_size} shards"
        )
    args = make_cpp_args(world_size)
    return load_jit(
        "engram_gather",
        *args,
        cuda_files=["deepseek_v4/engram_gather.cuh"],
        cuda_wrappers=[("gather_bf16", f"EngramGather<{args}>::gather_bf16")],
    )


def engram_gather_bf16(
    ids: torch.Tensor,
    out: torch.Tensor,
    w_ptrs: Sequence[int],
    s_ptrs: Sequence[int],
    shard_starts: Sequence[int],
) -> None:
    """``out[m] = dequant(table[ids[m, c]] for c in columns)`` reading each column
    from its shard: ids [M, 24] int32 global rows, out [M, 24 * 256] bf16. Shard r
    (rows from ``shard_starts[r]``) owns the columns ``r * 24 / world`` onward."""
    _jit_engram_gather_module(len(w_ptrs)).gather_bf16(
        ids, out, list(w_ptrs), list(s_ptrs), list(shard_starts)
    )
