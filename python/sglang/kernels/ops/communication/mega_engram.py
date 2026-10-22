"""DeepSeek-V4.1 Engram lookup for EP decode as an id push plus a row push.

``mega_engram_lookup`` sends every hash id of this rank to the rank that owns
its row, then copies the owned MXFP8 rows (FP8 e4m3 payload, E8M0 scale per 32
values) straight into the requester's wkv input, with the scales in the 128x4
swizzled layout the FlashInfer MXFP8 GEMM reads. See
``csrc/deepseek_v4/engram/mega.cuh`` for the protocol and the buffer-reuse
invariant (the GEMM that reads ``out`` must run on the same stream before the
next lookup).

The buffers are views of one symmetric workspace owned by the caller
(``sglang.srt.layers.communication.mega_engram``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

NUM_COLS = 24
ROW_BYTES = 256
SCALE_BYTES = 8
_SCATTER_THREADS = 512
_UNROLL = 4


@cache_once
def _jit_mega_engram_module(world_size: int) -> Module:
    if world_size > 64:
        raise ValueError(f"mega engram supports R <= 64, got R={world_size}")
    args = make_cpp_args(world_size, _SCATTER_THREADS, _UNROLL)
    return load_jit(
        "mega_engram",
        *args,
        cuda_files=["deepseek_v4/engram/mega.cuh"],
        cuda_wrappers=[
            ("lookup", f"MegaEngram<{args}>::lookup"),
            ("local_bytes", f"MegaEngram<{args}>::local_bytes"),
        ],
    )


def mega_engram_local_bytes(world_size: int, max_ids: int) -> int:
    """Bytes of the private, zero-filled buffer ``mega_engram_lookup`` takes as
    ``local`` for receive slots of ``max_ids`` ids."""
    return _jit_mega_engram_module(world_size).local_bytes(max_ids)


def mega_engram_lookup(
    comm: Communicator,
    hash_ids: torch.Tensor,
    weight_ptr: int,
    scale_ptr: int,
    recv: torch.Tensor,
    out: torch.Tensor,
    sf: torch.Tensor,
    owner_magic: int,
    row_start: int,
    num_rows: int,
    no_sort: bool,
    local: torch.Tensor,
) -> None:
    """One Engram lookup: push this rank's ``hash_ids`` ([T, 24] int32,
    contiguous, T may be 0) to their owners, then serve the rows the peers asked
    for into their ``out`` / ``sf``.

    ``weight_ptr`` / ``scale_ptr`` address this rank's shard ([rows, 256] fp8
    and [rows, 8] e8m0, device or registered host memory) whose first global
    row is ``row_start`` and which holds ``num_rows`` rows. A large shard is
    read in row order unless ``no_sort`` (set it for device shards).
    ``recv`` is [R, 1 + max_ids] uint64; ``local`` is this rank's private state
    (``mega_engram_local_bytes`` uint8, zeroed once). Every rank must call this
    every step, idle or not.
    """
    module = _jit_mega_engram_module(comm.world_size)
    module.lookup(
        comm,
        hash_ids,
        weight_ptr,
        scale_ptr,
        recv,
        out,
        sf,
        owner_magic,
        row_start,
        num_rows,
        no_sort,
        local,
    )


def get_owner_magic(*, world_size: int, num_rows: int, by_column: bool) -> int:
    if by_column:
        return 0
    magic = -(-(2**64) // num_rows)
    error = magic * num_rows - 2**64
    assert world_size * num_rows * error < 2**64, (
        f"no exact owner magic for {num_rows = }, {world_size = }"
    )
    return magic
