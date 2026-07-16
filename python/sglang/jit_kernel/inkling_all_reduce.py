"""CUDA-JIT all-reduce kernels for Inkling symmetric-memory buffers.

The producer writes its local shard into the symmetric buffer, and the reduced
result remains there so callers do not need staging or copy-out kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import empty_sentinel
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_inkling_all_reduce_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, world_size)
    return load_jit(
        "inkling_all_reduce",
        *args,
        cuda_files=["inkling/inkling_all_reduce.cuh"],
        cuda_wrappers=[
            ("two_shot_all_reduce", f"inkling_two_shot_all_reduce<{args}>"),
            ("two_shot_all_reduce_fused", f"inkling_two_shot_all_reduce_fused<{args}>"),
            ("multimem_one_shot_fused", f"inkling_multimem_one_shot_fused<{args}>"),
            ("multimem_full_oneshot", f"inkling_multimem_full_oneshot<{args}>"),
            ("multimem_push_oneshot", f"inkling_multimem_push_oneshot<{args}>"),
        ],
    )


# Barrier resources for the fused kernels:
#   * flags: a DEDICATED symmetric uint32 buffer, zero-initialized once at
#     setup: `world_size` single-leader slots (one per peer), then
#     world_size * MAX_BARRIER_BLOCKS per-(writer, block) slots for the
#     per-block barrier (v5's multi-block flavor).
#   * state: a device-LOCAL uint32 buffer: the 5 words
#     [arrival0, arrival1, release0, release1, xepoch] padded to 8, then
#     MAX_BARRIER_BLOCKS per-block epochs; persists across calls and advances
#     under CUDA-graph replay.
# Keep these sizes aligned with the CUDA barrier implementation.
MAX_BARRIER_BLOCKS = 256
STATE_SIZE = 8 + MAX_BARRIER_BLOCKS


def flags_numel(world_size: int) -> int:
    return world_size * (1 + MAX_BARRIER_BLOCKS)


# Tuned (kernel, num_blocks, block_size) per reduction row count. Kernels:
# "v5"=push one-shot with per-block barriers (single barrier, out-of-place;
# owns the latency band), "mm"=torch multimem, "v2"=two-shot explicit,
# "v3"=two-shot multimem (single-leader barriers), "v3b"=v3 with per-block
# barriers, and "v4"=full one-shot. nb/bs are 0 for "mm". Tables are keyed
# by world size; TP4 is the fallback.
_AR_TUNED_TP4 = {
    1: ("v5", 1, 1024),
    2: ("v5", 1, 1024),
    3: ("v5", 8, 512),
    4: ("v5", 8, 512),
    6: ("v5", 8, 1024),
    8: ("v5", 8, 1024),
    12: ("v5", 8, 512),
    16: ("v5", 8, 512),
    24: ("v5", 8, 1024),
    32: ("v5", 8, 1024),
    48: ("v5", 48, 1024),
    64: ("v5", 48, 1024),
    96: ("v5", 64, 1024),
    128: ("mm", 0, 0),
    192: ("mm", 0, 0),
    256: ("v3b", 64, 1024),
    384: ("v3b", 32, 1024),
    512: ("v3b", 32, 1024),
    768: ("v3b", 48, 512),
    1024: ("v3b", 32, 1024),
    1536: ("v3", 64, 512),
    2048: ("v3", 64, 512),
    3072: ("v3", 96, 512),
    4096: ("v3", 96, 512),
    6144: ("v3", 64, 512),
    8192: ("v3", 32, 1024),
    12288: ("v3", 96, 512),
    16384: ("v3", 96, 512),
}
# TP8 uses full one-shot for the smallest shapes, multimem through the
# medium-sized range, and two-shot multimem for larger reductions.
_AR_TUNED_TP8 = {
    1: ("v4", 1, 1024),
    2: ("v4", 1, 1024),
    3: ("mm", 0, 0),
    4: ("mm", 0, 0),
    6: ("mm", 0, 0),
    8: ("mm", 0, 0),
    12: ("mm", 0, 0),
    16: ("mm", 0, 0),
    24: ("mm", 0, 0),
    32: ("mm", 0, 0),
    48: ("mm", 0, 0),
    64: ("mm", 0, 0),
    96: ("mm", 0, 0),
    128: ("mm", 0, 0),
    192: ("mm", 0, 0),
    256: ("mm", 0, 0),
    384: ("mm", 0, 0),
    512: ("mm", 0, 0),
    768: ("mm", 0, 0),
    1024: ("v3", 32, 512),
    1536: ("v3", 16, 1024),
    2048: ("v3", 32, 512),
    3072: ("v3", 48, 512),
    4096: ("v3", 48, 512),
    6144: ("v3", 64, 512),
    8192: ("v3", 96, 256),
    12288: ("v3", 64, 512),
    16384: ("v3", 64, 512),
}
_AR_TUNED = {4: _AR_TUNED_TP4, 8: _AR_TUNED_TP8}
_AR_TUNED_TOKENS = sorted(_AR_TUNED_TP4)  # same token grid for every table
assert all(
    set(t) == set(_AR_TUNED_TP4) for t in _AR_TUNED.values()
), "all tuned tables must share the same token grid"


def select_ar_config(num_tokens: int, world_size: int = 4):
    """Return (kernel, num_blocks, block_size) for a ``[num_tokens, hidden]``
    reduction, from the autotuned table for ``world_size`` (round up to the
    nearest tested shape). Untuned world sizes fall back to the TP4 table.
    ``kernel`` is one of "v5"/"v4"/"mm"/"v2"/"v3"/"v3b"."""
    table = _AR_TUNED.get(world_size, _AR_TUNED_TP4)
    for t in _AR_TUNED_TOKENS:
        if num_tokens <= t:
            return table[t]
    return table[_AR_TUNED_TOKENS[-1]]


def compile_inkling_all_reduce(dtype: torch.dtype, world_size: int) -> None:
    """Warm the JIT module for (dtype, world_size) so the first call is cheap."""
    _jit_inkling_all_reduce_module(dtype, world_size)


def inkling_two_shot_all_reduce(
    buffer: torch.Tensor,
    peer_ptrs_dev: int,
    rank: int,
    world_size: int,
    num_items: int,
) -> None:
    """Two-shot all-reduce in place over ``num_items`` elements of the symm buffer.

    Args:
        buffer: this rank's symm buffer (1D, contiguous, bf16), sliced to
            ``num_items``; used for device/dtype validation. The producer must
            have already written this rank's shard into it.
        peer_ptrs_dev: ``hdl.buffer_ptrs_dev`` -- device address of the array of
            ``world_size`` peer buffer base pointers.
        rank: this rank within the TP group.
        world_size: TP world size (compile-time template arg).
        num_items: number of elements to reduce (multiple of 8 for bf16).

    The caller is responsible for ``hdl.barrier()`` before (producers done) and
    after (result visible) this call.
    """
    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.two_shot_all_reduce(buffer, peer_ptrs_dev, rank, num_items)


def inkling_two_shot_all_reduce_fused(
    buffer: torch.Tensor,
    data_ptrs_dev: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    shared: torch.Tensor | None = None,
) -> None:
    """Single-launch two-shot all-reduce with an in-kernel grid-level barrier.

    Args:
        buffer: this rank's symm data buffer (bf16), sliced to ``num_items``.
        data_ptrs_dev: ``hdl.buffer_ptrs_dev`` for the data buffer.
        flag_ptrs_dev: ``buffer_ptrs_dev`` of a DEDICATED symm ``uint32[world_size]``
            flags buffer, zero-initialized once at setup.
        state_ptr: ``data_ptr()`` of a device-local ``uint32[STATE_SIZE]`` barrier
            state buffer (persists across calls; advances under graph replay).
        rank, world_size: TP coordinates (world_size is a template arg).
        num_items: elements to reduce (multiple of 8 for bf16).

    No external barrier needed -- the kernel fences both sides itself.
    """
    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.two_shot_all_reduce_fused(
        buffer,
        data_ptrs_dev,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        shared if shared is not None else empty_sentinel(buffer.device, buffer.dtype)
    )


def inkling_multimem_one_shot_fused(
    buffer: torch.Tensor,
    multicast_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    shared: torch.Tensor | None = None,
) -> None:
    """Single-launch multimem one-shot all-reduce (NVLink multicast ld_reduce/st).

    Matches torch multimem for tiny, latency-bound (decode) messages, in a kernel
    we own so norm/sconv can fuse at the epilogue seam.

    Args:
        buffer: this rank's symm data buffer (bf16), sliced to ``num_items``.
        multicast_ptr: ``hdl.multicast_ptr`` for the data buffer (must be != 0).
        flag_ptrs_dev, state_ptr: dedicated barrier flags + local state buffer
            (same as the fused two-shot).
        rank, world_size, num_items: as above.
        per_block_barrier: use per-block peer handshakes for both barriers (no
            grid funnel; capped at MAX_BARRIER_BLOCKS blocks).
    """
    module = _jit_inkling_all_reduce_module(buffer.dtype, world_size)
    module.multimem_one_shot_fused(
        buffer,
        multicast_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        int(per_block_barrier),
        shared if shared is not None else empty_sentinel(buffer.device, buffer.dtype)
    )


def inkling_multimem_full_oneshot(
    in_buffer: torch.Tensor,
    out_buffer: torch.Tensor,
    multicast_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    shared: torch.Tensor | None = None,
) -> None:
    """Full one-shot all-reduce with a SINGLE (entry-only) barrier.

    Every rank ld_reduces the entire range (multicast hardware sum) into its
    local ``out_buffer`` -- no broadcast, no exit barrier. Fastest for tiny
    latency-bound messages, but the caller MUST double-buffer ``in_buffer`` (its
    reuse is not fenced by this kernel; the next AR's entry barrier orders it).

    Args:
        in_buffer: this rank's symm data buffer (bf16), sliced to ``num_items``.
        out_buffer: local output buffer (bf16, >= num_items); receives the sum.
        multicast_ptr: ``hdl.multicast_ptr`` of the in_buffer.
        flag_ptrs_dev, state_ptr: barrier flags + local state (as above).
        rank, world_size, num_items: as above.
    """
    module = _jit_inkling_all_reduce_module(in_buffer.dtype, world_size)
    module.multimem_full_oneshot(
        in_buffer,
        out_buffer,
        multicast_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        shared if shared is not None else empty_sentinel(in_buffer.device, in_buffer.dtype)
    )


def inkling_multimem_push_oneshot(
    in_buffer: torch.Tensor,
    out_buffer: torch.Tensor,
    mc_stage_ptr: int,
    local_stage_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    num_items: int,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    shared: torch.Tensor | None = None,
) -> None:
    """One-shot PUSH all-reduce (v5) with a SINGLE mid barrier.

    Each rank multicast-stores its full input into its per-rank slot of the
    symmetric staging area (slot ``r`` at elem offset ``r * num_items``), the
    barrier waits for all pushes to land, then each rank reduces the
    ``world_size`` staged shards locally (fp32 accum) into ``out_buffer``.
    Drops one barrier round trip vs the two-shot kernels -- wins the
    latency-bound small/medium band -- and each rank holds the full row at the
    epilogue seam (norm-fusion base, like v4 but scaling past 2 rows).

    Args:
        in_buffer: this rank's LOCAL input (any contiguous 16B-aligned bf16
            tensor -- need not be a symm buffer; it is only read locally).
        out_buffer: local output buffer (bf16, >= num_items); receives the sum.
        mc_stage_ptr: multicast address of the staging area (>= world_size *
            num_items elems). The caller MUST double-buffer the staging area
            (A/B rotation; the next AR's barrier orders the reuse, like v4).
        local_stage_ptr: this GPU's local address of the same staging area.
        flag_ptrs_dev, state_ptr: barrier flags + local state (as above).
        rank, world_size, num_items: as above.
        per_block_barrier: use the per-block peer handshake (no grid funnel;
            capped at MAX_BARRIER_BLOCKS blocks) instead of the single-leader
            grid barrier -- the multi-block latency winner.
    """
    module = _jit_inkling_all_reduce_module(in_buffer.dtype, world_size)
    module.multimem_push_oneshot(
        in_buffer,
        out_buffer,
        mc_stage_ptr,
        local_stage_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_items,
        num_blocks,
        block_size,
        int(per_block_barrier),
        shared if shared is not None else empty_sentinel(in_buffer.device, in_buffer.dtype)
    )
