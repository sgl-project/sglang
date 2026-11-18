"""Fused all-reduce and scattered short-convolution for Inkling.

The kernel reduces a per-rank hidden-channel slice, applies causal convolution,
and updates the convolution and prefix caches in one launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ar_scattered_sconv_module(
    dtype: torch.dtype,
    world_size: int,
    w: int,
    use_silu: bool,
    use_residual: bool,
) -> Module:
    args = make_cpp_args(dtype, world_size, w, use_silu, use_residual)
    return load_jit(
        "inkling_ar_scattered_sconv",
        *args,
        cuda_files=["inkling/inkling_ar_scattered_sconv.cuh"],
        cuda_wrappers=[
            ("ar_scattered_sconv", f"ArScatteredSconvKernel<{args}>::run"),
            ("ar_col_decode", f"ColDecodeKernel<{args}>::run"),
        ],
    )


def inkling_ar_scattered_sconv(
    in_buffer: torch.Tensor,
    x_scratch: torch.Tensor,
    sconv_cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    weight: torch.Tensor,
    track_rows: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    mc_in: int,
    mc_out: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    num_blocks: int = 0,
    block_size: int = 0,
    per_block_barrier: bool = False,
    track_from_cache: bool = False,
    out_local: torch.Tensor | None = None,
    norm_gamma: torch.Tensor | None = None,
    norm_residual: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    need_scratch: bool = True,
    use_stream: bool = False,
    stream_walk: int = 0,
    full_update: bool = False,
    cache_col0: int = 0,
) -> None:
    """Run the fused kernel. ``in_buffer`` is this rank's [T, H] view of the
    input symm region (partial sums already written by the producer);
    ``mc_in`` / ``mc_out`` are the multicast pointers of the input and OUT
    regions. On return the OUT region holds the gathered post-conv [T, H] on
    every rank and ``x_scratch`` holds the reduced pre-conv [T, Hc] shard.

    Tracking: empty ``track_mask`` disables it. ``track_from_cache`` (decode)
    snapshots the post-update conv window to ``track_dst`` (``track_rows`` may
    be empty); otherwise ``track_rows`` gathers pre-conv rows (extend).

    Fused add+RMSNorm tail (decode/verify): pass ``out_local`` (this rank's
    [T, H] OUT view), ``norm_gamma``/``norm_residual``/``norm_out``/``norm_eps``.
    Works under either barrier mode. The residual is updated in place;
    ``norm_out`` receives the normed hidden.

    FULL-WIDTH mode (non-scattered sconv): ``full_update=True`` with
    ``sconv_cache`` the replicated [slots, W-1, H] tensor, ``weight`` this
    rank's contiguous [Hc, W] row slice and ``cache_col0 = rank * Hc``. Conv
    still runs column-sharded; phase 3 updates/tracks ALL H cache columns on
    every rank (window rows re-ld_reduced full-width) so the replicated cache
    stays coherent. Verify (``need_scratch``) is unsupported full-width."""
    w = weight.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_buffer.dtype, world_size, w, use_silu, use_residual
    )
    do_norm = norm_gamma is not None
    if do_norm:
        assert (
            out_local is not None and norm_residual is not None and norm_out is not None
        )
    else:
        empty = in_buffer.new_empty((0,))
        out_local = norm_gamma = norm_residual = norm_out = empty
    module.ar_scattered_sconv(
        in_buffer,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        cache_indices,
        has_initial_state,
        cu,
        si,
        weight,
        track_rows,
        track_mask,
        track_dst,
        out_local,
        norm_gamma,
        norm_residual,
        norm_out,
        mc_in,
        mc_out,
        flag_ptrs_dev,
        state_ptr,
        rank,
        num_blocks,
        block_size,
        per_block_barrier,
        track_from_cache,
        norm_eps,
        need_scratch,
        use_stream,
        stream_walk,
        full_update,
        cache_col0,
    )


def inkling_ar_col_decode(
    in_buffer: torch.Tensor,
    out_local: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    weight_shard: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    mc_in: int,
    mc_out: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    *,
    activation: str | None,
    use_residual: bool,
    vecs_per_thread: int = 0,
) -> None:
    """Dedicated small-batch column decode: one block per token row, block-scoped two-round barriers, prefetch under the entry spin, conv from registers on the owner shard, inline cache update (+ decode track), and the full-row add+RMSNorm after the exit round. Decode-only (single-token sequences; every tap is cache prefix). in_buffer/out_local are this rank's views of the input/OUT symm regions."""
    w = weight_shard.shape[1]
    use_silu = activation in ("silu", "swish")
    module = _jit_ar_scattered_sconv_module(
        in_buffer.dtype, world_size, w, use_silu, use_residual
    )
    module.ar_col_decode(
        in_buffer,
        out_local,
        residual_in,
        residual_out,
        hs_out,
        norm_weight,
        norm_eps,
        sconv_cache,
        cache_indices,
        cache_mask,
        weight_shard,
        track_mask,
        track_dst,
        mc_in,
        mc_out,
        flag_ptrs_dev,
        state_ptr,
        rank,
        vecs_per_thread,
    )
