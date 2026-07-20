"""CUDA-JIT implementations of the Inkling short-convolution kernels.

Their signatures match the Triton entrypoints so model layers can select either
backend without adapting arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_causal_conv1d_module(
    w: int,
    use_silu: bool,
    use_residual: bool,
    is_decode: bool,
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(w, use_silu, use_residual, is_decode, dtype)
    return load_jit(
        "inkling_causal_conv1d",
        *args,
        cuda_files=["inkling/causal_conv1d.cuh"],
        cuda_wrappers=[("causal_conv1d", f"CausalConv1dKernel<{args}>::run")],
    )


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_mask: torch.Tensor,
    safe_idx: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    is_decode: bool = False,
) -> torch.Tensor:
    """Apply depthwise causal convolution to a packed token stream.

    Depthwise causal conv1d over a packed ``[T, D]`` token stream, with the W-1
    prefix taps gathered directly from ``sconv_cache`` (no intermediate prefix
    tensor). Metadata args (cache_mask, safe_idx, cu, si) are precomputed once
    per forward pass and reused across layers.
    """
    if activation == "swish":
        activation = "silu"

    T = x.shape[0]
    if T == 0:
        return torch.empty_like(x)

    D = x.shape[1]
    W = weight.shape[1]
    use_silu = activation in ("silu", "swish")

    # Contiguous [T, D] output (strides (D, 1)) regardless of x's layout.
    y = torch.empty(T, D, dtype=x.dtype, device=x.device)

    module = _jit_causal_conv1d_module(W, use_silu, use_residual, is_decode, x.dtype)
    module.causal_conv1d(x, sconv_cache, safe_idx, cache_mask, weight, cu, si, y)
    return y


@cache_once
def _jit_update_sconv_cache_module(w1: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(w1, dtype)
    return load_jit(
        "inkling_update_sconv_cache",
        *args,
        cuda_files=["inkling/update_sconv_cache.cuh"],
        cuda_wrappers=[("update_sconv_cache", f"UpdateSconvCacheKernel<{args}>::run")],
    )


def update_sconv_cache(
    x: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Update each sequence's convolution cache in place.

    Shift-updates each sequence's conv state to the last W-1 entries of
    ``[old_state(gated) ++ x[start:end]]``; PAD / empty lanes are untouched. Pure
    bit-exact select/copy.
    """
    W1 = sconv_cache.shape[1]
    module = _jit_update_sconv_cache_module(W1, x.dtype)
    module.update_sconv_cache(
        x, sconv_cache, cache_indices, has_initial_state, query_start_loc
    )


@cache_once
def _jit_gather_scatter_sconv_module(w1: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(w1, dtype)
    return load_jit(
        "inkling_gather_scatter_sconv",
        *args,
        cuda_files=["inkling/gather_scatter_sconv.cuh"],
        cuda_wrappers=[("gather_scatter", f"GatherScatterSconvKernel<{args}>::run")],
    )


def fused_gather_scatter_to_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    track_conv_indices: torch.Tensor,
    mask: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    """Gather selected hidden-state rows into the convolution cache.

    Scatters masked rows ``hidden_states[track_conv_indices[b, w]]`` into
    ``sconv_cache[dst_indices[b], w]`` in-place; masked-out lanes untouched.
    Bit-exact copy. (track int32, dst int64, per the model contract.)
    """
    W1 = sconv_cache.shape[1]
    module = _jit_gather_scatter_sconv_module(W1, hidden_states.dtype)
    module.gather_scatter(
        hidden_states, sconv_cache, track_conv_indices, mask, dst_indices
    )


@cache_once
def _jit_fused_decode_update_module(
    w: int, use_silu: bool, use_residual: bool, do_track: bool, dtype: torch.dtype
) -> Module:
    args = make_cpp_args(w, use_silu, use_residual, do_track, dtype)
    return load_jit(
        "inkling_fused_decode_update",
        *args,
        cuda_files=["inkling/fused_decode_update.cuh"],
        cuda_wrappers=[
            ("fused_decode_update", f"FusedDecodeUpdateKernel<{args}>::run")
        ],
    )


def fused_causal_conv1d_update_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    activation: str | None = None,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply decode convolution and update its cache in one kernel.

    Decode conv (W-1 cached taps + current token) fused with the cache shift-update
    (+ optional prefix-cache track-copy). Returns a contiguous ``[T, D]`` output.
    """
    if activation == "swish":
        activation = "silu"
    T, D = x.shape
    W = weight.shape[1]
    use_silu = activation in ("silu", "swish")
    do_track = track_mask is not None

    cm = cache_mask.reshape(-1)
    y = torch.empty(T, D, dtype=x.dtype, device=x.device)
    if do_track:
        tm = track_mask.reshape(-1)
        ti = track_indices
    else:  # dummy tensors satisfy the signature; DO_TRACK=false never reads them
        tm = torch.empty(0, dtype=torch.bool, device=x.device)
        ti = torch.empty(0, dtype=torch.int64, device=x.device)

    module = _jit_fused_decode_update_module(
        W, use_silu, use_residual, do_track, x.dtype
    )
    module.fused_decode_update(x, sconv_cache, cache_indices, cm, weight, y, tm, ti)
    return y


@cache_once
def _jit_draft_extend_sconv_module(
    w1: int, do_track: bool, dtype: torch.dtype
) -> Module:
    args = make_cpp_args(w1, do_track, dtype)
    return load_jit(
        "inkling_draft_extend_sconv",
        *args,
        cuda_files=["inkling/draft_extend_sconv.cuh"],
        cuda_wrappers=[("draft_extend", f"DraftExtendSconvKernel<{args}>::run")],
    )


def fused_draft_extend_sconv_cache(
    hidden_states: torch.Tensor,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    draft_token_num: int,
    do_tracking: bool = False,
    crossed: torch.Tensor | None = None,
    track_step: torch.Tensor | None = None,
    mamba_track_indices: torch.Tensor | None = None,
) -> None:
    """Update draft-extend convolution state in place.

    Selects each sequence's length-(W-1) conv-state window from the virtual
    ``[sconv_cache[ci] ++ hidden[b]]`` stream at ``num_accepted_tokens[b]`` (and, if
    tracking, at ``track_step[b]`` into ``mamba_track_indices[b]`` where crossed).
    Bit-exact copy.
    """
    W1 = sconv_cache.shape[1]
    module = _jit_draft_extend_sconv_module(W1, do_tracking, hidden_states.dtype)
    dev = hidden_states.device
    if do_tracking:
        cr, ts, mti = crossed, track_step, mamba_track_indices
    else:  # dummies; DO_TRACK=false never reads them
        cr = torch.empty(0, dtype=torch.bool, device=dev)
        ts = torch.empty(0, dtype=torch.int32, device=dev)
        mti = torch.empty(0, dtype=torch.int64, device=dev)
    module.draft_extend(
        hidden_states,
        sconv_cache,
        cache_indices,
        num_accepted_tokens,
        int(draft_token_num),
        cr,
        ts,
        mti,
    )
