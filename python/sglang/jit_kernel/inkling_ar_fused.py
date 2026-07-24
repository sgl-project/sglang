"""Fused all-reduce, decode short-convolution, and RMSNorm for Inkling.

The small-batch decode kernel processes one token per block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, empty_sentinel, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ar_fused_module(
    dtype: torch.dtype,
    world_size: int,
    w: int,
    use_silu: bool,
    use_residual: bool,
    do_track: bool,
) -> Module:
    args = make_cpp_args(dtype, world_size, w, use_silu, use_residual, do_track)
    return load_jit(
        "inkling_ar_fused_decode",
        *args,
        cuda_files=["inkling/inkling_ar_fused_decode.cuh"],
        cuda_wrappers=[
            ("ar_sconv_norm", f"ArSconvNormKernel<{args}>::run"),
            ("ar_sconv_norm_verify", f"ArSconvNormVerifyKernel<{args}>::run"),
        ],
    )


# Tuned vectors per thread by decode row count; round up to the next entry.
_FUSED_VPT_TUNED = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1, 32: 1, 64: 1, 96: 1}
_FUSED_VPT_TOKENS = sorted(_FUSED_VPT_TUNED)


def select_fused_vpt(num_tokens: int) -> int:
    for t in _FUSED_VPT_TOKENS:
        if num_tokens <= t:
            return _FUSED_VPT_TUNED[t]
    return _FUSED_VPT_TUNED[_FUSED_VPT_TOKENS[-1]]


def compile_inkling_ar_sconv_norm(
    dtype: torch.dtype,
    world_size: int,
    w: int,
    use_silu: bool,
    use_residual: bool,
    do_track: bool,
) -> None:
    """Warm the JIT module so the first fused call is cheap."""
    _jit_ar_fused_module(dtype, world_size, w, use_silu, use_residual, do_track)


def inkling_ar_sconv_norm(
    in_partial: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    conv_weight: torch.Tensor,
    mc_stage_ptr: int,
    local_stage_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    activation: str | None = None,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    enable_pdl: bool = True,
    vecs_per_thread: int = 0,
    shared: torch.Tensor | None = None,
) -> None:
    """Fused AR + decode sconv + add-RMSNorm over ``[T, D]`` decode rows.

    Args:
        in_partial: this rank's LOCAL partial sums (``[T, D]`` bf16, contiguous
            rows, 16B-aligned) -- e.g. the MoE combine output with
            ``reduce=False``. Read locally only (no stage-in copy).
        shared: optional LOCAL ``[T, D]`` shared-expert partials, folded into
            the pushed value in registers (fp32 add, one bf16 round --
            torch.add numerics), replacing the separate ``routed + shared``
            add kernel at zero extra traffic. All ranks must agree on passing it.
        residual_in / residual_out: the residual stream before/after the fused
            add (may alias); ``hs_out``: the normed output.
        norm_weight, eps: RMSNorm gamma (``[D]`` bf16) and epsilon.
        sconv_cache..conv_weight, track_*: exactly the tensors
            ``fused_causal_conv1d_update_decode`` takes; the conv state is
            shift-updated in place, identically to the unfused kernel.
        mc_stage_ptr / local_stage_ptr: multicast + local address of the v5
            staging rotation slot (>= world_size*T*D elems; caller rotates A/B,
            same reuse-distance rule as v5).
        flag_ptrs_dev / state_ptr / rank / world_size: barrier resources
            (shared with the other fused AR kernels).
    """
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    do_track = track_mask is not None
    w = conv_weight.shape[1]
    if do_track:
        tm = track_mask.reshape(-1)
        ti = track_indices
    else:  # dummies; DO_TRACK=false never reads them
        tm = torch.empty(0, dtype=torch.bool, device=in_partial.device)
        ti = torch.empty(0, dtype=torch.int64, device=in_partial.device)
    module = _jit_ar_fused_module(
        in_partial.dtype, world_size, w, use_silu, use_residual, do_track
    )
    if vecs_per_thread <= 0:
        vecs_per_thread = select_fused_vpt(in_partial.shape[0])
    sh = (
        shared
        if shared is not None
        else empty_sentinel(in_partial.device, in_partial.dtype)
    )
    module.ar_sconv_norm(
        in_partial,
        residual_in,
        residual_out,
        hs_out,
        norm_weight,
        float(eps),
        sconv_cache,
        cache_indices,
        cache_mask.reshape(-1),
        conv_weight,
        tm,
        ti,
        mc_stage_ptr,
        local_stage_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        int(enable_pdl),
        int(vecs_per_thread),
        sh,
    )


def inkling_ar_sconv_norm_verify(
    in_partial: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    hs_out: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    sconv_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    conv_weight: torch.Tensor,
    inter_out: torch.Tensor,
    draft_token_num: int,
    mc_stage_ptr: int,
    local_stage_ptr: int,
    flag_ptrs_dev: int,
    state_ptr: int,
    rank: int,
    world_size: int,
    activation: str | None = None,
    use_residual: bool = True,
    enable_pdl: bool = True,
    shared: torch.Tensor | None = None,
) -> None:
    """Target-verify fused {AR -> causal_conv1d -> save_intermediate_conv_windows
    -> add+RMSNorm} over ``[B*draft_token_num, D]`` rows.

    ``cache_indices``/``cache_mask`` are per-SEQUENCE (``[B]``); the working
    conv cache is read-only (the per-position windows go to ``inter_out``,
    exactly like ``save_intermediate_conv_windows``). Cross-token conv taps are
    re-reduced from the v5 staging slot, so the same rotation rules as
    ``inkling_ar_sconv_norm`` apply. ``shared``: optional LOCAL ``[T, D]``
    shared-expert partials folded into the push (torch.add numerics).
    """
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    w = conv_weight.shape[1]
    # do_track slot in the module key is unused by the verify kernel.
    module = _jit_ar_fused_module(
        in_partial.dtype, world_size, w, use_silu, use_residual, False
    )
    sh = (
        shared
        if shared is not None
        else empty_sentinel(in_partial.device, in_partial.dtype)
    )
    module.ar_sconv_norm_verify(
        in_partial,
        residual_in,
        residual_out,
        hs_out,
        norm_weight,
        float(eps),
        sconv_cache,
        cache_indices.to(torch.int32),
        cache_mask,
        conv_weight,
        inter_out,
        int(draft_token_num),
        mc_stage_ptr,
        local_stage_ptr,
        flag_ptrs_dev,
        state_ptr,
        rank,
        int(enable_pdl),
        sh,
    )
