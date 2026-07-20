"""Fused target-verify attention prologue: {k/v sconv + save_windows + qk-norm
+ KV-cache store} in one kernel (csrc/tml/inkling_attn_prologue_fused.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    empty_sentinel,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_attn_prologue_module(
    dtype: torch.dtype,
    w: int,
    use_silu: bool,
    use_residual: bool,
    use_mxfp8: bool,
) -> Module:
    args = make_cpp_args(
        dtype, w, use_silu, use_residual, use_mxfp8, is_arch_support_pdl()
    )
    return load_jit(
        "inkling_attn_prologue_fused",
        *args,
        cuda_files=["inkling/inkling_attn_prologue_fused.cuh"],
        cuda_wrappers=[
            ("attn_prologue", f"AttnPrologueKernel<{args}>::run"),
            ("attn_prologue_decode", f"AttnPrologueDecodeKernel<{args}>::run"),
            ("attn_prologue_extend", f"AttnPrologueExtendKernel<{args}>::run"),
        ],
    )


def compile_inkling_attn_prologue(
    dtype: torch.dtype,
    w: int,
    use_silu: bool,
    use_residual: bool,
    use_mxfp8: bool = False,
) -> None:
    _jit_attn_prologue_module(dtype, w, use_silu, use_residual, use_mxfp8)


def inkling_attn_prologue_verify(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_inter: torch.Tensor,
    v_inter: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    draft_token_num: int,
    activation: str | None = None,
    use_residual: bool = True,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
    page_size: int = 128,
    log_scaling_tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns fresh contiguous (q_normed, k_normed, v_conv) [T, dq/dkv];
    KV rows are also scattered into k_buf/v_buf at ``loc`` (the attention call
    should pass save_kv_cache=False)."""
    t = qkvr.shape[0]
    if mxfp8_quant:
        if dq % 128 != 0 or dkv % 128 != 0:
            raise ValueError("MXFP8 fused prologue requires head_dim-aligned Q/K/V.")
        if sfk is None or sfv is None:
            raise ValueError("MXFP8 fused prologue requires K/V scale buffers.")
        sf_shape = (k_buf.shape[0] // page_size, dkv // 128, 32, page_size // 32, 4)
        if sfk.shape != sf_shape or sfv.shape != sf_shape:
            raise ValueError(
                "MXFP8 fused prologue requires interleaved K/V scale buffers "
                f"with shape {sf_shape}, got {tuple(sfk.shape)} and {tuple(sfv.shape)}."
            )
        if not sfk.is_contiguous() or not sfv.is_contiguous():
            raise ValueError(
                "MXFP8 fused prologue requires contiguous interleaved SFK/SFV."
            )
        q_out = torch.empty(t, dq, dtype=torch.float8_e4m3fn, device=qkvr.device)
        sfq_u8 = torch.empty(
            (t, dq // 128, 128 // 32), dtype=torch.uint8, device=qkvr.device
        )
        sfk_u8 = sfk.view(torch.uint8)
        sfv_u8 = sfv.view(torch.uint8)
    else:
        q_out = torch.empty(t, dq, dtype=qkvr.dtype, device=qkvr.device)
        sfq_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfk_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfv_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
    k_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    v_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    w = k_weight.shape[1]
    module = _jit_attn_prologue_module(
        qkvr.dtype, w, use_silu, use_residual, mxfp8_quant
    )
    hkv = dkv // 128
    module.attn_prologue(
        qkvr,
        k_cache,
        v_cache,
        cache_indices.to(torch.int32),
        cache_mask,
        k_weight,
        v_weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        float(eps),
        q_out,
        k_out,
        v_out,
        loc,
        k_buf.view(-1, hkv * 128),
        v_buf.view(-1, hkv * 128),
        sfq_u8,
        sfk_u8,
        sfv_u8,
        int(q_off),
        int(k_off),
        int(v_off),
        int(draft_token_num),
        int(do_store),
        int(page_size),
        (
            log_scaling_tau.reshape(-1).float()
            if log_scaling_tau is not None
            else empty_sentinel(qkvr.device, torch.float32)
        ),
    )
    q_scale = sfq_u8.view(torch.float8_e8m0fnu) if mxfp8_quant else None
    return q_out, k_out, v_out, q_scale


def inkling_attn_prologue_extend(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    track_rows: torch.Tensor,
    track_mask: torch.Tensor,
    track_dst: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    activation: str | None = None,
    use_residual: bool = True,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
    page_size: int = 128,
    do_cache_update: bool = True,
    log_scaling_tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Extend (prefill) analog of ``inkling_attn_prologue_verify``: varlen
    sequences via ``cu``/``si``, no window save; instead a tiny trailing
    kernel does the k/v conv-cache update at sequence ends (+ the extend
    prefix-cache track when ``track_mask`` is non-empty -- pass empty tensors
    to disable). Returns fresh contiguous (q_normed, k_normed, v_conv) and
    scatters KV rows into k_buf/v_buf at ``loc`` when ``do_store`` (the
    attention call should then pass save_kv_cache=False)."""
    t = qkvr.shape[0]
    if mxfp8_quant:
        if dq % 128 != 0 or dkv % 128 != 0:
            raise ValueError("MXFP8 fused prologue requires head_dim-aligned Q/K/V.")
        if sfk is None or sfv is None:
            raise ValueError("MXFP8 fused prologue requires K/V scale buffers.")
        sf_shape = (k_buf.shape[0] // page_size, dkv // 128, 32, page_size // 32, 4)
        if sfk.shape != sf_shape or sfv.shape != sf_shape:
            raise ValueError(
                "MXFP8 fused prologue requires interleaved K/V scale buffers "
                f"with shape {sf_shape}, got {tuple(sfk.shape)} and {tuple(sfv.shape)}."
            )
        if not sfk.is_contiguous() or not sfv.is_contiguous():
            raise ValueError(
                "MXFP8 fused prologue requires contiguous interleaved SFK/SFV."
            )
        q_out = torch.empty(t, dq, dtype=torch.float8_e4m3fn, device=qkvr.device)
        sfq_u8 = torch.empty(
            (t, dq // 128, 128 // 32), dtype=torch.uint8, device=qkvr.device
        )
        sfk_u8 = sfk.view(torch.uint8)
        sfv_u8 = sfv.view(torch.uint8)
    else:
        q_out = torch.empty(t, dq, dtype=qkvr.dtype, device=qkvr.device)
        sfq_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfk_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfv_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
    k_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    v_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    w = k_weight.shape[1]
    module = _jit_attn_prologue_module(
        qkvr.dtype, w, use_silu, use_residual, mxfp8_quant
    )
    hkv = dkv // 128
    module.attn_prologue_extend(
        qkvr,
        k_cache,
        v_cache,
        cache_indices.to(torch.int32),
        cache_mask,
        has_initial_state,
        cu,
        si,
        k_weight,
        v_weight,
        track_rows,
        track_mask,
        track_dst,
        q_gamma,
        k_gamma,
        float(eps),
        q_out,
        k_out,
        v_out,
        loc,
        k_buf.view(-1, hkv * 128),
        v_buf.view(-1, hkv * 128),
        sfq_u8,
        sfk_u8,
        sfv_u8,
        int(q_off),
        int(k_off),
        int(v_off),
        int(do_store),
        int(page_size),
        int(do_cache_update),
        (
            log_scaling_tau.reshape(-1).float()
            if log_scaling_tau is not None
            else empty_sentinel(qkvr.device, torch.float32)
        ),
    )
    q_scale = sfq_u8.view(torch.float8_e8m0fnu) if mxfp8_quant else None
    return q_out, k_out, v_out, q_scale


def inkling_attn_prologue_decode(
    qkvr: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_indices: torch.Tensor,
    cache_mask: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    eps: float,
    loc: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_off: int,
    k_off: int,
    v_off: int,
    dq: int,
    dkv: int,
    activation: str | None = None,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    do_store: bool = True,
    mxfp8_quant: bool = False,
    sfk: torch.Tensor | None = None,
    sfv: torch.Tensor | None = None,
    page_size: int = 128,
    log_scaling_tau: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Decode {k/v decode-conv + conv-cache shift-update (+track) + qk-norm
    (+ KV store)} in one kernel. Returns fresh (q_normed, k_normed, v_conv).
    The k/v conv caches are shift-updated in place (fused_decode_update
    semantics). With ``do_store`` the KV rows are scattered into k_buf/v_buf at
    ``loc``; MXFP8 mode also quantizes Q and writes interleaved K/V scales."""
    t = qkvr.shape[0]
    if mxfp8_quant:
        if dq % 128 != 0 or dkv % 128 != 0:
            raise ValueError(
                "MXFP8 fused decode prologue requires head_dim-aligned Q/K/V."
            )
        if sfk is None or sfv is None:
            raise ValueError("MXFP8 fused decode prologue requires K/V scale buffers.")
        sf_shape = (k_buf.shape[0] // page_size, dkv // 128, 32, page_size // 32, 4)
        if sfk.shape != sf_shape or sfv.shape != sf_shape:
            raise ValueError(
                "MXFP8 fused decode prologue requires interleaved K/V scale buffers "
                f"with shape {sf_shape}, got {tuple(sfk.shape)} and {tuple(sfv.shape)}."
            )
        if not sfk.is_contiguous() or not sfv.is_contiguous():
            raise ValueError(
                "MXFP8 fused decode prologue requires contiguous interleaved SFK/SFV."
            )
        q_out = torch.empty(t, dq, dtype=torch.float8_e4m3fn, device=qkvr.device)
        sfq_u8 = torch.empty(
            (t, dq // 128, 128 // 32), dtype=torch.uint8, device=qkvr.device
        )
        sfk_u8 = sfk.view(torch.uint8)
        sfv_u8 = sfv.view(torch.uint8)
    else:
        q_out = torch.empty(t, dq, dtype=qkvr.dtype, device=qkvr.device)
        sfq_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfk_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
        sfv_u8 = torch.empty(0, dtype=torch.uint8, device=qkvr.device)
    k_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    v_out = torch.empty(t, dkv, dtype=qkvr.dtype, device=qkvr.device)
    if activation == "swish":
        activation = "silu"
    use_silu = activation in ("silu", "swish")
    w = k_weight.shape[1]
    do_track = track_mask is not None
    if do_track:
        tm, ti = track_mask.reshape(-1), track_indices
    else:
        tm = torch.empty(0, dtype=torch.bool, device=qkvr.device)
        ti = torch.empty(0, dtype=torch.int64, device=qkvr.device)
    hkv = dkv // 128
    module = _jit_attn_prologue_module(
        qkvr.dtype, w, use_silu, use_residual, mxfp8_quant
    )
    module.attn_prologue_decode(
        qkvr,
        k_cache,
        v_cache,
        cache_indices.to(torch.int32),
        cache_mask,
        k_weight,
        v_weight,
        tm,
        ti,
        q_gamma,
        k_gamma,
        float(eps),
        q_out,
        k_out,
        v_out,
        loc,
        k_buf.view(-1, hkv * 128),
        v_buf.view(-1, hkv * 128),
        sfq_u8,
        sfk_u8,
        sfv_u8,
        int(q_off),
        int(k_off),
        int(v_off),
        int(do_track),
        int(do_store),
        int(page_size),
        (
            log_scaling_tau.reshape(-1).float()
            if log_scaling_tau is not None
            else empty_sentinel(qkvr.device, torch.float32)
        ),
    )
    q_scale = sfq_u8.view(torch.float8_e8m0fnu) if mxfp8_quant else None
    return q_out, k_out, v_out, q_scale
