from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api

try:
    from flash_attn.cute import flash_attn_varlen_func as _pip_flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    _pip_flash_attn_varlen_func = None
    _pip_flash_attn_import_error = _e
else:
    _pip_flash_attn_import_error = None

try:
    from sglang.kernels.ops.attention.flash_attn.cute import (
        flash_attn_varlen_func as _inkling_flash_attn_varlen_func,
    )
except Exception as _e:  # pragma: no cover
    _inkling_flash_attn_varlen_func = None
    _inkling_flash_attn_import_error = _e
else:
    _inkling_flash_attn_import_error = None


def _select_flash_attn_varlen_func(
    *,
    has_zero_length: bool,
    rel_bias: Optional[torch.Tensor],
    rel_bias_prep_cache: Optional[dict],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    sfq: Optional[torch.Tensor],
    sfk: Optional[torch.Tensor],
    sfv: Optional[torch.Tensor],
):
    """Select pip FA4 unless the call needs the in-tree implementation."""
    needs_vendored_impl = has_zero_length or any(
        value is not None
        for value in (
            rel_bias,
            rel_bias_prep_cache,
            q_descale,
            k_descale,
            v_descale,
            sfq,
            sfk,
            sfv,
        )
    )

    if needs_vendored_impl:
        if _inkling_flash_attn_varlen_func is None:  # pragma: no cover
            raise ImportError(
                "The vendored FA4 implementation is not available, but this call "
                "has a zero-length input or uses rel_bias, FP8 descales, or MXFP8 "
                "block scales."
            ) from _inkling_flash_attn_import_error
        return _inkling_flash_attn_varlen_func

    if _pip_flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "The pip flash-attn-4 package is required for standard FA4 calls."
        ) from _pip_flash_attn_import_error
    return _pip_flash_attn_varlen_func


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@debug_kernel_api
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size: Tuple[Optional[int], Optional[int]] = (-1, -1),
    learnable_sink: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    q_descale: Optional[
        torch.Tensor
    ] = None,  # legacy per-tensor FP8 descale scalar (fp8_e4m3/e5m2 KV)
    k_descale: Optional[torch.Tensor] = None,  # legacy per-tensor FP8 descale scalar
    v_descale: Optional[torch.Tensor] = None,  # legacy per-tensor FP8 descale scalar
    sfq: Optional[
        torch.Tensor
    ] = None,  # MXFP8 UE8M0 per-32-elem block scales (block-scaled QK^T)
    sfk: Optional[
        torch.Tensor
    ] = None,  # MXFP8 UE8M0 per-32-elem block scales (block-scaled QK^T)
    sfv: Optional[
        torch.Tensor
    ] = None,  # MXFP8 UE8M0 per-32-elem block scales (in-kernel V dequant)
    rel_bias: Optional[torch.Tensor] = None,
    rel_bias_prep_cache: Optional[dict] = None,
    return_softmax_lse: bool = False,
    **_: object,
):
    q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
    cu_seqlens_q, cu_seqlens_k = [
        _maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)
    ]
    seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
    page_table = _maybe_contiguous(page_table)

    if learnable_sink is None and sinks is not None:
        learnable_sink = sinks

    if window_size == (-1, -1):
        window_size = (None, None)

    # sf* = MXFP8 UE8M0 block scale factors (per-32-element), for the
    # block-scaled QK^T / V-dequant path. *_descale = the legacy per-tensor
    # FP8 descale scalars (kv_cache_dtype fp8_e4m3/fp8_e5m2). Only one group is
    # ever populated for a given call. Non-None kwargs only, so bf16/other calls
    # don't hand these to the kernel.
    sf_kwargs = {}
    if sfq is not None:
        sf_kwargs["sfq"] = sfq
    if sfk is not None:
        sf_kwargs["sfk"] = sfk
    if sfv is not None:
        sf_kwargs["sfv"] = sfv

    descale_kwargs = {}
    if q_descale is not None:
        descale_kwargs["q_descale"] = q_descale
    if k_descale is not None:
        descale_kwargs["k_descale"] = k_descale
    if v_descale is not None:
        descale_kwargs["v_descale"] = v_descale

    rel_bias_kwargs = {}
    if rel_bias is not None:
        rel_bias_kwargs["rel_bias"] = rel_bias
    if rel_bias_prep_cache is not None:
        rel_bias_kwargs["rel_bias_prep_cache"] = rel_bias_prep_cache
    flash_attn_impl = _select_flash_attn_varlen_func(
        # TODO(mmangkad): Remove this fallback once SGLang picks up
        # flash-attn-4 4.0.0b20 or newer. Version 4.0.0b19 returns the wrong
        # arity from its zero-work fast path; the vendored version preserves
        # the established empty output/LSE behavior.
        has_zero_length=q.numel() == 0 or v.numel() == 0,
        rel_bias=rel_bias,
        rel_bias_prep_cache=rel_bias_prep_cache,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
    )
    if flash_attn_impl is _pip_flash_attn_varlen_func and num_splits == 0:
        # Preserve the previous pip FA4 behavior. Its automatic SplitKV path
        # can select a configuration that is unsupported on SM90.
        num_splits = 1
    result = flash_attn_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size=window_size,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_softmax_lse,
        **sf_kwargs,
        **descale_kwargs,
        **rel_bias_kwargs,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result


@debug_kernel_api
def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: Optional[int] = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata=None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    sinks: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    rel_bias: Optional[torch.Tensor] = None,
    rel_bias_prep_cache: Optional[dict] = None,
    return_softmax_lse: bool = False,
    **_: object,
):
    if k is not None or v is not None or qv is not None:
        raise NotImplementedError("FA4 does not support updating KV cache in-place.")
    if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
        raise NotImplementedError("FA4 path does not support rotary embedding.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError(
            "FA4 path does not support non-consecutive batch indices or left padding."
        )
    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    result = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=cache_seqlens,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap != 0.0 else None,
        window_size=window_size,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        rel_bias=rel_bias,
        rel_bias_prep_cache=rel_bias_prep_cache,
        return_softmax_lse=True,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result
