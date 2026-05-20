from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    _flash_attn_import_error = _e
else:
    _flash_attn_import_error = None


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@lru_cache(maxsize=1)
def _flash_attn_func_supports_blockscale() -> bool:
    if _flash_attn_func is None:
        return False
    try:
        return "mSFQ" in inspect.signature(_flash_attn_func).parameters
    except (TypeError, ValueError):
        return False


def _raise_missing_blockscale_support() -> None:
    raise ImportError(
        "FA4 block-scaled FP4 attention requires a flash_attn.cute build with "
        "mSFQ/mSFK support, for example hao-ai-lab/flash-attention-fp4@fp4."
    ) from _flash_attn_import_error


@lru_cache(maxsize=1)
def _flash_attn_varlen_signature() -> tuple[set[str], bool]:
    signature = inspect.signature(_flash_attn_varlen_func)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    return set(signature.parameters), accepts_kwargs


def _call_flash_attn_varlen_func(**kwargs):
    params, accepts_kwargs = _flash_attn_varlen_signature()
    if accepts_kwargs:
        return _flash_attn_varlen_func(**kwargs)

    call_kwargs = {}
    for key, value in kwargs.items():
        if key in params:
            call_kwargs[key] = value
            continue
        if key in ("max_seqlen_q", "max_seqlen_k"):
            continue
        if value is None or (key == "return_lse" and value is False):
            continue
        raise TypeError(
            "flash_attn.cute.flash_attn_varlen_func from this build does not "
            f"support the requested argument: {key}"
        )
    return _flash_attn_varlen_func(**call_kwargs)


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
    mSFQ: Optional[torch.Tensor] = None,
    mSFK: Optional[torch.Tensor] = None,
    mSFV: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
):
    if _flash_attn_varlen_func is None:  # pragma: no cover
        raise ImportError(
            "Vendored FlashAttention CUTE is not available (cannot import "
            "flash_attn.cute). Please check your source tree."
        ) from _flash_attn_import_error

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

    if mSFQ is not None or mSFK is not None or mSFV is not None:
        if not _flash_attn_func_supports_blockscale():
            _raise_missing_blockscale_support()
        if any(
            tensor is not None
            for tensor in (
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
            )
        ):
            raise NotImplementedError(
                "FA4 block-scaled FP4 attention currently supports only dense "
                "non-varlen diffusion attention."
            )
        if score_mod is not None or aux_tensors is not None:
            raise NotImplementedError(
                "FA4 block-scaled FP4 attention does not support score_mod or "
                "aux_tensors yet."
            )

        result = _flash_attn_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            softcap=0.0 if softcap is None else softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mSFQ=mSFQ,
            mSFK=mSFK,
            mSFV=mSFV,
        )
        if return_softmax_lse:
            return result
        if isinstance(result, tuple):
            return result[0]
        return result

    result = _call_flash_attn_varlen_func(
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
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError("FA4 path does not support descale.")

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
        num_splits=num_splits if num_splits != 0 else 1,
        pack_gqa=pack_gqa,
        learnable_sink=sinks,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_softmax_lse=True,
    )

    if return_softmax_lse:
        return result
    if isinstance(result, tuple):
        return result[0]
    return result
