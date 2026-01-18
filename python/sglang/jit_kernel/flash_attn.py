from functools import lru_cache
from typing import Optional, Union

import torch

try:
    from ._fa4_interface import flash_attn_varlen_func as flash_attn_varlen_func_v4
except ImportError:
    flash_attn_varlen_func_v4 = None


@lru_cache(maxsize=1)
def is_fa3_supported(device=None) -> bool:
    return (torch.version.cuda >= "12.3") and (
        torch.cuda.get_device_capability(device)[0] == 9
        or torch.cuda.get_device_capability(device)[0] == 8
    )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
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
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
):
    if ver == 4:
        assert (
            flash_attn_varlen_func_v4 is not None
        ), "FA4 is not available, please check your installation."
        assert k is None and v is None, "FA4 does not support updating KV cache in-place."
        assert (
            rotary_cos is None and rotary_sin is None and rotary_seqlens is None
        ), "FA4 does not support rotary embedding."
        assert (
            cache_batch_idx is None and cache_leftpad is None
        ), "FA4 does not support non-consecutive batch indices or left padding."
        assert (
            q_descale is None and k_descale is None and v_descale is None
        ), "FA4 does not support descale."

        if window_size == (-1, -1):
            window_size = (None, None)

        return flash_attn_varlen_func_v4(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            return_softmax_lse=return_softmax_lse,
            learnable_sink=sinks,
            page_table=page_table,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    q, k_cache, k, v = [maybe_contiguous(x) for x in (q, k_cache, k, v)]
    v_cache = (
        v_cache.contiguous()
        if v_cache.stride(-1) != 1 and v_cache.stride(-3) != 1
        else v_cache
    )
    cu_seqlens_q, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k_new)
    ]
    page_table, cache_batch_idx, cache_leftpad = [
        maybe_contiguous(x) for x in (page_table, cache_batch_idx, cache_leftpad)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    rotary_seqlens = maybe_contiguous(rotary_seqlens)
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,
        cu_seqlens_q,
        None,
        cu_seqlens_k_new,
        None,
        cache_seqlens,
        max_seqlen_q,
        None,
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
        sinks,
    )
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
):
    if ver == 4:
        assert (
            flash_attn_varlen_func_v4 is not None
        ), "FA4 is not available, please check your installation."
        if window_size == (-1, -1):
            window_size = (None, None)
        return flash_attn_varlen_func_v4(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            pack_gqa=pack_gqa,
            learnable_sink=sinks,
            return_softmax_lse=return_softmax_lse,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

    if not is_fa3_supported():
        raise NotImplementedError(
            "flash_attn is only supported on sm80+ with CUDA>=12.3 for FA3"
        )

    if max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("max_seqlen_q and max_seqlen_k are required for FA3")

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k,
        v,
        None,
        None,
        qv,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,
        None,
        None,
        None,
        None,
        None,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        softcap,
        is_rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        sinks=sinks,
    )

    return (out, softmax_lse, *rest) if return_softmax_lse else out
