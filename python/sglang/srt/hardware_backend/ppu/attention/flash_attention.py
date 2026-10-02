"""PPU flash attention operations.

Wraps the PPU flash-attention library (flash_attn_interface) to provide
the same call interface as sgl_kernel.flash_attn so that the rest of SGLang
can use PPU flash attention transparently.
"""

from typing import Optional, Union

import torch
from sgl_kernel.debug_utils import maybe_wrap_debug_kernel

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa_varlen_func
    from flash_attn_interface import flash_attn_with_kvcache as _fa_with_kvcache
    from flash_attn_interface import get_scheduler_metadata as _get_metadata
except ImportError:
    raise ImportError(
        "Cannot import flash-attention for PPU. "
        "Please check your installation: pip install flash-attn-3."
    )


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    headdim,
    cache_seqlens,
    qkv_dtype,
    headdim_v=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    leftpad_k=None,
    page_size=None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
):
    """Get scheduler metadata for PPU flash attention (wraps flash_attn_interface)."""
    return _get_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        headdim,
        cache_seqlens,
        qkv_dtype,
        headdim_v,
        cu_seqlens_q,
        cu_seqlens_k_new,
        leftpad_k,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size,
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )


@maybe_wrap_debug_kernel
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
    out=None,
    max_seqlen_k: Optional[int] = None,
):
    """PPU flash_attn_with_kvcache wrapper (flash_attn_interface backend)."""
    if ver == 4:
        raise NotImplementedError(
            "FA4 is not available on PPU: flash_attn_with_kvcache not implemented."
        )

    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    return _fa_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        s_aux=sinks,
    )


@maybe_wrap_debug_kernel
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
    out=None,
    deterministic=False,
):
    """PPU flash_attn_varlen_func wrapper (flash_attn_interface backend)."""
    if ver == 4:
        raise NotImplementedError(
            "FA4 is not available on PPU: flash_attn_varlen_func not implemented."
        )

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    attention_chunk = 0 if attention_chunk is None else int(attention_chunk)

    return _fa_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
        sm_margin=sm_margin,
        return_attn_probs=return_softmax_lse,
        s_aux=sinks,
    )
