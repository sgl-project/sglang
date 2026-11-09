from typing import Optional, Union

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import get_device_capability, is_musa


def _call_fa3_kernel(kernel, *args, out=None, **kwargs):
    if out is None:
        return kernel(*args, **kwargs)
    try:
        return kernel(*args, **kwargs, out=out)
    except TypeError as exc:
        if "unexpected keyword argument 'out'" not in str(exc):
            raise
        return kernel(*args, **kwargs)


@cache_once
def _load_fa3_kernels():
    from sgl_kernel.flash_attn import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    return {
        "flash_attn_with_kvcache": flash_attn_with_kvcache,
        "flash_attn_varlen_func": flash_attn_varlen_func,
    }


@cache_once
def _is_fa3_supported() -> bool:
    #  There some fa3 FYI
    #  FA3 can fail without a enough shared memory for a some shapes, such as higher
    #  hidden_dim or some special cases.
    #  Right now, fa3 is supported for sm80/sm87 and sm86/sm89. The main different
    #  Between sm80/sm87 and sm86/sm89 is the shared memory size. you can follow the link below for more information
    #  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x
    #  And for sgl-kernel right now, we can build fa3 on sm80/sm86/sm89/sm90a.
    #  That means if you use A100/A*0/L20/L40/L40s/4090 you can use fa3.
    if is_musa():
        major, _ = get_device_capability()
        return major >= 3
    if torch.version.cuda is not None and torch.version.cuda >= "12.3":
        arch = current_arch()
        return arch.is_sm(9) or arch.is_sm(8)
    return False


@debug_kernel_api
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
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk: Optional[int] = None,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    only_qv=False,  # Skip K matmul when qk rope dim is 0 (requires qv)
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
    sinks=None,
    out=None,
):
    if not _is_fa3_supported():
        raise NotImplementedError(
            "flash_attn at sgl-kernel is only supported on sm90 and above"
        )

    # When only_qv=True the caller may pass k_cache=None (synthetic K is
    # allocated inside the sgl-kernel wrapper). Skip the stride check in that
    # case so the rope=0 path doesn't trip the assertion.
    if k_cache is not None:
        assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"

    return _call_fa3_kernel(
        _load_fa3_kernels()["flash_attn_with_kvcache"],
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        rotary_cos,
        rotary_sin,
        cache_seqlens,
        cache_batch_idx,
        cache_leftpad,
        page_table,
        cu_seqlens_q,
        cu_seqlens_k_new,
        max_seqlen_q,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size,
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin=sm_margin,
        only_qv=only_qv,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
        out=out,
    )


@debug_kernel_api
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
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    out=None,
):

    if not _is_fa3_supported():
        # Fall back to flash_attn package (FA2) on platforms without sgl-kernel FA3
        # (e.g. ROCm, or CUDA < sm90)
        if cu_seqlens_q is not None:
            from flash_attn import flash_attn_varlen_func as fa2_flash_attn_varlen_func

            return fa2_flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                return_attn_probs=return_softmax_lse,
            )
        else:
            # 4D inputs (batch, seqlen, nheads, headdim) without cu_seqlens
            from flash_attn import flash_attn_func as fa2_flash_attn_func

            return fa2_flash_attn_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                return_attn_probs=return_softmax_lse,
            )

    return _call_fa3_kernel(
        _load_fa3_kernels()["flash_attn_varlen_func"],
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        only_qv=only_qv,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
        sinks=sinks,
        out=out,
    )
