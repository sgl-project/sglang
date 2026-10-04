"""PPU-specific FlashMLA ops.

Wraps the ``flash_mla`` library (PPU) with the same signatures
expected by :mod:`sglang.srt.layers.attention.flashmla_backend`.

Exported symbols
----------------
get_mla_metadata
flash_mla_with_kvcache
flash_mla_sparse_fwd
"""

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

try:
    import flash_mla as _flashmla
    from flash_mla import FlashMLASchedMeta

    _import_error = None
except Exception as _e:
    _flashmla = None
    _import_error = _e

_IMPORT_ERROR = ImportError(
    "Failed to import flash_mla for PPU. "
    "Ensure the flash_mla package is installed in your PPU environment."
)


# Add for nvtx profiling
SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
SGLANG_PROFILE_NVTX_PRINT_SEQLEN = envs.SGLANG_PROFILE_NVTX_PRINT_SEQLEN.get()
if SGLANG_PROFILE_NVTX:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError as e:
        SGLANG_PROFILE_NVTX = False
        SGLANG_PROFILE_NVTX_PRINT_SEQLEN = False


def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_tokens_per_head_k: Optional[int] = None,
    num_heads_k: Optional[int] = None,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return MLA tile scheduler metadata for PPU.

    Delegates to ``flash_mla.get_mla_metadata``.

    Arguments mirror :func:`sgl_kernel.flash_mla.get_mla_metadata`.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    if cache_seqlens is None:
        return FlashMLASchedMeta(), None

    assert num_q_tokens_per_head_k is not None
    assert num_heads_k is not None

    if is_fp8_kvcache and topk is None:
        raise NotImplementedError(
            "FlashMLA dense FP8 get_mla_metadata is not supported on PPU."
        )

    return _flashmla.get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q=num_heads_q,
        is_fp8_kvcache=is_fp8_kvcache,
        topk=topk,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor | FlashMLASchedMeta,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FlashMLA decode attention on PPU.

    Dense FP8 (``indices is None and q.element_size() == 1``) is not
    supported on PPU and will raise ``NotImplementedError``.

    All other cases delegate to ``flash_mla.flash_mla_with_kvcache``.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if indices is None and q.element_size() == 1:
        raise NotImplementedError(
            "FlashMLA dense FP8 with kvcache is not supported on PPU."
        )

    nvtx_pushed = SGLANG_PROFILE_NVTX and num_splits is not None
    if nvtx_pushed:
        batch_size = len(num_splits) - 1
        if q.dim() == 4:
            max_seqlen_q = q.shape[-3]
        else:
            max_seqlen_q = 1

        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_k_list = (
                    cache_seqlens.flatten().cpu().tolist()
                    if cache_seqlens is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal},num_blocks:{k_cache.shape[-4]},page_block_size:{k_cache.shape[-3]},cu_seqlens_k:{cu_seqlens_k_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        th_nvtx_range_push(nvtx_message)

    try:
        out, softmax_lse = _flashmla.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )
    finally:
        if nvtx_pushed:
            th_nvtx_range_pop()
    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse prefill attention on PPU.

    Delegates to ``flash_mla.flash_mla_sparse_fwd``.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    return _flashmla.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
