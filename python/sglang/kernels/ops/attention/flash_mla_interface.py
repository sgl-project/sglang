"""FlashMLA from the ``sgl-flash-mla`` wheel, behind the call signatures of the
FlashMLA ops sgl-kernel used to ship (``sgl_kernel.flash_mla`` and
``sgl_kernel.flash_mla_mega``).

The wheel carries two kernel sets. Everything the fork had before FlashMLA's
2026.09.30 release (SM90 dense decode, the V3.2 / V3.2-no-RoPE / V4 formats,
the block-split V4.1 caches and their MegaAttention) is its legacy side, the
same sources sgl-kernel built, so ``use_legacy`` defaults to True. Only the
token-interleaved V4.1 caches (``KVLayout.V41_NEW`` / ``V41_FP4_NEW``) take
``use_legacy=False``. CUDA only: HIP and XPU keep their own sgl-kernel ops.
"""

from typing import Optional, Tuple, Union

import flash_mla
import torch
from flash_mla import fused_norm_rope_attn_rope_cast as _mega
from flash_mla.flash_mla_interface import FlashMLASchedMeta

# 512 fp8 values + 4 fp32 scales; the V4.1 formats have the same size, which is
# why the tensor-metadata API (it predates V4.1) reads such a cache as V3.2.
_V32_NO_ROPE_BYTES_PER_TOKEN = 528

# The only RoPE / output-cast configuration MegaAttention accepts:
# both kernel sets assert every one of these values.
_MEGA_FIXED_PARAMS = dict(
    is_rope_neox_style=False,
    rope_dim=64,
    num_per_channels=32,
    use_tma_aligned_col_major_sf=True,
    round_sf=True,
    use_packed_ue8m0=True,
)

# The current (use_legacy=False) decode kernel splits the KV only past this
# many main + extra indices; below it it schedules nothing and leaves the
# metadata empty. Mirrors `enable_split_kv` in FlashMLA's csrc/api/sparse_decode.cpp.
SPLIT_KV_MIN_TOPK = 641


def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_tokens_per_head_k: Optional[int] = None,
    num_heads_k: Optional[int] = None,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[Union[torch.Tensor, FlashMLASchedMeta], Optional[torch.Tensor]]:
    """Without ``cache_seqlens``, an empty ``FlashMLASchedMeta`` that the first
    decode call fills; with it, the legacy split-KV schedule as tensors."""
    if cache_seqlens is None:
        return flash_mla.get_mla_metadata()[0], None
    assert num_q_tokens_per_head_k is not None
    assert num_heads_k is not None
    return flash_mla.get_mla_decoding_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
        use_legacy=True,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Union[torch.Tensor, FlashMLASchedMeta],
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
    kv_format: Optional[str] = None,
    *,
    use_legacy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paged MLA decode. ``tile_scheduler_metadata`` is a ``FlashMLASchedMeta``,
    or the schedule tensors from ``get_mla_metadata(cache_seqlens, ...)`` with
    ``num_splits``, which ride on a fresh ``FlashMLASchedMeta`` (legacy only:
    V3.2 sparse, dense bf16, or dense fp8 when ``q`` is fp8)."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if isinstance(tile_scheduler_metadata, FlashMLASchedMeta):
        assert num_splits is None, "num_splits must be None with FlashMLASchedMeta"
        sched_meta = tile_scheduler_metadata
    else:
        assert use_legacy, (
            "tensor scheduling metadata exists only for the legacy kernels"
        )
        assert num_splits is not None
        assert block_table is not None and cache_seqlens is not None
        assert all(
            x is None for x in (attn_sink, extra_k_cache, extra_indices_in_kvcache)
        ), "attn_sink and extra caches need FlashMLASchedMeta"
        assert (descale_q is None) == (descale_k is None), (
            "descale_q and descale_k should be both None or both not None"
        )
        sched_meta = FlashMLASchedMeta(
            tile_scheduler_metadata=tile_scheduler_metadata, num_splits=num_splits
        )
        if indices is None:
            is_fp8_kvcache = q.element_size() == 1
        elif (
            kv_format is None
            and q.shape[-1] == 512
            and k_cache.shape[-1] == _V32_NO_ROPE_BYTES_PER_TOKEN
        ):
            kv_format = "V32_NO_ROPE"
    return flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        use_legacy=use_legacy,
        kv_format=kv_format,
        descale_q=descale_q,
        descale_k=descale_k,
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    use_legacy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse prefill over a bf16 ``kv``: (output, max_logits, lse). The legacy
    kernel's statistics are base 2, as sgl-kernel returned them; the current
    kernel's are natural-log."""
    return flash_mla.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        use_legacy=use_legacy,
        return_log2_lse=True if use_legacy else None,
    )


def flash_mla_mega_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    enable_q_norm: bool = False,
    rms_norm_eps: float = 0.0,
    use_legacy: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """MegaAttention prefill over a bf16 ``kv``: (out_fp8, out_sf, max_logits, lse)."""
    return _mega.prefill(
        token_positions=positions,
        cos_sin_cache=cos_sin_cache,
        n_wv_group=q.shape[1] // 8,
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        enable_q_norm=enable_q_norm,
        rms_norm_eps=rms_norm_eps,
        use_legacy=use_legacy,
        **_MEGA_FIXED_PARAMS,
    )


def flash_mla_mega_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    enable_q_norm: bool = False,
    rms_norm_eps: float = 0.0,
    use_legacy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MegaAttention decode over a paged cache: (out_fp8, out_sf, lse)."""
    return _mega.decode(
        token_positions=positions,
        cos_sin_cache=cos_sin_cache,
        n_wv_group=q.shape[1] // 8,
        q=q,
        k_cache=kv,
        indices_in_kvcache=indices,
        sm_scale=softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        extra_k_cache=extra_kv,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
        enable_q_norm=enable_q_norm,
        rms_norm_eps=rms_norm_eps,
        use_legacy=use_legacy,
        **_MEGA_FIXED_PARAMS,
    )
