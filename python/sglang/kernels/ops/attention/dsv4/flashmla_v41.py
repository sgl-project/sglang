from typing import Optional, Tuple

import torch

try:
    import flash_mla
    from flash_mla import fused_norm_rope_attn_rope_cast as _mega
except ImportError as _e:
    _import_error: Optional[ImportError] = _e
else:
    _import_error = None

_MEGA_FIXED = dict(
    enable_q_norm=False,
    rms_norm_eps=0.0,
    is_rope_neox_style=False,
    rope_dim=64,
    num_per_channels=32,
    use_tma_aligned_col_major_sf=True,
    round_sf=True,
    use_packed_ue8m0=True,
)


def load() -> None:
    if _import_error is not None:
        raise ImportError(
            "The token-interleaved DeepSeek-V4.1 KV cache needs the sgl-flash-mla wheel "
            "(import name flash_mla)"
        ) from _import_error


def is_available() -> bool:
    return _import_error is None


def sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    head_dim_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    load()
    return flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=kv,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=flash_mla.get_mla_metadata()[0],
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_kv,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        use_legacy=False,
    )


def mega_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    load()
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
        use_legacy=False,
        **_MEGA_FIXED,
    )


def mega_decode(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    load()
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
        use_legacy=False,
        **_MEGA_FIXED,
    )
