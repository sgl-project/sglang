from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.utils.custom_op import register_custom_op


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla

    return flash_mla.flash_mla_with_kvcache(**kwargs)


def _flash_mla_with_kvcache_output_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    tile_scheduler_metadata: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    *,
    head_dim_v: int,
    softmax_scale: float,
    is_fp8_kvcache: bool,
    backend: str,
) -> torch.Tensor:
    return q.new_empty((*q.shape[:-1], head_dim_v))


@register_custom_op(fake_impl=_flash_mla_with_kvcache_output_fake)
def flash_mla_with_kvcache_output_pcg_op(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    tile_scheduler_metadata: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    *,
    head_dim_v: int,
    softmax_scale: float,
    is_fp8_kvcache: bool,
    backend: str,
) -> torch.Tensor:
    assert backend == "kernel", f"unsupported backend {backend!r}"
    import flash_mla

    return flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        head_dim_v=head_dim_v,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=tile_scheduler_metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
    )[0]
