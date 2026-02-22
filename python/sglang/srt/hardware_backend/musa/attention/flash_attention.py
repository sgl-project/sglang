"""MUSA Flash Attention wrapper with automatic scheduler_metadata injection.

This module provides a wrapper for mate's flash_attn_with_kvcache that automatically
computes and injects scheduler_metadata based on the current FlashAttentionContext.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from mate import flash_attn_with_kvcache as _mate_flash_attn_with_kvcache
from mate.mha_interface import get_scheduler_metadata

from sglang.srt.distributed import get_pp_group, get_pp_indices
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

# Global workspace buffer for MLA
_MATE_MLA_WORKSPACE_BUFFER: torch.Tensor | None = None

# Cache for non-MLA scheduler metadata by prefix
_MATE_NO_MLA_SCHEDULER_METADATA_DICT: dict = {}

# Thread-local storage for flash attention context
_flash_attention_context = threading.local()


@dataclass
class FlashAttentionContext:
    """Context for MUSA flash attention calls.

    This context stores the information needed to compute scheduler_metadata
    for mate's flash_attn_with_kvcache.
    """

    # Static config (set once per backend)
    device: torch.device
    use_mla: bool
    num_hidden_layers: int
    first_k_dense_replace: int
    full_attention_interval: Optional[int]

    # Dynamic state (set per forward call)
    layer: "RadixAttention"
    prefix: str
    max_seqlen_k: int
    can_run_tbo: bool


class FlashAttentionContextManager:
    """Context manager for MUSA flash attention.

    Automatically sets and clears the flash attention context on entry/exit.
    This ensures cleanup happens even on early returns or exceptions.

    Usage:
        with FlashAttentionContextManager(ctx):
            # flash_attn_with_kvcache calls will auto-inject scheduler_metadata
            ...
    """

    def __init__(self, ctx: FlashAttentionContext):
        self.ctx = ctx

    def __enter__(self) -> "FlashAttentionContextManager":
        _flash_attention_context.current = self.ctx
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _flash_attention_context.current = None
        return None  # Don't suppress exceptions


def get_flash_attention_context() -> Optional[FlashAttentionContext]:
    """Get the current flash attention context."""
    return getattr(_flash_attention_context, "current", None)


def update_flash_attention_context(
    prefix: Optional[str] = None,
    max_seqlen_k: Optional[int] = None,
) -> None:
    """Update specific fields of the current flash attention context.

    This is useful for cascade attention where prefix and max_seqlen_k change.
    """
    ctx = get_flash_attention_context()
    if ctx is not None:
        if prefix is not None:
            ctx.prefix = prefix
        if max_seqlen_k is not None:
            ctx.max_seqlen_k = max_seqlen_k


def _compute_scheduler_metadata(
    ctx: FlashAttentionContext,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: Optional[torch.Tensor],
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    page_size: int,
    causal: bool,
    window_size: Tuple[int, int],
    num_splits: int,
) -> Tuple[torch.Tensor, bool] | torch.Tensor:
    """Compute scheduler metadata based on context."""
    global _MATE_MLA_WORKSPACE_BUFFER, _MATE_NO_MLA_SCHEDULER_METADATA_DICT

    layer = ctx.layer
    current_layer_id = layer.layer_id
    batch_size = cu_seqlens_q.shape[-1] - 1

    # Determine if scheduler metadata should be updated
    should_update = True
    pp_group = get_pp_group()
    pp_rank = pp_group.rank_in_group
    start_layer_id, _ = get_pp_indices(
        ctx.num_hidden_layers, pp_group.rank_in_group, pp_group.world_size
    )
    if ctx.can_run_tbo and pp_rank == 0:
        start_layer_id += (
            ctx.first_k_dense_replace if ctx.first_k_dense_replace is not None else 0
        )

    if ctx.full_attention_interval is not None:
        start_layer_id += ctx.full_attention_interval - 1

    if current_layer_id > start_layer_id:
        should_update = False

    if envs.SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA.get():
        should_update = True

    if ctx.use_mla:
        if _MATE_MLA_WORKSPACE_BUFFER is None:
            _MATE_MLA_WORKSPACE_BUFFER = torch.empty(
                128 * 1024 * 1024, device=ctx.device, dtype=torch.uint8
            )
        return (_MATE_MLA_WORKSPACE_BUFFER, not should_update)
    else:
        if should_update or ctx.prefix not in _MATE_NO_MLA_SCHEDULER_METADATA_DICT:
            _MATE_NO_MLA_SCHEDULER_METADATA_DICT[ctx.prefix] = get_scheduler_metadata(
                batch_size=batch_size,
                num_heads_q=layer.tp_q_head_num,
                num_heads_kv=layer.tp_k_head_num,
                headdim=layer.qk_head_dim,
                headdim_v=layer.v_head_dim,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                page_size=page_size,
                causal=causal,
                window_size=window_size,
                num_splits=num_splits,
            )
        return _MATE_NO_MLA_SCHEDULER_METADATA_DICT[ctx.prefix]


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
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa=None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    **kwargs,
):
    """MUSA flash_attn_with_kvcache wrapper that auto-injects scheduler_metadata.

    This wrapper retrieves the current FlashAttentionContext and computes
    scheduler_metadata automatically, so call sites don't need to be modified.
    """
    # Get context and compute scheduler_metadata if not provided
    if scheduler_metadata is None:
        ctx = get_flash_attention_context()
        if ctx is not None:
            page_size = k_cache.shape[1] if k_cache is not None else 1
            scheduler_metadata = _compute_scheduler_metadata(
                ctx=ctx,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                cache_seqlens=cache_seqlens,
                max_seqlen_q=max_seqlen_q,
                page_size=page_size,
                causal=causal,
                window_size=window_size,
                num_splits=num_splits,
            )

    return _mate_flash_attn_with_kvcache(
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
    )
