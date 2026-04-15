"""Attention wrapper for MLX backend using native paged attention."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.kv_pool import MlxKVPool

_thread_local = threading.local()


@dataclass
class PagedAttentionContext:
    """Context set before forward pass, read by attention wrappers."""
    
    kv_pool: MlxKVPool
    cu_seqlens: list[int]
    offsets: list[int] | None
    slot_mapping: list[int]
    block_tables: list[list[int]]
    context_lens: list[int]


def set_context(ctx: Optional[PagedAttentionContext]) -> None:
    _thread_local.paged_ctx = ctx


def get_context() -> Optional[PagedAttentionContext]:
    return getattr(_thread_local, "paged_ctx", None)


def clear_context() -> None:
    _thread_local.paged_ctx = None


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention for paged attention.

    When ``PagedAttentionContext`` is set, performs paged attention using
    the native Metal kernels. Otherwise delegates to inner module.
    """

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        
        # Lazy import to avoid circular dependency
        from sglang.srt.hardware_backend.mlx.attention.attention_sdpa import sdpa_forward
        return sdpa_forward(self._inner, x, ctx, self._layer_idx)
