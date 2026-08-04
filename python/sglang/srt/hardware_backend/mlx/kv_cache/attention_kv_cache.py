"""Attention KV cache adapters for the MLX backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
        MlxAttentionKVPool,
    )


def make_attention_mask(N, offset, return_array=False, window_size=None):
    """Mirror mlx_lm ``cache.create_attention_mask`` for cache shims.

    Containers delegate mask creation to ``cache.make_mask`` whenever the
    cache exposes it, so the shims must honor ``window_size`` (sliding-window
    layers pass it, including for N == 1) or windowed models silently fall
    back to full attention.
    """
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    if N == 1:
        return None
    if return_array:
        return create_causal_mask(N, offset)
    return "causal"


class AttentionOffsetCache:
    """Data-free shim satisfying mlx-lm's cache protocol.

    Provides ``make_mask`` and ``state`` without storing actual K/V.
    """

    def __init__(self, offset: int = 0):
        self.offset = offset

    @property
    def state(self):
        return ()  # Empty — safe for mx.eval unpacking

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def update_and_fetch(self, keys, values):
        raise RuntimeError("AttentionOffsetCache should not store data")


_DEFAULT_MAX_SEQ_LEN = 4096


class ContiguousAttentionKVCache:
    """Pre-allocated attention KV buffer for one request and one layer.

    Shape ``(1, n_kv_heads, max_seq_len, head_dim)``.  Slice assignment
    instead of ``mx.concatenate``.  Lazy-allocated on first write.
    """

    __slots__ = ("keys", "values", "offset", "max_seq_len")

    def __init__(
        self,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = _DEFAULT_MAX_SEQ_LEN,
        dtype: mx.Dtype | None = None,
    ):
        if n_kv_heads is not None and head_dim is not None and dtype is not None:
            self.keys = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
            self.values = mx.zeros((1, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        else:
            self.keys = None
            self.values = None
        self.offset = 0
        self.max_seq_len = max_seq_len

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def _allocate(self, keys: mx.array) -> None:
        """Allocate buffers matching the first key tensor's shape."""
        B, n_kv_heads, _, head_dim = keys.shape
        self.keys = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )
        self.values = mx.zeros(
            (B, n_kv_heads, self.max_seq_len, head_dim), dtype=keys.dtype
        )

    @property
    def state(self):
        """Arrays for ``mx.eval`` unpacking."""
        if self.keys is None:
            return ()
        return (self.keys, self.values)

    def _grow(self, required: int) -> None:
        """Double the buffer until it can hold *required* tokens."""
        new_max = self.max_seq_len
        while new_max < required:
            new_max *= 2
        B, n_kv_heads, _, head_dim = self.keys.shape
        new_k = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, new_max, head_dim), dtype=self.values.dtype)
        if self.offset > 0:
            new_k[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
            new_v[:, :, : self.offset, :] = self.values[:, :, : self.offset, :]
        self.keys = new_k
        self.values = new_v
        self.max_seq_len = new_max

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append K/V and return all valid K/V up to current offset."""
        if self.keys is None:
            self._allocate(keys)
        S = keys.shape[2]
        end = self.offset + S
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = keys
        self.values[:, :, self.offset : end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def write_token(self, k: mx.array, v: mx.array) -> None:
        """Write one token. k, v shape: (1, n_kv_heads, 1, head_dim)."""
        end = self.offset + 1
        if end > self.max_seq_len:
            self._grow(end)
        self.keys[:, :, self.offset : end, :] = k
        self.values[:, :, self.offset : end, :] = v
        self.offset = end

    def get_kv(self) -> tuple[mx.array, mx.array]:
        """Return valid K/V: (1, n_kv_heads, offset, head_dim)."""
        return self.keys[:, :, : self.offset, :], self.values[:, :, : self.offset, :]


class PoolBackedAttentionKVCache:
    """Lazily gathers cached attention KV from the shared pool during forward.

    Each ``update_and_fetch`` gathers this layer's prefix from the pool
    on demand, keeping operations in the lazy compute graph.  Convert to
    ``ContiguousAttentionKVCache`` via ``to_contiguous`` after the forward pass.
    """

    __slots__ = (
        "_pool",
        "_layer_idx",
        "_slots",
        "offset",
        "_full_keys",
        "_full_values",
        "_new_keys",
        "_new_values",
    )

    def __init__(
        self,
        pool: MlxAttentionKVPool,
        layer_idx: int,
        slots: mx.array,
        prefix_len: int,
    ):
        self._pool = pool
        self._layer_idx = layer_idx
        self._slots = slots
        self.offset = prefix_len
        self._full_keys: mx.array | None = None
        self._full_values: mx.array | None = None
        self._new_keys: mx.array | None = None
        self._new_values: mx.array | None = None

    @property
    def keys(self) -> mx.array | None:
        return self._full_keys

    @property
    def values(self) -> mx.array | None:
        return self._full_values

    @property
    def state(self):
        if self._full_keys is not None:
            return (self._full_keys, self._full_values)
        return ()

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return make_attention_mask(
            N, self.offset, return_array=return_array, window_size=window_size
        )

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Gather cached prefix from pool, concatenate with new K/V."""
        S = keys.shape[2]

        if self.offset > 0:
            k_cached, v_cached = self._pool.get_kv(
                self._layer_idx, self._slots[: self.offset]
            )
            # Pool layout (S, n_kv_heads, head_dim) → cache (1, n_kv_heads, S, head_dim)
            k_cached = k_cached.transpose(1, 0, 2)[None]
            v_cached = v_cached.transpose(1, 0, 2)[None]
            k_all = mx.concatenate([k_cached, keys], axis=2)
            v_all = mx.concatenate([v_cached, values], axis=2)
        else:
            k_all = keys
            v_all = values

        self.offset += S
        self._full_keys = k_all
        self._full_values = v_all
        self._new_keys = keys
        self._new_values = values
        return k_all, v_all

    def to_contiguous(self, max_seq_len: int = 4096) -> ContiguousAttentionKVCache:
        """Convert to contiguous attention KV reusing forward-pass arrays."""
        cache = ContiguousAttentionKVCache(max_seq_len=max_seq_len)
        if self._full_keys is not None:
            cache.update_and_fetch(self._full_keys, self._full_values)
        return cache
