"""KV cache abstractions for causal video diffusion inference."""

from __future__ import annotations

import torch
from torch import Tensor


class SelfAttentionKVCache:
    """Per-layer self-attention KV buffer with sliding window eviction."""

    __slots__ = (
        "k",
        "v",
        "_global_end",
        "_local_end",
        "_sink_size",
        "_frame_seq_length",
        "_batch_size",
        "_max_size",
        "_num_heads",
        "_head_dim",
        "_dtype",
        "_device",
    )

    def __init__(
        self,
        batch_size: int,
        max_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        sink_size: int = 0,
        frame_seq_length: int = 1,
    ):
        self._batch_size: int = batch_size
        self._max_size: int = max_size
        self._num_heads: int = num_heads
        self._head_dim: int = head_dim
        self._dtype: torch.dtype = dtype
        self._device: torch.device = device
        self._global_end: int = 0
        self._local_end: int = 0
        self._sink_size: int = sink_size
        self._frame_seq_length: int = frame_seq_length
        self.k: Tensor | None = None
        self.v: Tensor | None = None
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        self.k = torch.zeros(
            self._batch_size,
            self._max_size,
            self._num_heads,
            self._head_dim,
            dtype=self._dtype,
            device=self._device,
        ).contiguous()
        self.v = torch.zeros(
            self._batch_size,
            self._max_size,
            self._num_heads,
            self._head_dim,
            dtype=self._dtype,
            device=self._device,
        ).contiguous()

    def _ensure_allocated(self) -> None:
        if self.k is None or self.v is None:
            self._allocate_buffers()

    @property
    def max_size(self) -> int:
        return self._max_size

    def reset(self) -> None:
        self._global_end = 0
        self._local_end = 0

    def release(self) -> None:
        """Release GPU memory; buffers are lazily re-allocated on next write."""
        self.k = None
        self.v = None
        self._global_end = 0
        self._local_end = 0

    def bulk_write(self, key: Tensor, value: Tensor) -> None:
        """Overwrite from position 0 (recompute phase with block_mask)."""
        self._ensure_allocated()
        seq_len = key.shape[1]
        if seq_len > self.max_size:
            raise ValueError(
                f"bulk_write seq_len ({seq_len}) exceeds buffer capacity ({self.max_size})"
            )
        self.k[:, :seq_len] = key
        self.v[:, :seq_len] = value
        self._global_end = seq_len
        self._local_end = seq_len

    def append(
        self,
        key: Tensor,
        value: Tensor,
        current_start: int,
    ) -> None:
        """Append new KV; evicts old tokens (preserving sink) when buffer is full."""
        self._ensure_allocated()
        current_end = current_start + key.shape[1]
        num_new = key.shape[1]
        buf_size = self.max_size
        sink_tokens = self._sink_size * self._frame_seq_length

        if num_new > buf_size:
            raise ValueError(
                f"append num_new ({num_new}) exceeds buffer capacity ({buf_size})"
            )

        needs_eviction = (
            current_end > self._global_end and num_new + self._local_end > buf_size
        )

        if needs_eviction:
            num_evicted = num_new + self._local_end - buf_size
            num_rolled = self._local_end - num_evicted - sink_tokens
            if num_rolled < 0:
                raise ValueError(
                    f"sink_tokens ({sink_tokens}) too large for eviction: "
                    f"local_end={self._local_end}, num_evicted={num_evicted}, "
                    f"num_rolled={num_rolled}"
                )
            self.k[:, sink_tokens : sink_tokens + num_rolled] = self.k[
                :,
                sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
            ].clone()
            self.v[:, sink_tokens : sink_tokens + num_rolled] = self.v[
                :,
                sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
            ].clone()
            local_end = self._local_end + current_end - self._global_end - num_evicted
        else:
            self.k = self.k.detach()
            self.v = self.v.detach()
            local_end = self._local_end + current_end - self._global_end

        local_start = local_end - num_new
        self.k[:, local_start:local_end] = key
        self.v[:, local_start:local_end] = value
        self._global_end = current_end
        self._local_end = local_end

    def get_active_kv(self, max_attention_size: int) -> tuple[Tensor, Tensor]:
        """Return (k, v) sliced to at most ``max_attention_size`` from the end."""
        self._ensure_allocated()
        start = max(0, self._local_end - max_attention_size)
        return (
            self.k[:, start : self._local_end],
            self.v[:, start : self._local_end],
        )


class CrossAttentionKVCache:
    """Per-layer cross-attention KV cache with compute-once semantics.

    First ``update()`` stores K,V; subsequent calls use ``get()``.
    Lazy-initialized: no memory allocated until ``update()``.
    """

    __slots__ = ("k", "v")

    def __init__(self):
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    @property
    def is_initialized(self) -> bool:
        return self.k is not None

    def reset(self) -> None:
        self.k = None
        self.v = None

    def release(self) -> None:
        """Alias of reset() for lifecycle symmetry with self-attn cache."""
        self.reset()

    def update(self, k: Tensor, v: Tensor) -> None:
        self.k = k
        self.v = v

    def get(self) -> tuple[Tensor, Tensor]:
        return self.k, self.v


class KVCacheManager:
    """Per-layer KV cache container for all transformer blocks.

    Self-attention and cross-attention caches are independently optional:
    omit ``sa_*`` params to skip self-attention, set ``create_cross_attn=False``
    to skip cross-attention.
    """

    def __init__(
        self,
        num_blocks: int,
        sa_batch_size: int | None = None,
        sa_max_size: int | None = None,
        sa_num_heads: int | None = None,
        sa_head_dim: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        sink_size: int = 0,
        frame_seq_length: int = 1,
        create_cross_attn: bool = True,
    ):
        sa_params = (
            sa_batch_size,
            sa_max_size,
            sa_num_heads,
            sa_head_dim,
            dtype,
            device,
        )
        sa_any_set = any(p is not None for p in sa_params)
        sa_all_set = all(p is not None for p in sa_params)
        if sa_any_set and not sa_all_set:
            missing = [
                name
                for name, val in zip(
                    (
                        "sa_batch_size",
                        "sa_max_size",
                        "sa_num_heads",
                        "sa_head_dim",
                        "dtype",
                        "device",
                    ),
                    sa_params,
                )
                if val is None
            ]
            raise ValueError(
                f"Self-attention cache requires all parameters; missing: {missing}"
            )

        if sa_all_set:
            self.self_attn_caches: list[SelfAttentionKVCache] | None = [
                SelfAttentionKVCache(
                    sa_batch_size,
                    sa_max_size,
                    sa_num_heads,
                    sa_head_dim,
                    dtype,
                    device,
                    sink_size=sink_size,
                    frame_seq_length=frame_seq_length,
                )
                for _ in range(num_blocks)
            ]
        else:
            self.self_attn_caches = None

        if create_cross_attn:
            self.cross_attn_caches: list[CrossAttentionKVCache] | None = [
                CrossAttentionKVCache() for _ in range(num_blocks)
            ]
        else:
            self.cross_attn_caches = None

    def reset_self_attn(self) -> None:
        if self.self_attn_caches is not None:
            for cache in self.self_attn_caches:
                cache.reset()

    def reset_cross_attn(self) -> None:
        if self.cross_attn_caches is not None:
            for cache in self.cross_attn_caches:
                cache.reset()

    def release(self) -> None:
        """Release all GPU memory from both cache types."""
        if self.self_attn_caches is not None:
            for cache in self.self_attn_caches:
                cache.release()
        if self.cross_attn_caches is not None:
            for cache in self.cross_attn_caches:
                cache.release()
