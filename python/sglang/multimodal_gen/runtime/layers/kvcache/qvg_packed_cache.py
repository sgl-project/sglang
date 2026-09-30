# SPDX-License-Identifier: Apache-2.0
"""Quant-VideoGen KV cache with PRQ-packed storage for completed chunks.

Storage model (mirrors Quant-VideoGen's ChunkedKVCache, fitted to SGLang's
``update_and_get_attention_kv`` contract):

  * The retained window is split into frame/chunk-aligned *segments* in global
    token order. Each segment is either BF16 (`k`/`v` tensors resident) or
    PRQ-packed (`packed_k`/`packed_v` dicts resident, BF16 freed).
  * The current (still-denoising) chunk and the newest ``keep_recent_chunks``
    completed chunks stay BF16 (rewritten each denoise step / attended cleanly).
  * Older completed segments are PRQ-packed once and their BF16 freed -> the
    resident footprint drops to ~(sink + recent) BF16 + packed tail.
  * On read the visible window is reconstructed densely on the fly (dequantize
    packed segments + cat BF16 ones) and returned to attention; that transient
    dense tensor is freed after the layer's attention, so only ONE layer is
    dense at a time vs. all ``num_layers`` resident dense windows before.

Scope: the LingBot realtime causal path only (sliding window + sink, chunk-
aligned writes, optional ulysses head-slice, ``recent_window_tokens`` None or a
non-negative int). Unsupported base-class features raise NotImplementedError
rather than silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@cache
def _qvg_functions():
    try:
        from quant_videogen.functions import (
            triton_prq_dequantize_tensor,
            triton_prq_quantize_tensor,
        )
    except ImportError as e:
        raise ImportError(
            "Quant-VideoGen KV-cache quantization requires its optional "
            "runtime dependencies. Install them with: "
            "pip install 'sglang[diffusion-qvg]' && "
            "pip install --no-deps quant-videogen==0.1.0."
        ) from e
    return triton_prq_quantize_tensor, triton_prq_dequantize_tensor


@dataclass
class _Segment:
    g0: int  # global start token (inclusive)
    g1: int  # global end token (exclusive)
    is_sink: bool  # sink segments are never evicted
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    packed_k: dict | None = None
    packed_v: dict | None = None

    @property
    def packed(self) -> bool:
        return self.packed_k is not None

    def nbytes(self) -> int:
        if self.packed:
            return _packed_nbytes(self.packed_k) + _packed_nbytes(self.packed_v)
        return (
            self.k.numel() * self.k.element_size()
            + self.v.numel() * self.v.element_size()
        )


def _packed_nbytes(packed: dict) -> int:
    total = 0
    for key in ("centroids_list", "cluster_ids_list"):
        for t in packed.get(key) or []:
            total += t.numel() * t.element_size()
    for key in ("residual_quant", "scales", "zeros", "residual", "scale_factor"):
        t = packed.get(key)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


class QVGPackedCausalKVCache:
    """Chunk-segmented causal KV cache with PRQ-packed cold segments."""

    def __init__(
        self,
        *,
        batch_size: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        global_end_index: torch.Tensor,
        local_end_index: torch.Tensor,
        use_int_indices: bool = False,
        sink_tokens: int = 0,
        attention_window_size: int | None = None,
        quant_args: QVGKVQuantArgs,
    ) -> None:
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.sink_tokens = sink_tokens
        self.global_sink_tokens = 0
        self.attention_window_size = attention_window_size or cache_size
        self.q = quant_args
        # kept for API compatibility with the dense cache (indices are not read
        # by consumers, but reset/patterns touch them)
        self.global_end_index = global_end_index
        self.local_end_index = local_end_index
        self.global_end_index_int = 0 if use_int_indices else None
        self.local_end_index_int = 0 if use_int_indices else None

        self._segments: list[_Segment] = []  # completed, global-ordered
        self._cur: _Segment | None = None  # current (mutable) chunk
        self._global_end = 0
        self._chunk_tokens = 0  # inferred from first advance

    # ------------------------------------------------------------------ api
    def reset_indices(self) -> None:
        self._segments = []
        self._cur = None
        self._global_end = 0
        if self.global_end_index_int is not None:
            self.global_end_index_int = 0
            self.local_end_index_int = 0
        self.global_end_index.zero_()
        self.local_end_index.zero_()

    def can_direct_current_attention(self, num_new_tokens: int) -> bool:
        return (
            self.sink_tokens == 0
            and self.cache_size == num_new_tokens
            and self.attention_window_size == num_new_tokens
        )

    @property
    def num_cache_heads(self) -> int:
        return self.num_heads

    def pin_current_chunk(self, current_num_tokens: int) -> None:
        raise NotImplementedError(
            "QVGPackedCausalKVCache does not support pinned-sink (longlive2); "
            "packed KV quant is scoped to the LingBot realtime path."
        )

    def resident_nbytes(self) -> int:
        total = sum(s.nbytes() for s in self._segments)
        if self._cur is not None:
            total += self._cur.nbytes()
        return total

    # -------------------------------------------------------------- helpers
    def _new_bf16_segment(self, g0: int, g1: int, is_sink: bool) -> _Segment:
        n = g1 - g0
        k = torch.zeros(
            self.batch_size,
            n,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v = torch.zeros_like(k)
        return _Segment(g0=g0, g1=g1, is_sink=is_sink, k=k, v=v)

    def _write(self, seg: _Segment, key, value, head_slice) -> None:
        if head_slice is None:
            seg.k.copy_(key)
            seg.v.copy_(value)
        else:
            seg.k[:, :, head_slice, :] = key
            seg.v[:, :, head_slice, :] = value

    def _pack(self, seg: _Segment) -> None:
        if seg.packed or seg.k is None:
            return
        triton_prq_quantize_tensor, _ = _qvg_functions()

        def q(x):
            xb = x.permute(0, 2, 1, 3).contiguous()  # [B,S,H,D]->[B,H,S,D]
            devices = [xb.device] if xb.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(1234)
                return triton_prq_quantize_tensor(
                    xb,
                    num_stages=self.q.stages,
                    num_clusters=self.q.centroids,
                    block_size=self.q.block_size,
                    max_iters=self.q.kmeans_iters,
                    quantize_fn=lambda _t: self.q.bits,
                    asymmetric=self.q.asymmetric,
                )

        seg.packed_k = q(seg.k)
        seg.packed_v = q(seg.v)
        seg.k = None
        seg.v = None
        logger.info_once(f"Using QVG packed KV cache: {self.q.describe()}")

    def _dequant(self, packed: dict) -> torch.Tensor:
        _, triton_prq_dequantize_tensor = _qvg_functions()
        return triton_prq_dequantize_tensor(
            packed, self.q.block_size, self.q.bits, output_dtype=self.dtype
        )  # [B,H,S,D]

    def _all_segments(self) -> list[_Segment]:
        segs = list(self._segments)
        if self._cur is not None:
            segs.append(self._cur)
        return segs

    def _sink_end(self) -> int:
        return min(self.sink_tokens, self._global_end)

    def _tail_start(self) -> int:
        """Global start of the rolling recent tail (sink occupies its own
        budget at the window front, matching the dense cache's roll)."""
        sink_end = self._sink_end()
        recent_budget = max(0, self.attention_window_size - sink_end)
        return max(sink_end, self._global_end - recent_budget)

    def _pack_and_evict(self) -> None:
        """Pack completed segments older than the recency guard; drop segments
        that have slid entirely out of the window (sink is never evicted)."""
        tail_start = self._tail_start()

        # eviction: drop non-sink segments fully left of the rolling tail
        kept = []
        for s in self._segments:
            if not s.is_sink and s.g1 <= tail_start:
                continue
            kept.append(s)
        self._segments = kept

        if not self.q.enabled:
            return
        # recency guard: keep the newest `keep_recent_chunks` completed
        # non-sink chunks in bf16; pack everything older.
        recent = self.q.keep_recent_chunks
        nonsink = [s for s in self._segments if not s.is_sink]
        cutoff_idx = len(nonsink) - recent
        for i, s in enumerate(nonsink):
            if i < cutoff_idx:
                self._pack(s)
        # sink packing policy
        if self.q.sink:
            sink_keep_tokens = self.q.sink_keep_chunks * max(1, self._chunk_tokens)
            for s in self._segments:
                if s.is_sink and s.g0 >= sink_keep_tokens:
                    # only pack sink chunks past the protected prefix, and only
                    # once they are no longer the current recency-recent region
                    self._pack(s)

    # -------------------------------------------------------------- contract
    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        cache_head_start: int | None = None,
        recent_window_tokens: int | None = None,
        debug_name: str = "QVG packed KV cache",
    ) -> CausalAttentionKVView:
        num_new = key.shape[1]
        num_input_heads = key.shape[2]
        head_slice = None
        if num_input_heads != self.num_heads:
            if cache_head_start is None:
                raise ValueError(
                    f"{debug_name}: cache_head_start required for head slice"
                )
            head_slice = slice(cache_head_start, cache_head_start + num_input_heads)
        cend = current_chunk_start + num_new

        if self._cur is not None and current_chunk_start == self._cur.g0:
            # rewrite current chunk in place (denoise step)
            if cend != self._cur.g1:
                raise NotImplementedError(
                    f"{debug_name}: current-chunk rewrite size changed"
                )
            self._write(self._cur, key, value, head_slice)
        elif current_chunk_start == self._global_end:
            # advance: finalize current chunk, start a new one
            if self._cur is not None:
                self._segments.append(self._cur)
            is_sink = current_chunk_start < self.sink_tokens
            if self._chunk_tokens == 0:
                self._chunk_tokens = num_new
            self._cur = self._new_bf16_segment(current_chunk_start, cend, is_sink)
            self._write(self._cur, key, value, head_slice)
            self._global_end = cend
            self._pack_and_evict()
        else:
            raise NotImplementedError(
                f"{debug_name}: non-sequential write current_start="
                f"{current_chunk_start} global_end={self._global_end} "
                f"cur={None if self._cur is None else self._cur.g0}"
            )

        local_end = min(self._global_end, self.cache_size)
        if self.global_end_index_int is not None:
            self.global_end_index_int = self._global_end
            self.local_end_index_int = local_end
        else:
            self.global_end_index.fill_(self._global_end)
            self.local_end_index.fill_(local_end)

        vk, vv = self._reconstruct(current_chunk_start, recent_window_tokens)
        return CausalAttentionKVView(
            k=vk,
            v=vv,
            local_start_index=0,
            local_end_index=num_new,
            visible_local_end=min(self._global_end, self.cache_size),
            visible_global_end=self._global_end,
        )

    def _reconstruct(
        self,
        current_chunk_start: int,
        recent_window_tokens: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense visible window = sink prefix ++ rolling recent tail, matching
        the dense cache's [sink | rolled-recent] buffer content."""
        sink_end = self._sink_end()
        if recent_window_tokens is None:
            tail_start = self._tail_start()
        else:
            if recent_window_tokens < 0:
                raise ValueError("recent_window_tokens must be >= 0 or None")
            tail_start = max(sink_end, current_chunk_start - recent_window_tokens)

        if tail_start <= sink_end:
            ranges = [(0, self._global_end)]
        else:
            ranges = [(0, sink_end), (tail_start, self._global_end)]

        visible_segments: list[tuple[_Segment, int, int]] = []
        visible_tokens = 0
        for g_lo, g_hi in ranges:
            for seg in self._all_segments():
                a = max(g_lo, seg.g0)
                b = min(g_hi, seg.g1)
                if b <= a:
                    continue
                visible_segments.append((seg, a - seg.g0, b - seg.g0))
                visible_tokens += b - a

        if len(visible_segments) == 1 and not visible_segments[0][0].packed:
            seg, i0, i1 = visible_segments[0]
            return seg.k[:, i0:i1], seg.v[:, i0:i1]

        output_shape = (
            self.batch_size,
            visible_tokens,
            self.num_heads,
            self.head_dim,
        )
        vk = torch.empty(output_shape, dtype=self.dtype, device=self.device)
        vv = torch.empty_like(vk)
        output_start = 0

        # Dequantize one tensor at a time so reconstruction needs only the
        # final dense view plus one segment-sized temporary.
        for seg, i0, i1 in visible_segments:
            output_end = output_start + i1 - i0
            if seg.packed:
                dequantized = self._dequant(seg.packed_k)
                vk[:, output_start:output_end].copy_(
                    dequantized[:, :, i0:i1].permute(0, 2, 1, 3)
                )
                del dequantized

                dequantized = self._dequant(seg.packed_v)
                vv[:, output_start:output_end].copy_(
                    dequantized[:, :, i0:i1].permute(0, 2, 1, 3)
                )
                del dequantized
            else:
                vk[:, output_start:output_end].copy_(seg.k[:, i0:i1])
                vv[:, output_start:output_end].copy_(seg.v[:, i0:i1])
            output_start = output_end

        return vk, vv
