# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 window-softmax backend on the MiniMax-H3 packed layout.

An exact softmax over a chunk-aligned frame window: frame t belongs to chunk
t // chunk and attends to chunks [c - radius, c + radius]; frames 0 and F-1
are dense anchors; text and audio rows are dense both ways; padding rows sit
outside every mask; a per-(token, head) sigmoid gate scales the output. The
linear branch (``minimax_h3_vdn.py``) covers the window's complement. The
metadata is request-static and installed once per request through
``set_forward_context``. The window runs as a union of dense varlen
FlashAttention calls: the dense-query rows against all keys, then per-chunk
gathered [globals | window | anchors] K/V; same math as a masked kernel up to
bf16 reduction order.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.configs.models.dits.minimax_h3_vdn import (
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    flash_attn as _flash_attn_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import VDNH3Layout
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_DIT_BLOCK_PREFIX = re.compile(r"^blocks\.(\d+)\.")


class HybridWindowAttentionH3Backend(AttentionBackend):
    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3

    @staticmethod
    def get_impl_cls() -> type[HybridWindowAttentionH3Impl]:
        return HybridWindowAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type[HybridWindowAttentionH3Metadata]:
        return HybridWindowAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type[HybridWindowAttentionH3MetadataBuilder]:
        return HybridWindowAttentionH3MetadataBuilder


def window_mask_frames(
    hybrid: VDNHybridAttentionArchConfig, num_frames: int
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """(clamped per-frame window bounds, dense-ROW frames, dense-COLUMN frames)."""
    bounds = [
        (max(lo, 0), min(hi, num_frames - 1))
        for lo, hi in hybrid.window_bounds(num_frames)
    ]
    anchors = {0, num_frames - 1} if hybrid.anchor_frames != "none" else set()
    dense_rows = anchors if hybrid.anchor_frames in ("rows", "both") else set()
    dense_cols = anchors if hybrid.anchor_frames in ("columns", "both") else set()
    return bounds, dense_rows, dense_cols


def window_mask_reference(
    hybrid: VDNHybridAttentionArchConfig, layout: VDNH3Layout, device: torch.device
) -> torch.Tensor:
    """Dense boolean [used, used] mask of the softmax branch, for tests."""
    used = layout.used
    keep = torch.ones(used, used, dtype=torch.bool, device=device)
    vs, ve = layout.video_start, layout.video_end
    bounds, dense_rows, dense_cols = window_mask_frames(hybrid, layout.num_frames)
    tpf = layout.tokens_per_frame
    rows = torch.arange(vs, ve, device=device)
    frame_of = (rows - vs) // tpf
    qf = frame_of[:, None]
    kf = frame_of[None, :]
    lo = torch.tensor([b[0] for b in bounds], device=device)[qf]
    hi = torch.tensor([b[1] for b in bounds], device=device)[qf]
    inside = (kf >= lo) & (kf <= hi)
    for f in dense_rows:
        inside |= qf == f
    for f in dense_cols:
        inside |= kf == f
    keep[vs:ve, vs:ve] = inside
    return keep


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and out[-1][1] >= a:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _cat_ranges(ranges: list[tuple[int, int]], *, device: torch.device) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(
        [torch.arange(a, b, device=device, dtype=torch.long) for a, b in ranges]
    )


def _chunk_groups(
    raw_bounds: list[tuple[int, int]], dense_rows: set[int]
) -> list[list[int]]:
    # consecutive window frames with identical bounds share one varlen segment
    groups: list[list[int]] = []
    for f in range(len(raw_bounds)):
        if f in dense_rows:
            continue
        if (
            groups
            and raw_bounds[groups[-1][-1]] == raw_bounds[f]
            and groups[-1][-1] == f - 1
        ):
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


def _window_passes(
    layout: VDNH3Layout,
    per_group: list[tuple[list[int], torch.Tensor, torch.Tensor]],
    max_gather_rows: int,
    device: torch.device,
) -> list[dict]:
    # one varlen call per pass; consecutive chunk groups fill up to max_gather_rows
    passes: list[dict] = []
    current: list[tuple] = []
    current_rows = 0

    def flush() -> None:
        if not current:
            return
        q_idx = [g[1] for g in current]
        kv_idx = [g[2] for g in current]
        q_lens = [int(t.numel()) for t in q_idx]
        k_lens = [int(t.numel()) for t in kv_idx]
        flat = [f for g in current for f in g[0]]
        zero = torch.zeros(1, dtype=torch.long)
        contiguous = flat == list(range(flat[0], flat[0] + len(flat)))
        passes.append(
            dict(
                win_q=torch.cat(q_idx),
                # contiguous frames -> slice instead of gather/scatter
                win_q_slice=(
                    (layout.frame_rows(flat[0])[0], layout.frame_rows(flat[-1])[1])
                    if contiguous
                    else None
                ),
                kv_gather=torch.cat(kv_idx),
                cu_q=torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(
                    device, torch.int32
                ),
                cu_k=torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(
                    device, torch.int32
                ),
                max_q=max(q_lens),
                max_k=max(k_lens),
            )
        )

    for frames, qi, ki in per_group:
        rows = int(ki.numel())
        if current and current_rows + rows > max_gather_rows:
            flush()
            current, current_rows = [], 0
        current.append((frames, qi, ki))
        current_rows += rows
    flush()
    return passes


class _DecomposedPlan:
    """Query-row groups with identical kept key sets, as dense varlen calls:
    the dense-q rows against all ``used`` keys, then each chunk of frames
    against its gathered [globals | window | anchors] keys."""

    __slots__ = ("dense_q", "dense_cu_q", "dense_cu_k", "passes", "has_windows")

    def __init__(
        self,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
        max_gather_rows: int = 200_000,
    ) -> None:
        used, num_frames = layout.used, layout.num_frames
        bounds, dense_rows, dense_cols = window_mask_frames(hybrid, num_frames)
        rows = functools.partial(_cat_ranges, device=device)
        dense_ranges = _merge_ranges(
            layout.global_ranges + [layout.frame_rows(f) for f in sorted(dense_rows)]
        )
        self.dense_q = rows(dense_ranges)
        # built once: a tensor from a Python list costs a pageable H2D copy + sync
        self.dense_cu_q = torch.tensor(
            [0, int(self.dense_q.numel())], dtype=torch.int32, device=device
        )
        self.dense_cu_k = torch.tensor([0, used], dtype=torch.int32, device=device)
        per_group = []
        for frames in _chunk_groups(hybrid.window_bounds(num_frames), dense_rows):
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(lo, hi + 1)) | dense_cols)
            per_group.append(
                (
                    frames,
                    rows(_merge_ranges([layout.frame_rows(f) for f in frames])),
                    rows(
                        _merge_ranges(
                            layout.global_ranges
                            + [layout.frame_rows(f) for f in kv_frames]
                        )
                    ),
                )
            )
        self.passes = _window_passes(layout, per_group, max_gather_rows, device)
        self.has_windows = bool(self.passes)
        win_rows = sum(int(p["win_q"].numel()) for p in self.passes)
        covered = int(self.dense_q.numel()) + win_rows
        if covered != used:
            raise ValueError(
                f"window decomposition covers {covered} of {used} packed rows"
            )


@dataclass
class HybridWindowAttentionH3Metadata(AttentionMetadata):
    layout: VDNH3Layout
    hybrid: VDNHybridAttentionArchConfig
    # radius >= F: the window IS dense attention and the linear branch is off
    full_cover: bool
    decomposed: _DecomposedPlan | None = None
    # (cos_sin [seq_len, rope_dim] bf16, positions [seq_len]) under Ulysses, else None
    rope_cache_full: tuple[torch.Tensor, torch.Tensor] | None = None


class HybridWindowAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        *,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
        rope_cache_full: tuple[torch.Tensor, torch.Tensor] | None = None,
        current_timestep: int = 0,
        max_gather_rows: int = 200_000,
        **kwargs: dict[str, Any],
    ) -> HybridWindowAttentionH3Metadata:
        full_cover = hybrid.full_cover(layout.num_frames)
        decomposed = None
        if not full_cover:
            decomposed = _DecomposedPlan(
                layout, hybrid, device, max_gather_rows=max_gather_rows
            )
        return HybridWindowAttentionH3Metadata(
            current_timestep=current_timestep,
            layout=layout,
            hybrid=hybrid,
            full_cover=full_cover,
            decomposed=decomposed,
            rope_cache_full=rope_cache_full,
        )


def _fa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    result = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=scale,
        causal=False,
        ver=_flash_attn_backend.fa_ver,
        out=out,
    )
    result = result[0] if isinstance(result, tuple) else result
    if out is not None and result.data_ptr() != out.data_ptr():
        out.copy_(result)
        return out
    return result


class HybridWindowAttentionH3Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        match = _DIT_BLOCK_PREFIX.match(prefix)
        self.layer_idx = int(match.group(1)) if match else None
        # non-DiT callers (the token refiner) resolve this backend too: dense FA
        self._dense_fallback = _flash_attn_backend.FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Dense FlashAttention for the non-DiT layers this backend reaches."""
        return self._dense_fallback.forward(query, key, value, attn_metadata)

    def dense_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        return self._dense_fallback.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        attn_metadata: HybridWindowAttentionH3Metadata | None = None,
        softmax_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query/key/value: [T, H, D] packed rows (post-norm, post-RoPE) ->
        [T, H, D]; ``softmax_gate`` [T, H] scales the output per (row, head).
        Rows at and past ``used`` (padding) are zero."""
        if self.layer_idx is not None and attn_metadata is None:
            raise RuntimeError(
                "hybrid_window_attn_h3 needs per-request attention metadata "
                "from the MiniMax-H3 denoising stage; none was set in the "
                "forward context."
            )
        if self.layer_idx is None:
            return self.dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        meta = attn_metadata
        layout = meta.layout
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        used = int(bounds[1])
        if used != layout.used or query.shape[0] != layout.seq_len:
            raise ValueError(
                f"hybrid_window_attn_h3 metadata was built for used={layout.used} "
                f"of seq_len={layout.seq_len} rows, got used={used} of "
                f"{query.shape[0]}. The request metadata and the packed layout "
                "have diverged."
            )

        if meta.full_cover:
            out = self.dense_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        else:
            out = self._decomposed(query, key, value, meta.decomposed, used)

        if softmax_gate is not None:
            out.mul_(softmax_gate.to(out.dtype).unsqueeze(-1))
        if used < out.shape[0]:
            out[used:].zero_()
        return out

    def _decomposed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: _DecomposedPlan,
        used: int,
    ) -> torch.Tensor:
        out = torch.empty_like(query)
        key_used = key[:used]
        value_used = value[:used]
        if not key_used.is_contiguous():
            key_used = key_used.contiguous()
        if not value_used.is_contiguous():
            value_used = value_used.contiguous()
        if plan.dense_q.numel():
            out[plan.dense_q] = _fa_varlen(
                torch.index_select(query, 0, plan.dense_q),
                key_used,
                value_used,
                cu_q=plan.dense_cu_q,
                cu_k=plan.dense_cu_k,
                max_q=int(plan.dense_q.numel()),
                max_k=used,
                scale=self.softmax_scale,
            )
        for p in plan.passes:
            # index_select on contiguous copies takes the vectorized gather kernel
            kw = torch.index_select(key_used, 0, p["kv_gather"])
            vw = torch.index_select(value_used, 0, p["kv_gather"])
            if p["win_q_slice"] is not None:
                start, stop = p["win_q_slice"]
                _fa_varlen(
                    query[start:stop],
                    kw,
                    vw,
                    cu_q=p["cu_q"],
                    cu_k=p["cu_k"],
                    max_q=p["max_q"],
                    max_k=p["max_k"],
                    scale=self.softmax_scale,
                    out=out[start:stop],
                )
            else:
                out[p["win_q"]] = _fa_varlen(
                    torch.index_select(query, 0, p["win_q"]),
                    kw,
                    vw,
                    cu_q=p["cu_q"],
                    cu_k=p["cu_k"],
                    max_q=p["max_q"],
                    max_k=p["max_k"],
                    scale=self.softmax_scale,
                )
            del kw, vw
        return out


__all__ = [
    "HybridWindowAttentionH3Backend",
    "HybridWindowAttentionH3Impl",
    "HybridWindowAttentionH3Metadata",
    "HybridWindowAttentionH3MetadataBuilder",
    "window_mask_frames",
    "window_mask_reference",
]
