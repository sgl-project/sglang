# SPDX-License-Identifier: Apache-2.0
"""Hybrid window-softmax attention backend for VDN-H3 (MiniMax-H3 packed layout).

VDN-H3 replaces every DiT block's dense self-attention with two branches. The
SOFTMAX branch (this backend) is an exact softmax over a chunk-aligned frame
window: video frame t belongs to chunk t // chunk and attends to whole chunks
[c - radius, c + radius]; frames 0 and F-1 are "anchors" (dense as rows and
as columns under ``anchor_frames="both"``); every pair involving a text or
audio row stays dense in both directions; padding rows sit outside every mask.
A per-(token, head) sigmoid gate scales the softmax output. The LINEAR branch
(``runtime/models/dits/minimax_h3_vdn.py``) covers exactly the complement of
the window and is driven by the attention module, not by this backend.

The window is request-static, so the metadata (row-group plan, gather indices)
is built once per request in the MiniMax-H3 denoising stage and installed
through ``set_forward_context`` for every step and every block.

Kernel paths behind one backend (``--attention-backend-config
{"vdn_window_kernel": ...}``):

``decomposed``  (default; what VDN ships on SM100) the mask as a union of dense
                varlen FlashAttention calls: one call for the dense-query rows
                (text, audio, anchor frames) against all keys, one varlen call
                over per-chunk gathered [globals | window frames | anchors]
                K/V for the remaining frames. Same math as a masked kernel; it
                differs from it by bf16 reduction order only.
``tiles``       static-tile block-sparse Triton kernel (in-tree VSA-H3 tile-64
                kernel) with index lists from the window mask. See
                ``hybrid_window_h3_kernels.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
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
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import VDNH3Layout
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

HYBRID_WINDOW_H3_KERNELS = ("decomposed", "tiles")

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
    def get_impl_cls() -> type["HybridWindowAttentionH3Impl"]:
        return HybridWindowAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type["HybridWindowAttentionH3Metadata"]:
        return HybridWindowAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type["HybridWindowAttentionH3MetadataBuilder"]:
        return HybridWindowAttentionH3MetadataBuilder

    @classmethod
    def unsupported_requirements(
        cls, requirements: AttentionRequirements
    ) -> tuple[str, ...]:
        # Only MiniMax-H3's packed varlen attention is served here. Auxiliary
        # components (the Qwen3-VL conditioner, VAE attention) that ask for
        # plain dense attention must fall back to their default backend.
        if not requirements.packed_varlen:
            return ("dense (non-packed) attention",)
        return super().unsupported_requirements(requirements)


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


class _DecomposedPlan:
    """Query-row groups with identical kept key sets, as dense varlen calls.

    dense-q rows (globals + anchor-row frames) attend to all ``used`` keys in
    one call; every remaining chunk of frames attends to its gathered
    [globals | window frames | anchor-column frames] keys in one varlen call.
    """

    __slots__ = (
        "dense_q",
        "win_q",
        "kv_gather",
        "cu_q",
        "cu_k",
        "max_q",
        "max_k",
        "has_windows",
        "gathered_rows",
    )

    def __init__(
        self,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
    ) -> None:
        used = layout.used
        num_frames = layout.num_frames
        bounds, dense_rows, dense_cols = window_mask_frames(hybrid, num_frames)
        raw_bounds = hybrid.window_bounds(num_frames)

        def merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges: list[tuple[int, int]]) -> torch.Tensor:
            if not ranges:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.cat(
                [torch.arange(a, b, device=device, dtype=torch.long) for a, b in ranges]
            )

        global_ranges = layout.global_ranges
        dense_ranges = merge(
            global_ranges + [layout.frame_rows(f) for f in sorted(dense_rows)]
        )
        self.dense_q = cat_ranges(dense_ranges)

        groups: list[list[int]] = []
        for f in range(num_frames):
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

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(lo, hi + 1)) | dense_cols)
            qi = cat_ranges(merge([layout.frame_rows(f) for f in frames]))
            ki = cat_ranges(merge(global_ranges + [layout.frame_rows(f) for f in kv_frames]))
            q_idx.append(qi)
            kv_idx.append(ki)
            q_lens.append(int(qi.numel()))
            k_lens.append(int(ki.numel()))

        self.has_windows = bool(groups)
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = (
                torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(device, torch.int32)
            )
            self.cu_k = (
                torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(device, torch.int32)
            )
            self.max_q, self.max_k = max(q_lens), max(k_lens)
            self.gathered_rows = int(self.kv_gather.numel())
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)
            self.kv_gather = self.win_q
            self.cu_q = self.cu_k = None
            self.max_q = self.max_k = 0
            self.gathered_rows = 0

        covered = int(self.dense_q.numel()) + int(self.win_q.numel())
        if covered != used:
            raise ValueError(
                f"window decomposition covers {covered} of {used} packed rows"
            )


@dataclass
class HybridWindowAttentionH3Metadata(AttentionMetadata):
    layout: VDNH3Layout
    hybrid: VDNHybridAttentionArchConfig
    kernel: str
    # radius >= F: the window IS dense attention and the linear branch is off
    full_cover: bool
    decomposed: _DecomposedPlan | None = None
    tiles: Any = None
    # Full-sequence RoPE cache (cos_sin [seq_len, rope_dim] bf16, positions
    # [seq_len]) for QK-norm + RoPE after the Ulysses all-to-all; None when no
    # sequence parallelism is active.
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
        kernel: str = "decomposed",
        rope_cache_full: tuple[torch.Tensor, torch.Tensor] | None = None,
        current_timestep: int = 0,
        **kwargs: dict[str, Any],
    ) -> HybridWindowAttentionH3Metadata:
        if kernel not in HYBRID_WINDOW_H3_KERNELS:
            raise ValueError(
                f"vdn_window_kernel={kernel!r}; expected one of {HYBRID_WINDOW_H3_KERNELS}"
            )
        full_cover = hybrid.full_cover(layout.num_frames)
        decomposed = None
        tiles = None
        if not full_cover:
            if kernel == "decomposed":
                decomposed = _DecomposedPlan(layout, hybrid, device)
            else:
                from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_h3_kernels import (
                    build_window_tile_plan,
                )

                tiles = build_window_tile_plan(layout, hybrid, device)
        return HybridWindowAttentionH3Metadata(
            current_timestep=current_timestep,
            layout=layout,
            hybrid=hybrid,
            kernel=kernel,
            full_cover=full_cover,
            decomposed=decomposed,
            tiles=tiles,
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
) -> torch.Tensor:
    out = flash_attn_varlen_func(
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
    )
    return out[0] if isinstance(out, tuple) else out


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
        # The token refiner (text only, no frame axis) and any other caller
        # resolve the same backend object; they run the exact dense kernel.
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        # The platform resolver selects FA4 on Blackwell before the first
        # forward; direct constructions (tests, tools) go through the same gate.
        if current_platform.is_cuda():
            current_platform._prepare_flash_attention_for_blackwell()

        self._dense_fallback = FlashAttentionImpl(
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
        raise NotImplementedError(
            "hybrid_window_attn_h3 serves MiniMax-H3's packed varlen attention; "
            "use forward_varlen."
        )

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
        if self.layer_idx is None or attn_metadata is None:
            if attn_metadata is None and self.layer_idx is not None:
                raise RuntimeError(
                    "hybrid_window_attn_h3 needs per-request attention metadata "
                    "from the MiniMax-H3 denoising stage; none was set in the "
                    "forward context."
                )
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
        elif meta.kernel == "decomposed":
            out = self._decomposed(query, key, value, meta.decomposed, used)
        else:
            from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_h3_kernels import (
                window_tile_attention,
            )

            out = window_tile_attention(query, key, value, meta.tiles, used)

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
            n_dense = int(plan.dense_q.numel())
            cu_q = torch.tensor([0, n_dense], dtype=torch.int32, device=query.device)
            cu_k = torch.tensor([0, used], dtype=torch.int32, device=query.device)
            out[plan.dense_q] = _fa_varlen(
                query[plan.dense_q],
                key_used,
                value_used,
                cu_q=cu_q,
                cu_k=cu_k,
                max_q=n_dense,
                max_k=used,
                scale=self.softmax_scale,
            )
        if plan.has_windows:
            out[plan.win_q] = _fa_varlen(
                query[plan.win_q],
                key[plan.kv_gather],
                value[plan.kv_gather],
                cu_q=plan.cu_q,
                cu_k=plan.cu_k,
                max_q=plan.max_q,
                max_k=plan.max_k,
                scale=self.softmax_scale,
            )
        return out


__all__ = [
    "HYBRID_WINDOW_H3_KERNELS",
    "HybridWindowAttentionH3Backend",
    "HybridWindowAttentionH3Impl",
    "HybridWindowAttentionH3Metadata",
    "HybridWindowAttentionH3MetadataBuilder",
    "window_mask_frames",
    "window_mask_reference",
]
