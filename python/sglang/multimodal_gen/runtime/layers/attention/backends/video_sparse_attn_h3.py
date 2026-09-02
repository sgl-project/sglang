# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# (the video_sparse_attn_h3 backend), rewritten for SGLang's packed-varlen
# MiniMax-H3 attention contract.

# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax-H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``. Tiles are
``[segment-pure prefix chunks] + [3D video tiles]``; prefix tiles never
straddle segment boundaries. Selection is a top-k over pooled tile scores
emitted directly as the per-query-tile index lists the vendored Triton
tile-64 kernel consumes, with per-tile valid sizes so ragged interior tiles
mask exactly.

Non-video queries are always dense. Non-video keys are either
always-selected for every query ("exempt", default) or compete in top-k
under a FLOP-matched budget ("compete"). The compression branch is gated by
``to_gate_compress``: base H3 has no such weights and the gate loads as
zeros (pure sparse); VSA-distilled students (FastH3) ship trained gates
that activate the branch.
"""

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    construct_variable_block_sizes,
    get_non_pad_index,
    get_tile_partition_indices,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (
    vsa_h3_block_sparse_attn_forward,
    vsa_h3_pack_tiles,
    vsa_h3_untile,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# The FastH3 checkpoints are trained and served at the 64-token (4, 4, 4)
# tile geometry; the kernel block size matches it exactly.
VSA_H3_TILE_SHAPE = (4, 4, 4)
VSA_H3_TILE_ELEMS = math.prod(VSA_H3_TILE_SHAPE)

_DIT_BLOCK_PREFIX = re.compile(r"^blocks\.(\d+)\.")


@functools.lru_cache(maxsize=8)
def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Segment-pure prefix chunks, then video tiles.

    Returns (variable_block_sizes int32 [n_tiles], pack_index int32 [S_pad]:
    padded position -> packed row or -1, unpack_index int32 [used]: packed row
    -> padded position, num_prefix_tiles, num_video_tiles).
    """
    prefix_len = sum(prefix_segments)

    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, VSA_H3_TILE_ELEMS)
        prefix_sizes.extend([VSA_H3_TILE_ELEMS] * full)
        if remainder:
            prefix_sizes.append(remainder)
    num_prefix_tiles = len(prefix_sizes)

    num_tiles = (
        math.ceil(dit_seq_shape[0] / VSA_H3_TILE_SHAPE[0]),
        math.ceil(dit_seq_shape[1] / VSA_H3_TILE_SHAPE[1]),
        math.ceil(dit_seq_shape[2] / VSA_H3_TILE_SHAPE[2]),
    )
    video_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, device)
    num_video_tiles = int(video_sizes.numel())

    video_indices = (
        get_tile_partition_indices(dit_seq_shape, VSA_H3_TILE_SHAPE, device)
        + prefix_len
    )
    tile_partition_indices = torch.cat(
        [
            torch.arange(prefix_len, device=device, dtype=torch.long),
            video_indices,
        ]
    )
    variable_block_sizes = torch.cat(
        [
            torch.tensor(prefix_sizes, dtype=torch.long, device=device),
            video_sizes.to(torch.long),
        ]
    )
    non_pad_index = get_non_pad_index(variable_block_sizes, VSA_H3_TILE_ELEMS)

    total = prefix_len + math.prod(dit_seq_shape)
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_sum != total or non_pad_index.numel() != total:
        raise ValueError(
            f"VSA-H3 tile geometry mismatch for prefix={prefix_segments}, "
            f"video={dit_seq_shape}: sizes sum {sizes_sum}, non-pad "
            f"{non_pad_index.numel()}, expected {total}."
        )

    seq_pad = variable_block_sizes.numel() * VSA_H3_TILE_ELEMS
    pack_index = torch.full((seq_pad,), -1, dtype=torch.int32, device=device)
    pack_index[non_pad_index] = tile_partition_indices.to(torch.int32)
    unpack_index = torch.empty(total, dtype=torch.int32, device=device)
    unpack_index[tile_partition_indices] = non_pad_index.to(torch.int32)
    return (
        variable_block_sizes.to(torch.int32),
        pack_index,
        unpack_index,
        num_prefix_tiles,
        num_video_tiles,
    )


class VideoSparseAttentionH3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionH3Impl"]:
        return VideoSparseAttentionH3Impl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionH3Metadata"]:
        return VideoSparseAttentionH3Metadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionH3MetadataBuilder"]:
        return VideoSparseAttentionH3MetadataBuilder


@dataclass
class VideoSparseAttentionH3Metadata(AttentionMetadata):
    VSA_sparsity: float
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    exempt: bool
    variable_block_sizes: torch.Tensor
    pack_index: torch.Tensor
    unpack_index: torch.Tensor
    dense_layers: tuple[int, ...] = ()
    workspace_cache: dict = field(default_factory=dict)

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles


class VideoSparseAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        self._workspace_cache: dict = {}

    def prepare(self) -> None:
        pass

    def build(  # type: ignore[override]
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        prefix_segments: tuple[int, ...],
        device: torch.device,
        exempt: bool = True,
        dense_layers: tuple[int, ...] = (),
        dense_first_n_steps: int = 0,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionH3Metadata:
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        prefix_segments = tuple(int(s) for s in prefix_segments if s > 0)
        if current_timestep < dense_first_n_steps:
            VSA_sparsity = 0.0

        (
            variable_block_sizes,
            pack_index,
            unpack_index,
            num_prefix_tiles,
            num_video_tiles,
        ) = _h3_tile_geometry(prefix_segments, dit_seq_shape, device)

        return VideoSparseAttentionH3Metadata(
            current_timestep=current_timestep,
            VSA_sparsity=float(VSA_sparsity),
            total_seq_length=sum(prefix_segments) + math.prod(dit_seq_shape),
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            pack_index=pack_index,
            unpack_index=unpack_index,
            dense_layers=tuple(int(layer) for layer in dense_layers),
            workspace_cache=self._workspace_cache,
        )


def _compute_topk(sparsity: float, num_video_tiles: int) -> int:
    keep = math.ceil((1.0 - sparsity) * num_video_tiles)
    return max(1, min(keep, num_video_tiles))


def _topk_tile_lists(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
) -> torch.Tensor:
    """scores [H, n_tiles, n_tiles] -> ascending int32 kv-tile lists
    [H, num_video_tiles, width]; width = num_prefix_tiles + keep (exempt) or
    min(keep + num_prefix_tiles, n_tiles) (compete)."""
    prefix = num_prefix_tiles
    keep = _compute_topk(sparsity, num_video_tiles)
    video_rows = scores[:, prefix:, :]
    if exempt or prefix == 0:
        picked = video_rows[:, :, prefix:].topk(keep, dim=-1).indices + prefix
        picked = picked.sort(dim=-1).values
        if prefix == 0:
            return picked.to(torch.int32)
        prefix_cols = torch.arange(prefix, device=scores.device).expand(
            *picked.shape[:-1], prefix
        )
        return torch.cat([prefix_cols, picked], dim=-1).to(torch.int32)
    keep_total = min(keep + prefix, scores.shape[-1])
    return (
        video_rows.topk(keep_total, dim=-1).indices.sort(dim=-1).values.to(torch.int32)
    )


def _workspace_key(
    meta: VideoSparseAttentionH3Metadata,
    heads: int,
    head_dim: int,
    has_gate: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    return (
        meta.num_tiles,
        meta.num_prefix_tiles,
        meta.exempt,
        heads,
        head_dim,
        has_gate,
        dtype,
        device,
    )


class _Workspace:
    """Per-geometry scratch: tiled q/k/v(/gate) [3|4, H, S_pad, D], pooled fp32
    tile means [3, H, n_tiles, D], and the kernel index lists (prefix rows and
    prefix columns are static; only the top-k video columns change per layer).
    """

    def __init__(
        self,
        meta: VideoSparseAttentionH3Metadata,
        heads: int,
        head_dim: int,
        has_gate: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        n_tiles = meta.num_tiles
        seq_pad = n_tiles * VSA_H3_TILE_ELEMS
        self.key = _workspace_key(meta, heads, head_dim, has_gate, dtype, device)
        self.tiled = torch.empty(
            (3 + int(has_gate), heads, seq_pad, head_dim), dtype=dtype, device=device
        )
        self.pooled = torch.empty(
            (3, heads, n_tiles, head_dim), dtype=torch.float32, device=device
        )
        self.out_tiled = torch.empty(
            (heads, seq_pad, head_dim), dtype=dtype, device=device
        )
        all_tiles = torch.arange(n_tiles, dtype=torch.int32, device=device)
        self.dense_index = all_tiles.repeat(heads, n_tiles, 1)
        self.dense_num = torch.full(
            (heads, n_tiles), n_tiles, dtype=torch.int32, device=device
        )
        self.q2k_index = self.dense_index.clone()
        self.q2k_num = self.dense_num.clone()

    def sparse_lists(
        self, video_lists: torch.Tensor, num_prefix_tiles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = video_lists.shape[-1]
        self.q2k_index[:, num_prefix_tiles:, :width] = video_lists
        self.q2k_num[:, num_prefix_tiles:] = width
        return self.q2k_index, self.q2k_num


def _get_workspace(
    meta: VideoSparseAttentionH3Metadata, query: torch.Tensor, has_gate: bool
) -> _Workspace:
    heads, head_dim = query.shape[-2], query.shape[-1]
    key = _workspace_key(meta, heads, head_dim, has_gate, query.dtype, query.device)
    workspace = meta.workspace_cache.get("workspace")
    if workspace is None or workspace.key != key:
        workspace = _Workspace(
            meta, heads, head_dim, has_gate, query.dtype, query.device
        )
        meta.workspace_cache["workspace"] = workspace
    return workspace


def _select_kv_lists(
    ws: _Workspace,
    meta: VideoSparseAttentionH3Metadata,
    scores: torch.Tensor | None,
    sparsity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel index lists for this layer: dense, or top-k video columns."""
    keep = _compute_topk(sparsity, meta.num_video_tiles)
    if sparsity <= 0.0 or keep >= meta.num_video_tiles:
        return ws.dense_index, ws.dense_num
    return ws.sparse_lists(
        _topk_tile_lists(
            scores,
            meta.num_prefix_tiles,
            meta.num_video_tiles,
            sparsity,
            meta.exempt,
        ),
        meta.num_prefix_tiles,
    )


class VideoSparseAttentionH3Impl(AttentionImpl):
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
        # The token refiner and any other non-packed caller resolve the same
        # backend object; they run the exact dense kernel instead.
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
        )

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
            "VSA-H3 serves MiniMax-H3's packed varlen attention; use " "forward_varlen."
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
        attn_metadata: VideoSparseAttentionH3Metadata | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query/key/value: [T, H, D] packed rows (post-norm, post-RoPE)."""
        if self.layer_idx is None or attn_metadata is None:
            if attn_metadata is None and self.layer_idx is not None:
                raise RuntimeError(
                    "VSA-H3 needs per-step attention metadata from the "
                    "MiniMax-H3 denoising stage; none was set in the forward "
                    "context."
                )
            return self._dense_fallback.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        meta = attn_metadata
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        used = int(bounds[1])
        if used != meta.total_seq_length:
            raise ValueError(
                f"VSA-H3 metadata was built for {meta.total_seq_length} packed "
                f"rows, got {used}. The step metadata and the packed sequence "
                "layout have diverged."
            )

        sparsity = 0.0 if self.layer_idx in meta.dense_layers else meta.VSA_sparsity
        has_gate = gate_compress is not None
        ws = _get_workspace(meta, query, has_gate)

        vsa_h3_pack_tiles(
            query,
            key,
            value,
            gate_compress,
            meta.pack_index,
            meta.variable_block_sizes,
            ws.tiled,
            ws.pooled,
        )
        q_pooled, k_pooled, v_pooled = ws.pooled

        scores = None
        if sparsity > 0.0 or has_gate:
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * (
                self.head_size**-0.5
            )

        q2k_index, q2k_num = _select_kv_lists(ws, meta, scores, sparsity)
        vsa_h3_block_sparse_attn_forward(
            ws.tiled[0:1],
            ws.tiled[1:2],
            ws.tiled[2:3],
            q2k_index[None],
            q2k_num[None],
            meta.variable_block_sizes,
            out=ws.out_tiled[None],
        )

        out_compress = None
        if has_gate:
            out_compress = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)

        result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        vsa_h3_untile(
            ws.out_tiled,
            ws.tiled[3] if has_gate else None,
            out_compress,
            meta.unpack_index,
            used,
            result,
        )
        return result
