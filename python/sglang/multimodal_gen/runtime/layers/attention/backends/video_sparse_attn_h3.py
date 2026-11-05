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

Ref2VA interleaves reference videos with dense rows. Each reference video is
its own 3-D tiled region; every video query keeps its own top-k of EACH
region (references at ``ref_keep_rate``, the generated video at the step
sparsity), FastVideo's "p2_multi_region" policy. Checkpoints trained on
128-token (4, 4, 8) tiles select and compress at that logical granularity;
each logical tile runs as two contiguous 64-token kernel tiles, so the
selected key set is unchanged.
"""

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.kernels.ops.diffusion import can_use_vsa_block_sparse_sm100
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
    vsa_h3_gate_add,
    vsa_h3_pack_tiles,
    vsa_h3_topk_lists,
    vsa_h3_untile,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# The kernels run 64-token blocks. FastH3 is trained on (4, 4, 4) tiles;
# FastH3 Omni on 128-token (4, 4, 8) tiles, each served as two kernel blocks.
VSA_H3_TILE_SHAPE = (4, 4, 4)
VSA_H3_TILE_ELEMS = math.prod(VSA_H3_TILE_SHAPE)
VSA_H3_TILE_SHAPES = {64: VSA_H3_TILE_SHAPE, 128: (4, 4, 8)}

_DIT_BLOCK_PREFIX = re.compile(r"^blocks\.(\d+)\.")

PackedSegment = int | tuple[int, int, int]


@functools.lru_cache(maxsize=8)
def _h3_segment_tile_geometry(
    segments: tuple[PackedSegment, ...],
    tile_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[tuple[int, int], ...]]:
    """Tile a packed sequence of interleaved dense segments and video regions.

    ``segments`` lists the packed order: an ``int`` is a dense segment's row
    count, a ``(t, h, w)`` triple one video region's token grid. Dense
    segments are chunked into the leading prefix tiles (never straddling a
    segment boundary); each region follows as a contiguous run of 3-D tiles.

    Returns, at kernel (64-token block) granularity, (variable_block_sizes
    int32 [n_blocks], pack_index int32 [S_pad]: padded position -> packed row
    or -1, unpack_index int32 [used]: packed row -> padded position), then
    num_prefix_tiles and the per-region [start, end) spans in logical tiles.
    A 128-token logical tile is two consecutive kernel blocks; its upper half
    is empty when the tile holds at most 64 rows.
    """
    tile_elems = math.prod(tile_shape)
    split = tile_elems // VSA_H3_TILE_ELEMS

    prefix_sizes: list[int] = []
    prefix_rows: list[torch.Tensor] = []
    video_entries: list[tuple[int, tuple[int, int, int]]] = []
    cursor = 0
    for segment in segments:
        if isinstance(segment, tuple):
            video_entries.append((cursor, segment))
            cursor += math.prod(segment)
            continue
        full, remainder = divmod(segment, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if remainder:
            prefix_sizes.append(remainder)
        prefix_rows.append(
            torch.arange(cursor, cursor + segment, device=device, dtype=torch.long)
        )
        cursor += segment
    num_prefix_tiles = len(prefix_sizes)

    video_rows: list[torch.Tensor] = []
    size_parts = [torch.tensor(prefix_sizes, dtype=torch.long, device=device)]
    spans: list[tuple[int, int]] = []
    tile_cursor = num_prefix_tiles
    for start, grid in video_entries:
        num_tiles = tuple(math.ceil(n / t) for n, t in zip(grid, tile_shape))
        sizes = construct_variable_block_sizes(grid, num_tiles, device, tile_shape)
        video_rows.append(get_tile_partition_indices(grid, tile_shape, device) + start)
        size_parts.append(sizes.to(torch.long))
        spans.append((tile_cursor, tile_cursor + int(sizes.numel())))
        tile_cursor += int(sizes.numel())

    tile_partition_indices = torch.cat(prefix_rows + video_rows)
    logical_sizes = torch.cat(size_parts)
    non_pad_index = get_non_pad_index(logical_sizes, tile_elems)
    if (
        int(logical_sizes.sum()) != cursor
        or non_pad_index.numel() != cursor
        or tile_partition_indices.numel() != cursor
    ):
        raise ValueError(
            f"VSA-H3 tile geometry mismatch for segments={segments}, "
            f"tile={tile_shape}: sizes sum {int(logical_sizes.sum())}, non-pad "
            f"{non_pad_index.numel()}, expected {cursor}."
        )

    # A logical tile's padded slots are its kernel blocks' slots in order,
    # so the pack/unpack maps are shared; only the per-block sizes split.
    seq_pad = logical_sizes.numel() * tile_elems
    pack_index = torch.full((seq_pad,), -1, dtype=torch.int32, device=device)
    pack_index[non_pad_index] = tile_partition_indices.to(torch.int32)
    unpack_index = torch.empty(cursor, dtype=torch.int32, device=device)
    unpack_index[tile_partition_indices] = non_pad_index.to(torch.int32)
    offsets = torch.arange(split, device=device) * VSA_H3_TILE_ELEMS
    block_sizes = (
        (logical_sizes[:, None] - offsets).clamp(0, VSA_H3_TILE_ELEMS).flatten()
    )
    return (
        block_sizes.to(torch.int32),
        pack_index,
        unpack_index,
        num_prefix_tiles,
        tuple(spans),
    )


def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Segment-pure prefix chunks, then the generated video's 64-token tiles.

    Returns (variable_block_sizes, pack_index, unpack_index,
    num_prefix_tiles, num_video_tiles).
    """
    sizes, pack_index, unpack_index, num_prefix_tiles, spans = (
        _h3_segment_tile_geometry(
            (*prefix_segments, tuple(dit_seq_shape)), VSA_H3_TILE_SHAPE, device
        )
    )
    return sizes, pack_index, unpack_index, num_prefix_tiles, spans[0][1] - spans[0][0]


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
    # Kernel (64-token block) granularity; a 128-token logical tile is two
    # consecutive blocks. Tile counts above are logical.
    variable_block_sizes: torch.Tensor
    pack_index: torch.Tensor
    unpack_index: torch.Tensor
    dense_layers: tuple[int, ...] = ()
    workspace_cache: dict = field(default_factory=dict)
    tile_elems: int = VSA_H3_TILE_ELEMS
    # Logical-tile [start, end) per video region, the generated video last,
    # with each region's sparsity. None: the generated video alone, at
    # VSA_sparsity.
    video_tile_spans: tuple[tuple[int, int], ...] | None = None
    span_sparsities: tuple[float, ...] | None = None

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def blocks_per_tile(self) -> int:
        return self.tile_elems // VSA_H3_TILE_ELEMS

    @property
    def uses_regions(self) -> bool:
        return self.video_tile_spans is not None


class VideoSparseAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        self._workspace_cache: dict = {}

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
        packed_segments: tuple[PackedSegment, ...] | None = None,
        tile_size: int = VSA_H3_TILE_ELEMS,
        ref_keep_rate: float | None = None,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionH3Metadata:
        """``packed_segments`` replaces ``prefix_segments`` for layouts with
        reference videos: the packed order of dense row counts and video
        token grids, ending with the generated video's grid. Every earlier
        grid is a reference region kept at ``ref_keep_rate``."""
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        tile_shape = VSA_H3_TILE_SHAPES.get(int(tile_size))
        if tile_shape is None:
            raise ValueError(
                f"VSA-H3 tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, "
                f"got {tile_size!r}"
            )
        if packed_segments is None:
            packed_segments = (*prefix_segments, dit_seq_shape)
        segments = tuple(
            tuple(int(v) for v in s) if isinstance(s, tuple) else int(s)
            for s in packed_segments
            if isinstance(s, tuple) or s > 0
        )
        if not segments or segments[-1] != dit_seq_shape:
            raise ValueError(
                f"VSA-H3 packed segments must end with the generated video "
                f"{dit_seq_shape}, got {segments}"
            )
        num_references = sum(isinstance(s, tuple) for s in segments) - 1
        if num_references:
            if ref_keep_rate is None or not 0.0 < float(ref_keep_rate) <= 1.0:
                raise ValueError(
                    f"VSA-H3 with {num_references} reference-video region(s) "
                    f"needs vsa_ref_keep_rate in (0, 1], got {ref_keep_rate!r}"
                )
            if not exempt:
                raise ValueError(
                    "vsa_mode='compete' supports only the generated-video "
                    "region; use vsa_mode='exempt' with reference videos"
                )
        if current_timestep < dense_first_n_steps:
            VSA_sparsity = 0.0

        (
            variable_block_sizes,
            pack_index,
            unpack_index,
            num_prefix_tiles,
            spans,
        ) = _h3_segment_tile_geometry(segments, tile_shape, device)

        regional = num_references > 0 or tile_shape != VSA_H3_TILE_SHAPE
        span_sparsities = None
        if regional:
            span_sparsities = (1.0 - float(ref_keep_rate or 1.0),) * num_references
            span_sparsities += (float(VSA_sparsity),)
        return VideoSparseAttentionH3Metadata(
            current_timestep=current_timestep,
            VSA_sparsity=float(VSA_sparsity),
            total_seq_length=sum(
                math.prod(s) if isinstance(s, tuple) else s for s in segments
            ),
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=spans[-1][1] - num_prefix_tiles,
            exempt=exempt,
            variable_block_sizes=variable_block_sizes,
            pack_index=pack_index,
            unpack_index=unpack_index,
            dense_layers=tuple(int(layer) for layer in dense_layers),
            workspace_cache=self._workspace_cache,
            tile_elems=math.prod(tile_shape),
            video_tile_spans=spans if regional else None,
            span_sparsities=span_sparsities,
        )


def _compute_topk(sparsity: float, num_video_tiles: int) -> int:
    keep = math.ceil((1.0 - sparsity) * num_video_tiles)
    return max(1, min(keep, num_video_tiles))


def _region_tile_lists(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    spans: tuple[tuple[int, int], ...],
    sparsities: tuple[float, ...],
) -> torch.Tensor:
    """scores [H, n_tiles, n_tiles] -> ascending int32 logical kv-tile lists
    [H, num_video_tiles, width]: every prefix tile, then each region's own
    top-k (the whole region when its budget covers it)."""
    prefix = num_prefix_tiles
    video_rows = scores[:, prefix:, :]
    parts = [
        torch.arange(prefix, device=scores.device).expand(
            *video_rows.shape[:-1], prefix
        )
    ]
    for (start, end), sparsity in zip(spans, sparsities):
        keep = _compute_topk(sparsity, end - start)
        if keep == end - start:
            parts.append(
                torch.arange(start, end, device=scores.device).expand(
                    *video_rows.shape[:-1], keep
                )
            )
            continue
        picked = video_rows[..., start:end].topk(keep, dim=-1).indices
        parts.append(picked.sort(dim=-1).values + start)
    return torch.cat(parts, dim=-1).to(torch.int32)


def _pool_logical_tiles(
    pooled: torch.Tensor, meta: VideoSparseAttentionH3Metadata
) -> torch.Tensor:
    """Kernel-block means [3, H, n_blocks, D] -> logical-tile means."""
    split = meta.blocks_per_tile
    if split == 1:
        return pooled
    sizes = meta.variable_block_sizes.view(-1, split).to(torch.float32)
    blocks = pooled.unflatten(2, (-1, split))
    # Empty upper halves pool to 0/0; they carry no weight.
    weighted = torch.where(sizes[..., None] > 0, blocks * sizes[..., None], 0.0)
    return weighted.sum(3) / sizes.sum(-1)[:, None]


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
    # serving takes vsa_h3_topk_lists for this branch; it stays as its reference
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
        meta.tile_elems,
        meta.uses_regions,
        heads,
        head_dim,
        has_gate,
        dtype,
        device,
    )


class _Workspace:
    """Per-geometry scratch: tiled q/k/v(/gate) [3|4, H, S_pad, D], pooled fp32
    block means [3, H, n_blocks, D], and the kernel index lists (prefix rows
    and prefix columns are static; only the top-k video columns change per
    layer). All of it is at kernel-block granularity. The native kernel pairs
    query blocks, so the block axis is padded to an even count with one empty
    block.
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
        n_tiles = int(meta.variable_block_sizes.numel())
        self.n_tiles = n_tiles
        self.use_sm100_kernel = can_use_vsa_block_sparse_sm100(
            device.index, dtype, head_dim
        )
        n_tiles_padded = n_tiles + (n_tiles % 2 if self.use_sm100_kernel else 0)
        seq_pad = n_tiles_padded * VSA_H3_TILE_ELEMS
        self.key = _workspace_key(meta, heads, head_dim, has_gate, dtype, device)
        self.tiled = torch.zeros(
            (3 + int(has_gate), heads, seq_pad, head_dim), dtype=dtype, device=device
        )
        self.pooled = torch.empty(
            (3, heads, n_tiles, head_dim), dtype=torch.float32, device=device
        )
        self.out_tiled = torch.empty(
            (heads, seq_pad, head_dim), dtype=dtype, device=device
        )
        self.block_sizes = torch.zeros(n_tiles_padded, dtype=torch.int32, device=device)
        self.block_sizes[:n_tiles] = meta.variable_block_sizes
        all_tiles = torch.arange(n_tiles, dtype=torch.int32, device=device)
        self.dense_index = torch.zeros(
            (heads, n_tiles_padded, n_tiles), dtype=torch.int32, device=device
        )
        self.dense_index[:, :n_tiles] = all_tiles
        self.dense_num = torch.zeros(
            (heads, n_tiles_padded), dtype=torch.int32, device=device
        )
        self.dense_num[:, :n_tiles] = n_tiles
        self.q2k_index = self.dense_index.clone()
        self.q2k_num = self.dense_num.clone()
        num_video_tiles = 0 if meta.uses_regions else meta.num_video_tiles
        # Zeroed scratch for the exempt top-k compaction; kept zero between uses.
        self.topk_mask = torch.zeros(
            (heads, num_video_tiles, num_video_tiles), dtype=torch.int8, device=device
        )

    def sparse_lists(
        self, video_lists: torch.Tensor, num_prefix_tiles: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = video_lists.shape[-1]
        self.q2k_index[:, num_prefix_tiles : self.n_tiles, :width] = video_lists
        self.q2k_num[:, num_prefix_tiles : self.n_tiles] = width
        return self.q2k_index, self.q2k_num


def _gate_tile_index(
    meta: VideoSparseAttentionH3Metadata, row_start: int, num_rows: int
) -> torch.Tensor:
    """Kernel block of each packed row in the shard, -1 for pad rows."""
    key = ("gate_tile_index", row_start, num_rows)
    tile_index = meta.workspace_cache.get(key)
    if tile_index is None:
        unpack_index = meta.unpack_index
        rows = torch.arange(row_start, row_start + num_rows, device=unpack_index.device)
        tile_index = torch.full_like(rows, -1, dtype=torch.int32)
        in_used = rows < meta.total_seq_length
        tile_index[in_used] = unpack_index[rows[in_used]] // VSA_H3_TILE_ELEMS
        meta.workspace_cache[key] = tile_index
    return tile_index


def vsa_h3_fold_gate(
    out: torch.Tensor,
    gate_compress: torch.Tensor,
    out_compress: torch.Tensor,
    meta: VideoSparseAttentionH3Metadata,
    row_start: int,
) -> None:
    """Fold the compression branch into ``out`` [rows, H, D], the row shard at
    ``row_start``."""
    vsa_h3_gate_add(
        out,
        gate_compress,
        out_compress,
        _gate_tile_index(meta, row_start, out.shape[0]),
    )


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
    if meta.uses_regions:
        return _select_region_kv_lists(ws, meta, scores, sparsity)
    keep = _compute_topk(sparsity, meta.num_video_tiles)
    if sparsity <= 0.0 or keep >= meta.num_video_tiles:
        return ws.dense_index, ws.dense_num
    prefix = meta.num_prefix_tiles
    if meta.exempt or prefix == 0:
        # Same lists as _topk_tile_lists without sorting values or indices.
        picked = scores[:, prefix:, prefix:].topk(keep, dim=-1, sorted=False).indices
        vsa_h3_topk_lists(
            picked, ws.topk_mask, ws.q2k_index[:, prefix : ws.n_tiles], prefix
        )
        ws.q2k_num[:, prefix : ws.n_tiles] = prefix + keep
        return ws.q2k_index, ws.q2k_num
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


def _select_region_kv_lists(
    ws: _Workspace,
    meta: VideoSparseAttentionH3Metadata,
    scores: torch.Tensor | None,
    sparsity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-region top-k over logical tiles, expanded to kernel blocks."""
    spans = meta.video_tile_spans
    sparsities = meta.span_sparsities
    # Dense steps and layers keep every region dense.
    if sparsity <= 0.0:
        return ws.dense_index, ws.dense_num
    if all(
        _compute_topk(s, end - start) == end - start
        for (start, end), s in zip(spans, sparsities)
    ):
        return ws.dense_index, ws.dense_num
    lists = _region_tile_lists(scores, meta.num_prefix_tiles, spans, sparsities)
    split = meta.blocks_per_tile
    if split > 1:
        offsets = torch.arange(split, device=lists.device, dtype=lists.dtype)
        lists = (lists[..., None] * split + offsets).flatten(-2)
        lists = lists.repeat_interleave(split, dim=1)
    prefix_blocks = meta.num_prefix_tiles * split
    width = lists.shape[-1]
    ws.q2k_index[:, prefix_blocks : ws.n_tiles, :width] = lists
    ws.q2k_num[:, prefix_blocks : ws.n_tiles] = width
    return ws.q2k_index, ws.q2k_num


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
            "VSA-H3 serves MiniMax-H3's packed varlen attention only; select it "
            "for the transformer with --component-attention-backends "
            "transformer=video_sparse_attn_h3 so dense layers keep their backends."
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
        return_compress: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """query/key/value: [T, H, D] packed rows (post-norm, post-RoPE).
        ``return_compress`` returns the compression branch's ``[H, n_tiles, D]``
        output instead of folding it, for ``vsa_h3_fold_gate`` after the
        caller's collectives."""
        assert not return_compress or (
            gate_compress is None and attn_metadata is not None
        ), "return_compress replaces gate_compress and needs VSA-H3 metadata"
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
        needs_compress = has_gate or return_compress
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
        q_pooled, k_pooled, v_pooled = _pool_logical_tiles(ws.pooled, meta)

        scores = None
        if sparsity > 0.0 or needs_compress:
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
            ws.block_sizes,
            out=ws.out_tiled[None],
            use_sm100_kernel=ws.use_sm100_kernel,
        )

        out_compress = None
        if needs_compress:
            out_compress = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)
            if meta.blocks_per_tile > 1:
                # The gate folds per kernel block; both halves of a logical
                # tile share its compressed row.
                out_compress = out_compress.repeat_interleave(
                    meta.blocks_per_tile, dim=1
                )

        result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        vsa_h3_untile(
            ws.out_tiled,
            ws.tiled[3] if has_gate else None,
            out_compress if has_gate else None,
            meta.unpack_index,
            used,
            result,
        )
        if return_compress:
            return result, out_compress
        return result
