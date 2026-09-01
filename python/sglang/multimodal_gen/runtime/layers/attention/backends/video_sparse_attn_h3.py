# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# (the video_sparse_attn_h3 backend), rewritten for SGLang's packed-varlen
# MiniMax-H3 attention contract.

# SPDX-License-Identifier: Apache-2.0
"""VSA for MiniMax-H3's packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``. Tiles are
``[segment-pure prefix chunks] + [3D video tiles]``; prefix tiles never
straddle segment boundaries. Selection is pure Python over pooled tile
scores; the vendored Triton tile-64 kernel consumes an explicit index list
plus per-tile valid sizes, so ragged interior tiles mask exactly.

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
from dataclasses import dataclass
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
    vsa_h3_map_to_index,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

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
    """Tile the packed sequence: segment-pure prefix chunks, then video tiles.

    Returns (tile_partition_indices, variable_block_sizes,
    non_pad_index, num_prefix_tiles, num_video_tiles).
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
    return (
        tile_partition_indices,
        variable_block_sizes,
        non_pad_index,
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
    tile_partition_indices: torch.Tensor
    variable_block_sizes: torch.Tensor
    non_pad_index: torch.Tensor
    dense_layers: tuple[int, ...] = ()
    # Per-step padded tile scratch shared by all layers; pad rows stay zero
    # because only valid rows are ever written (same contract as Wan VSA).
    tile_buf: torch.Tensor | None = None


class VideoSparseAttentionH3MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

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
            tile_partition_indices,
            variable_block_sizes,
            non_pad_index,
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
            tile_partition_indices=tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
            dense_layers=tuple(int(layer) for layer in dense_layers),
        )


def _pool_tiles(x: torch.Tensor, variable_block_sizes: torch.Tensor) -> torch.Tensor:
    """fp32 masked mean per tile. x: [B, S_pad, H, D] -> [B, H, n_tiles, D].

    Pad positions in the tile buffer are zero (zeros-init, never written), so
    the sum divided by each tile's true size is the masked mean exactly.
    """
    batch, seq_len, heads, dim = x.shape
    n_tiles = seq_len // VSA_H3_TILE_ELEMS
    pooled = x.view(batch, n_tiles, VSA_H3_TILE_ELEMS, heads, dim).sum(
        dim=2, dtype=torch.float32
    )
    pooled = pooled / variable_block_sizes.view(1, -1, 1, 1)
    return pooled.permute(0, 2, 1, 3)


def _compute_topk(sparsity: float, num_video_tiles: int) -> int:
    keep = math.ceil((1.0 - sparsity) * num_video_tiles)
    return max(1, min(keep, num_video_tiles))


def _build_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float,
    exempt: bool,
) -> torch.Tensor:
    """scores: [B, H, n_tiles, n_tiles] -> bool mask, same shape."""
    n_tiles = scores.shape[-1]
    keep_video = _compute_topk(sparsity, num_video_tiles)
    if keep_video == num_video_tiles:
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if exempt or num_prefix_tiles == 0:
        video_cols = scores[..., num_prefix_tiles:]
        idx = video_cols.topk(keep_video, dim=-1).indices + num_prefix_tiles
        mask.scatter_(-1, idx, True)
        mask[..., :num_prefix_tiles] = True
    else:
        keep_total = min(keep_video + num_prefix_tiles, n_tiles)
        idx = scores.topk(keep_total, dim=-1).indices
        mask.scatter_(-1, idx, True)
    # Non-video queries are always dense.
    mask[:, :, :num_prefix_tiles, :] = True
    return mask


def _tile_scores_and_mask(
    q_tiled: torch.Tensor,
    k_tiled: torch.Tensor,
    meta: VideoSparseAttentionH3Metadata,
    *,
    sparsity: float,
    head_size: int,
    need_scores: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Pooled tile scores (None when nothing consumes them) and the bool
    block mask, both [B, H, n_tiles, n_tiles]."""
    if sparsity <= 0.0 and not need_scores:
        n_tiles = meta.variable_block_sizes.numel()
        mask = torch.ones(
            1,
            q_tiled.shape[2],
            n_tiles,
            n_tiles,
            dtype=torch.bool,
            device=q_tiled.device,
        )
        return None, mask
    q_pooled = _pool_tiles(q_tiled, meta.variable_block_sizes)
    k_pooled = _pool_tiles(k_tiled, meta.variable_block_sizes)
    scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * (head_size**-0.5)
    mask = _build_block_mask(
        scores=scores,
        num_prefix_tiles=meta.num_prefix_tiles,
        num_video_tiles=meta.num_video_tiles,
        sparsity=sparsity,
        exempt=meta.exempt,
    )
    return scores, mask


def _add_compression_branch(
    out: torch.Tensor,
    *,
    scores: torch.Tensor,
    v_tiled: torch.Tensor,
    gate_tiled: torch.Tensor,
    variable_block_sizes: torch.Tensor,
) -> torch.Tensor:
    """Dense attention over pooled tiles, broadcast to each tile's rows and
    scaled by the learned gate. Zero gates contribute nothing, which is the
    base-H3 contract."""
    n_tiles = variable_block_sizes.numel()
    v_pooled = _pool_tiles(v_tiled, variable_block_sizes)
    out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)
    out_c = out_c.permute(0, 2, 1, 3).to(out.dtype)
    batch, seq_pad, heads, dim = out.shape
    out_tiled = out.view(batch, n_tiles, VSA_H3_TILE_ELEMS, heads, dim)
    gate_view = gate_tiled.view(batch, n_tiles, VSA_H3_TILE_ELEMS, heads, dim)
    return (out_tiled + out_c.unsqueeze(2) * gate_view).view(batch, seq_pad, heads, dim)


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

    def _tile(self, x: torch.Tensor, meta: VideoSparseAttentionH3Metadata):
        """Scatter [B, used, H, D] rows into the padded tile buffer."""
        n_tiles = meta.variable_block_sizes.numel()
        target_shape = (
            x.shape[0],
            n_tiles * VSA_H3_TILE_ELEMS,
            x.shape[-2],
            x.shape[-1],
        )
        buf = meta.tile_buf
        if buf is None or buf.shape != target_shape or buf.dtype != x.dtype:
            buf = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
            meta.tile_buf = buf
        buf[:, meta.non_pad_index] = x[:, meta.tile_partition_indices]
        return buf

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

        stack = [query[:used], key[:used], value[:used]]
        if gate_compress is not None:
            stack.append(gate_compress[:used])
        tiled = self._tile(torch.stack(stack, dim=0), meta)
        q_tiled, k_tiled, v_tiled = tiled[0:1], tiled[1:2], tiled[2:3]
        gate_tiled = tiled[3:4] if gate_compress is not None else None

        scores, mask = _tile_scores_and_mask(
            q_tiled,
            k_tiled,
            meta,
            sparsity=sparsity,
            head_size=self.head_size,
            need_scores=gate_tiled is not None,
        )
        q2k_index, q2k_num = vsa_h3_map_to_index(mask)
        out_bhsd = vsa_h3_block_sparse_attn_forward(
            q=q_tiled.transpose(1, 2).contiguous(),
            k=k_tiled.transpose(1, 2).contiguous(),
            v=v_tiled.transpose(1, 2).contiguous(),
            q2k_index=q2k_index,
            q2k_num=q2k_num,
            variable_block_sizes=meta.variable_block_sizes.to(torch.int32),
        )
        out = out_bhsd.transpose(1, 2)

        if gate_tiled is not None:
            out = _add_compression_branch(
                out,
                scores=scores,
                v_tiled=v_tiled,
                gate_tiled=gate_tiled,
                variable_block_sizes=meta.variable_block_sizes,
            )

        # Untile back to packed rows; alignment-pad rows come out zero.
        result = query.new_zeros(query.shape)
        packed_rows = torch.empty_like(query[:used])
        packed_rows[meta.tile_partition_indices] = out[0, meta.non_pad_index]
        result[:used] = packed_rows
        return result
