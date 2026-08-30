# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0
"""Official FastH3 VSA for MiniMax H3 packed mixed-modality self-attention.

H3 runs one joint bidirectional attention over
``[text | condition keyframes | audio | generated video]``. This is not the
Wan ``video_sparse_attn`` backend: prefix tiles stay segment-pure, video
uses 64-token ``(4, 4, 4)`` tiles, and non-video keys are always selected
(FastVideo "exempt"). ``to_gate_compress`` is the trained coarse-gate
branch; all-zero weights skip the GEMM.

H200 / sm90 uses Triton ``block_sparse_attn_from_indices``. QKV stay in
reusable BHSD scratch so the kernel sees contiguous ``[B, H, S_pad, D]``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

try:
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices
except ImportError:
    block_sparse_attn_from_indices = None

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

# Tile / pad index helpers only. The attention kernel is Triton
# ``block_sparse_attn_from_indices`` above, not Wan ``vsa.video_sparse_attn``.
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    construct_variable_block_sizes,
    get_non_pad_index,
    get_tile_partition_indices,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

VSA_H3_TILE_SHAPE: tuple[int, int, int] = (4, 4, 4)
VSA_H3_DEFAULT_TILE_ELEMS = 64


def h3_vsa_prefix_segments(
    text_len: int,
    cond_rows: int = 0,
    audio_rows: int = 0,
) -> tuple[int, ...]:
    """Packed prefix lengths FastVideo VSA-H3 expects, dropping empty spans."""
    return tuple(
        int(size) for size in (text_len, cond_rows, audio_rows) if int(size) > 0
    )


def compute_topk(sparsity: float, num_kv_blocks: int) -> int:
    if num_kv_blocks <= 0:
        return 0
    return max(
        1, min(num_kv_blocks, math.ceil((1.0 - float(sparsity)) * num_kv_blocks))
    )


def _validate_h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    variable_block_sizes: torch.Tensor,
    untile_combined_index: torch.Tensor,
    tile_elems: int,
) -> None:
    total = sum(prefix_segments) + math.prod(dit_seq_shape)
    n_pad = variable_block_sizes.numel() * tile_elems
    sizes_min = int(variable_block_sizes.min())
    sizes_max = int(variable_block_sizes.max())
    sizes_sum = int(variable_block_sizes.sum())
    if sizes_min < 1 or sizes_max > tile_elems or sizes_sum != total:
        raise ValueError(
            f"VSA-H3 tile sizes out of bounds for prefix={prefix_segments}, "
            f"video={dit_seq_shape}, tile_elems={tile_elems}: min={sizes_min}, "
            f"max={sizes_max}, sum={sizes_sum}, expected sum={total}."
        )
    if untile_combined_index.numel() != total:
        raise ValueError(
            f"VSA-H3 untile index has {untile_combined_index.numel()} entries "
            f"for a packed sequence of {total} rows (prefix={prefix_segments}, "
            f"video={dit_seq_shape})."
        )
    idx_min = int(untile_combined_index.min())
    idx_max = int(untile_combined_index.max())
    if idx_min < 0 or idx_max >= n_pad:
        raise ValueError(
            f"VSA-H3 untile index is not an injective map into non-pad slots: "
            f"range [{idx_min}, {idx_max}] vs padded length {n_pad} "
            f"(prefix={prefix_segments}, video={dit_seq_shape})."
        )
    in_tile_offset = untile_combined_index % tile_elems
    maps_into_pad = bool(
        (
            in_tile_offset >= variable_block_sizes[untile_combined_index // tile_elems]
        ).any()
    )
    if maps_into_pad or int(torch.unique(untile_combined_index).numel()) != total:
        raise ValueError(
            f"VSA-H3 untile index is not an injective map into non-pad slots: "
            f"pad-slot hit={maps_into_pad} "
            f"(prefix={prefix_segments}, video={dit_seq_shape})."
        )


@functools.lru_cache(maxsize=10)
def _h3_tile_geometry(
    prefix_segments: tuple[int, ...],
    dit_seq_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    tile_elems = VSA_H3_DEFAULT_TILE_ELEMS
    prefix_len = sum(prefix_segments)

    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, rem = divmod(segment, tile_elems)
        prefix_sizes.extend([tile_elems] * full)
        if rem:
            prefix_sizes.append(rem)
    num_prefix_tiles = len(prefix_sizes)

    ts_t, ts_h, ts_w = VSA_H3_TILE_SHAPE
    t, h, w = dit_seq_shape
    num_tiles = (math.ceil(t / ts_t), math.ceil(h / ts_h), math.ceil(w / ts_w))
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
            video_sizes,
        ]
    )
    non_pad_index = get_non_pad_index(variable_block_sizes, tile_elems)
    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    _validate_h3_tile_geometry(
        prefix_segments,
        dit_seq_shape,
        variable_block_sizes,
        untile_combined_index,
        tile_elems,
    )
    return (
        tile_partition_indices,
        variable_block_sizes,
        untile_combined_index,
        num_prefix_tiles,
        num_video_tiles,
    )


class MiniMaxH3VSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3

    @staticmethod
    def get_impl_cls() -> type[MiniMaxH3VSAImpl]:
        return MiniMaxH3VSAImpl

    @staticmethod
    def get_metadata_cls() -> type[MiniMaxH3VSAMetadata]:
        return MiniMaxH3VSAMetadata

    @staticmethod
    def get_builder_cls() -> type[MiniMaxH3VSAMetadataBuilder]:
        return MiniMaxH3VSAMetadataBuilder


class _MiniMaxH3VSATileBufferHolder:
    """Builder-owned BHSD scratch, one allocation per Q/K/V/gate stream."""

    def __init__(self) -> None:
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.gate: torch.Tensor | None = None
        self.q_geo: torch.Tensor | None = None
        self.k_geo: torch.Tensor | None = None
        self.v_geo: torch.Tensor | None = None
        self.gate_geo: torch.Tensor | None = None


@dataclass
class MiniMaxH3VSAMetadata(AttentionMetadata):
    current_timestep: int
    VSA_sparsity: float
    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    variable_block_sizes: torch.Tensor
    untile_combined_index: torch.Tensor
    tile_elems: int = VSA_H3_DEFAULT_TILE_ELEMS
    tile_buf_holder: _MiniMaxH3VSATileBufferHolder | None = None


class MiniMaxH3VSAMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        self._tile_buf_holder = _MiniMaxH3VSATileBufferHolder()

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
        tile_size: int = VSA_H3_DEFAULT_TILE_ELEMS,
        **kwargs: dict[str, Any],
    ) -> MiniMaxH3VSAMetadata:
        if int(tile_size) != VSA_H3_DEFAULT_TILE_ELEMS:
            raise ValueError(
                f"VSA-H3 tile_size must be {VSA_H3_DEFAULT_TILE_ELEMS} "
                f"(FastH3 / H200 Triton), got {tile_size!r}."
            )
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )
        prefix_segments = tuple(int(size) for size in prefix_segments if size > 0)
        total_seq_length = sum(prefix_segments) + math.prod(dit_seq_shape)
        (
            _tile_partition_indices,
            variable_block_sizes,
            untile_combined_index,
            num_prefix_tiles,
            num_video_tiles,
        ) = _h3_tile_geometry(prefix_segments, dit_seq_shape, device)
        return MiniMaxH3VSAMetadata(
            current_timestep=current_timestep,
            VSA_sparsity=float(VSA_sparsity),
            total_seq_length=total_seq_length,
            num_prefix_tiles=num_prefix_tiles,
            num_video_tiles=num_video_tiles,
            variable_block_sizes=variable_block_sizes,
            untile_combined_index=untile_combined_index,
            tile_elems=VSA_H3_DEFAULT_TILE_ELEMS,
            tile_buf_holder=self._tile_buf_holder,
        )


def _pool_tiles_bhsd(
    x: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    tile_elems: int,
) -> torch.Tensor:
    """fp32 mean over each tile. x: [B, H, S_pad, D] -> [B, H, n_tiles, D]."""
    batch, heads, seq_len, dim = x.shape
    n_tiles = seq_len // tile_elems
    pooled = x.view(batch, heads, n_tiles, tile_elems, dim).sum(
        dim=3, dtype=torch.float32
    )
    return pooled / variable_block_sizes.view(1, 1, -1, 1)


def _dense_indices(
    batch: int,
    heads: int,
    n_tiles: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q2k_idx = (
        torch.arange(n_tiles, device=device, dtype=torch.int32)
        .view(1, 1, 1, n_tiles)
        .expand(batch, heads, n_tiles, n_tiles)
        .contiguous()
    )
    q2k_num = torch.full(
        (batch, heads, n_tiles), n_tiles, dtype=torch.int32, device=device
    )
    return q2k_idx, q2k_num


def _sparse_indices_from_scores(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    VSA_sparsity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official FastH3 exempt indices: every prefix tile plus top-k video."""
    batch, heads, n_tiles, _ = scores.shape
    k_vid = compute_topk(VSA_sparsity, num_video_tiles)
    if k_vid == num_video_tiles:
        return _dense_indices(batch, heads, n_tiles, scores.device)
    video_idx = scores[..., num_prefix_tiles:].topk(k_vid, dim=-1).indices + (
        num_prefix_tiles
    )
    if num_prefix_tiles == 0:
        q2k_idx = video_idx.to(torch.int32).contiguous()
    else:
        prefix_idx = (
            torch.arange(num_prefix_tiles, device=scores.device, dtype=video_idx.dtype)
            .view(1, 1, 1, -1)
            .expand(batch, heads, n_tiles, num_prefix_tiles)
        )
        q2k_idx = (
            torch.cat([prefix_idx, video_idx], dim=-1).to(torch.int32).contiguous()
        )
    q2k_num = torch.full(
        (batch, heads, n_tiles),
        q2k_idx.shape[-1],
        dtype=torch.int32,
        device=scores.device,
    )
    return q2k_idx, q2k_num


def _reuse_or_alloc_bhsd(
    buf: torch.Tensor | None,
    packed: torch.Tensor,
    s_pad: int,
    dest_index: torch.Tensor,
    last_geometry: torch.Tensor | None,
) -> torch.Tensor:
    """Scatter packed ``[B, S, H, D]`` into a reusable BHSD tile buffer."""
    batch, _, heads, dim = packed.shape
    target_shape = (batch, heads, s_pad, dim)
    matches = (
        buf is not None
        and buf.shape == target_shape
        and buf.dtype == packed.dtype
        and buf.device == packed.device
    )
    if not matches:
        buf = packed.new_zeros(target_shape)
    elif last_geometry is not dest_index:
        buf.zero_()
    buf[:, :, dest_index] = packed.permute(0, 2, 1, 3)
    return buf


class MiniMaxH3VSAImpl(AttentionImpl):
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
        del causal, num_heads, num_kv_heads, extra_impl_args
        self.prefix = prefix
        self.head_size = head_size
        self.softmax_scale = float(softmax_scale)

    def _tile_bhsd(
        self,
        x: torch.Tensor,
        attn_metadata: MiniMaxH3VSAMetadata,
        which: str,
    ) -> torch.Tensor:
        if x.shape[1] != attn_metadata.total_seq_length:
            raise ValueError(
                f"VSA-H3 metadata was built for sequence length "
                f"{attn_metadata.total_seq_length}, got {x.shape[1]}. A "
                "non-packed sequence (e.g. the token refiner) must stay on FA."
            )
        holder = attn_metadata.tile_buf_holder
        if holder is None:
            raise RuntimeError(
                "VSA-H3 metadata has no builder-owned tile buffer holder"
            )
        s_pad = attn_metadata.variable_block_sizes.numel() * attn_metadata.tile_elems
        dest = attn_metadata.untile_combined_index
        current = getattr(holder, which)
        last_geo = getattr(holder, f"{which}_geo")
        tiled = _reuse_or_alloc_bhsd(current, x, s_pad, dest, last_geo)
        setattr(holder, which, tiled)
        setattr(holder, f"{which}_geo", dest)
        return tiled

    def preprocess_qkv(
        self, qkv: torch.Tensor, attn_metadata: MiniMaxH3VSAMetadata
    ) -> torch.Tensor:
        raise NotImplementedError("VSA-H3 tiles Q/K/V separately in BHSD")

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: MiniMaxH3VSAMetadata,
    ) -> torch.Tensor:
        return output[:, :, attn_metadata.untile_combined_index].permute(0, 2, 1, 3)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor | None,
        attn_metadata: MiniMaxH3VSAMetadata,
    ) -> torch.Tensor:
        if block_sparse_attn_from_indices is None:
            raise NotImplementedError(
                "fastvideo_kernel.block_sparse_attn is not installed"
            )
        if attn_metadata.tile_elems != VSA_H3_DEFAULT_TILE_ELEMS:
            raise NotImplementedError(
                f"VSA-H3 Triton path only supports {VSA_H3_DEFAULT_TILE_ELEMS}-token "
                f"tiles, got {attn_metadata.tile_elems}"
            )

        tile_elems = attn_metadata.tile_elems
        n_tiles = attn_metadata.variable_block_sizes.numel()
        logical_seq_len = n_tiles * tile_elems
        if query.shape[-2] != logical_seq_len:
            raise ValueError(
                f"VSA-H3 tiled query has length {query.shape[-2]}, "
                f"expected {logical_seq_len}."
            )
        for name, tensor in (("key", key), ("value", value)):
            if tensor.shape != query.shape:
                raise ValueError(
                    f"VSA-H3 tiled {name} shape {tuple(tensor.shape)} does not "
                    f"match query {tuple(query.shape)}."
                )
        if gate_compress is not None and gate_compress.shape != query.shape:
            raise ValueError(
                f"VSA-H3 tiled gate shape {tuple(gate_compress.shape)} does not "
                f"match query {tuple(query.shape)}."
            )

        scores = None
        if attn_metadata.VSA_sparsity > 0.0 or gate_compress is not None:
            q_pooled = _pool_tiles_bhsd(
                query, attn_metadata.variable_block_sizes, tile_elems
            )
            k_pooled = _pool_tiles_bhsd(
                key, attn_metadata.variable_block_sizes, tile_elems
            )
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) * (
                self.softmax_scale
            )

        if scores is None:
            q2k_idx, q2k_num = _dense_indices(
                query.shape[0], query.shape[1], n_tiles, query.device
            )
        else:
            q2k_idx, q2k_num = _sparse_indices_from_scores(
                scores,
                attn_metadata.num_prefix_tiles,
                attn_metadata.num_video_tiles,
                attn_metadata.VSA_sparsity,
            )

        out_bhsd, _ = block_sparse_attn_from_indices(
            query,
            key,
            value,
            q2k_idx,
            q2k_num,
            attn_metadata.variable_block_sizes,
        )

        if gate_compress is not None:
            assert scores is not None
            v_pooled = _pool_tiles_bhsd(
                value, attn_metadata.variable_block_sizes, tile_elems
            )
            out_c = torch.matmul(torch.softmax(scores, dim=-1), v_pooled)
            batch, heads, seq_len, dim = out_bhsd.shape
            out_tiled = out_bhsd.view(batch, heads, n_tiles, tile_elems, dim)
            gate_tiled = gate_compress.view(batch, heads, n_tiles, tile_elems, dim)
            out_bhsd = (
                out_tiled + out_c.unsqueeze(3).to(out_bhsd.dtype) * gate_tiled
            ).view(batch, heads, seq_len, dim)
        return out_bhsd

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        del cu_seqlens, cu_seqlens_host
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, MiniMaxH3VSAMetadata):
            raise RuntimeError(
                "VSA-H3 requires MiniMaxH3VSAMetadata in the forward context"
            )
        used = int(max_seqlen)
        if query.shape[0] < used:
            raise ValueError(
                f"VSA-H3 packed query length {query.shape[0]} < used={used}"
            )
        if used != attn_metadata.total_seq_length:
            raise ValueError(
                f"VSA-H3 metadata total_seq_length={attn_metadata.total_seq_length} "
                f"does not match packed used={used}"
            )
        q = query[:used].unsqueeze(0)
        k = key[:used].unsqueeze(0)
        v = value[:used].unsqueeze(0)
        g = None if gate_compress is None else gate_compress[:used].unsqueeze(0)
        q_t = self._tile_bhsd(q, attn_metadata, "q")
        k_t = self._tile_bhsd(k, attn_metadata, "k")
        v_t = self._tile_bhsd(v, attn_metadata, "v")
        g_t = None if g is None else self._tile_bhsd(g, attn_metadata, "gate")
        out = self.forward(q_t, k_t, v_t, g_t, attn_metadata)
        packed = self.postprocess_output(out, attn_metadata)[0]
        result = query.new_zeros(query.shape)
        result[:used] = packed
        return result
