# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math

import msgspec
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_mask, flex_attention

from sglang.multimodal_gen.configs.models.dits.magi2 import Magi2RefinerArchConfig

_TILE = 128

# Each distinct sequence length compiles its own graph, hence SEQ_BUCKET.
_flex_attention = torch.compile(flex_attention, dynamic=False)


class Magi2BlockGrid(msgspec.Struct, frozen=True):
    """Frozen so it can key the mask cache."""

    latent_thw: tuple[int, int, int]
    block_thw: tuple[int, int, int]
    radius_thw: tuple[int, int, int]
    num_tail_tokens: int
    num_pad_tokens: int = 0

    @classmethod
    def from_arch_config(
        cls,
        *,
        arch_config: Magi2RefinerArchConfig,
        latent_thw: tuple[int, int, int],
        num_tail_tokens: int,
        num_pad_tokens: int = 0,
    ) -> Magi2BlockGrid:
        return cls(
            latent_thw=latent_thw,
            block_thw=(
                arch_config.block_t_size,
                arch_config.block_size,
                arch_config.block_size,
            ),
            radius_thw=(
                arch_config.block_t_radius,
                arch_config.block_h_radius,
                arch_config.block_w_radius,
            ),
            num_tail_tokens=num_tail_tokens,
            num_pad_tokens=num_pad_tokens,
        )

    @property
    def num_video_tokens(self) -> int:
        return math.prod(self.latent_thw)

    @property
    def num_valid_tokens(self) -> int:
        return self.num_video_tokens + self.num_tail_tokens

    @property
    def seq_len(self) -> int:
        return self.num_valid_tokens + self.num_pad_tokens

    @property
    def block_grid_thw(self) -> tuple[int, int, int]:
        t, h, w = (math.ceil(n / b) for n, b in zip(self.latent_thw, self.block_thw))
        return (t, h, w)

    @property
    def num_tiles(self) -> int:
        return math.ceil(self.seq_len / _TILE)


# Divides every supported sp degree, so the shard adds no padding of its own.
SEQ_BUCKET = 8 * _TILE


def _dim_block_sizes(*, dim: int, block: int, device: torch.device) -> torch.Tensor:
    sizes = torch.full(
        (math.ceil(dim / block),), block, dtype=torch.int64, device=device
    )
    sizes[-1] = dim - block * (sizes.numel() - 1)
    return sizes


def _block_token_counts(*, grid: Magi2BlockGrid, device: torch.device) -> torch.Tensor:
    t_sizes, h_sizes, w_sizes = (
        _dim_block_sizes(dim=dim, block=block, device=device)
        for dim, block in zip(grid.latent_thw, grid.block_thw)
    )
    return (
        t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
    ).flatten()


def block_scan_order(
    *, grid: Magi2BlockGrid, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    nt, nh, nw = grid.block_grid_thw
    bt, bh, bw = grid.block_thw
    ids = torch.arange(grid.num_video_tokens, dtype=torch.int64, device=device)
    padded = ids.new_full((nt * bt, nh * bh, nw * bw), -1)
    t, h, w = grid.latent_thw
    padded[:t, :h, :w] = ids.view(t, h, w)
    scan = padded.view(nt, bt, nh, bh, nw, bw).permute(0, 2, 4, 1, 3, 5).reshape(-1)
    tail = torch.arange(
        grid.num_video_tokens, grid.seq_len, dtype=torch.int64, device=device
    )
    order = torch.cat([scan[scan >= 0], tail])
    restore = torch.empty_like(order)
    restore[order] = torch.arange(order.numel(), dtype=torch.int64, device=device)
    return order, restore


def _token_block_coords(
    *, grid: Magi2BlockGrid, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Must cover the padded length: the kernel evaluates ``mask_mod`` on the full tile including its ragged end."""
    counts = _block_token_counts(grid=grid, device=device)
    ids = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.int64, device=device), counts
    )
    ids = F.pad(ids, (0, grid.num_tiles * _TILE - grid.num_video_tokens))
    _, nh, nw = grid.block_grid_thw
    return (
        (ids // (nh * nw)).to(torch.int32),
        (ids // nw % nh).to(torch.int32),
        (ids % nw).to(torch.int32),
    )


def build_mask_mod(*, grid: Magi2BlockGrid, device: torch.device):
    # Only video->video is grid-local; the tail attends densely.
    block_t, block_h, block_w = _token_block_coords(grid=grid, device=device)
    # dynamo guards a captured int BY VALUE, so the prompt-dependent valid length
    # must be a tensor; num_video is tier-fixed and safe as an int.
    num_video = grid.num_video_tokens
    is_valid_key = (
        torch.arange(grid.num_tiles * _TILE, device=device) < grid.num_valid_tokens
    )
    radius_t, radius_h, radius_w = grid.radius_thw

    def mask_mod(b, h, q_idx, kv_idx):
        video_pair = (q_idx < num_video) & (kv_idx < num_video)
        near = (
            ((block_t[q_idx] - block_t[kv_idx]).abs() <= radius_t)
            & ((block_h[q_idx] - block_h[kv_idx]).abs() <= radius_h)
            & ((block_w[q_idx] - block_w[kv_idx]).abs() <= radius_w)
        )
        # Pad stays attendable as a query: a fully masked row softmaxes to NaN.
        return (near | ~video_pair) & is_valid_key[kv_idx]

    return mask_mod


def _tile_coord_bounds(
    *, grid: Magi2BlockGrid, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``lo``/``hi`` are [3, num_tiles]; a tile with no video token gets ``lo > hi``."""
    coords = torch.stack(_token_block_coords(grid=grid, device=device))
    positions = torch.arange(coords.shape[1], dtype=torch.int64, device=device)
    is_video = (positions < grid.num_video_tokens).expand(3, -1)
    unreachable = torch.iinfo(torch.int32).max // 2
    tiles = (3, grid.num_tiles, _TILE)
    lo = torch.where(is_video, coords, unreachable).view(tiles).amin(-1)
    hi = torch.where(is_video, coords, -unreachable).view(tiles).amax(-1)
    has_tail = (
        ((positions >= grid.num_video_tokens) & (positions < grid.seq_len))
        .view(grid.num_tiles, _TILE)
        .any(-1)
    )
    return lo, hi, has_tail


def _tile_occupancy(*, grid: Magi2BlockGrid, device: torch.device) -> torch.Tensor:
    """A superset of the true mask, which ``mask_mod`` then masks elementwise."""
    lo, hi, has_tail = _tile_coord_bounds(grid=grid, device=device)
    near = torch.ones((grid.num_tiles, grid.num_tiles), dtype=torch.bool, device=device)
    for axis, radius in enumerate(grid.radius_thw):
        near &= (lo[axis][:, None] - radius <= hi[axis][None, :]) & (
            hi[axis][:, None] + radius >= lo[axis][None, :]
        )
    return near | has_tail[:, None] | has_tail[None, :]


@functools.lru_cache(maxsize=2)
def cached_block_mask(*, grid: Magi2BlockGrid, device: torch.device) -> BlockMask:
    """Built sparsely because ``create_block_mask`` materializes a dense [seq_len, seq_len] bool first."""
    allow = _tile_occupancy(grid=grid, device=device)
    counts = allow.sum(-1, dtype=torch.int32)
    # kv_indices must keep all num_tiles columns, not counts.max(): from_kv_blocks
    # sizes its transposed-mask buffer from kv_indices.shape[-1].
    indices = allow.to(torch.int8).argsort(dim=-1, descending=True, stable=True)
    return BlockMask.from_kv_blocks(
        kv_num_blocks=counts[None, None],
        kv_indices=indices.to(torch.int32)[None, None].contiguous(),
        BLOCK_SIZE=(_TILE, _TILE),
        mask_mod=build_mask_mod(grid=grid, device=device),
        seq_lengths=(grid.seq_len, grid.seq_len),
    )


class Magi2BlockGridAttention(nn.Module):
    """Q/K/V must already be in ``block_scan_order``; only video->video is grid-local."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        dense_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale or head_dim**-0.5
        self.enable_gqa = self.num_kv_heads != num_heads
        self.dense_fallback = dense_fallback

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid: Magi2BlockGrid,
    ) -> torch.Tensor:
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_ulysses_parallel_world_size,
        )
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all,
            _usp_output_all_to_all,
        )

        ulysses = get_ulysses_parallel_world_size() > 1
        if ulysses:
            # Per-tensor: the packed helper asserts q/k/v share a shape, and this is GQA.
            q, k, v = (_usp_input_all_to_all(t[None], head_dim=2)[0] for t in (q, k, v))

        out = self._attend(
            q=q.transpose(0, 1)[None],
            k=k.transpose(0, 1)[None],
            v=v.transpose(0, 1)[None],
            grid=grid,
        )
        out = out[0].transpose(0, 1)

        if ulysses:
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out

    def _attend(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid: Magi2BlockGrid,
    ) -> torch.Tensor:
        if self.dense_fallback:
            mask = create_mask(
                build_mask_mod(grid=grid, device=q.device),
                1,
                1,
                grid.seq_len,
                grid.seq_len,
                device=q.device,
            )
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                scale=self.softmax_scale,
                enable_gqa=self.enable_gqa,
            )
        return _flex_attention(
            q,
            k,
            v,
            block_mask=cached_block_mask(grid=grid, device=q.device),
            scale=self.softmax_scale,
            enable_gqa=self.enable_gqa,
        )
