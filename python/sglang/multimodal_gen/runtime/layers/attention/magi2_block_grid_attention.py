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

try:
    from magi_attention.api import flex_flash_attn_func

    _HAS_FFA = True
except ImportError:
    # Optional: the flex path below is equivalent, just 2.5x slower here.
    flex_flash_attn_func = None
    _HAS_FFA = False


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


@functools.lru_cache(maxsize=4)
def cached_ffa_ranges(
    *, grid: Magi2BlockGrid, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mask_mod`` as (q, k) token-range pairs: 1.4 MB against the BlockMask's 123 MB.

    A block's neighbours along ``w`` are consecutive block ids, so each ``(t, h)``
    offset collapses to one contiguous key range.
    """
    counts = _block_token_counts(grid=grid, device=device)
    starts = F.pad(counts.cumsum(0), (1, 0)).to(torch.int32)
    nt, nh, nw = grid.block_grid_thw
    radius_t, radius_h, radius_w = grid.radius_thw

    ids = torch.arange(nt * nh * nw, dtype=torch.int64, device=device)
    block_t, block_h, block_w = ids // (nh * nw), ids // nw % nh, ids % nw
    offs_t = torch.arange(-radius_t, radius_t + 1, device=device)
    offs_h = torch.arange(-radius_h, radius_h + 1, device=device)

    near_t = block_t[:, None, None] + offs_t[None, :, None]
    near_h = block_h[:, None, None] + offs_h[None, None, :]
    inside = (near_t >= 0) & (near_t < nt) & (near_h >= 0) & (near_h < nh)
    base = (near_t.clamp(0, nt - 1) * nh + near_h.clamp(0, nh - 1)) * nw
    lo = (block_w - radius_w).clamp(min=0)[:, None, None]
    hi = (block_w + radius_w).clamp(max=nw - 1)[:, None, None]

    q_lo = starts[ids][:, None, None].expand_as(base)[inside]
    q_hi = starts[ids + 1][:, None, None].expand_as(base)[inside]
    q_ranges = [torch.stack((q_lo, q_hi), dim=-1)]
    k_ranges = [
        torch.stack((starts[base + lo][inside], starts[base + hi + 1][inside]), dim=-1)
    ]

    num_video, num_valid = grid.num_video_tokens, grid.num_valid_tokens
    extra = []
    if num_valid > num_video:
        # Only video->video is grid-local, so every video query sees the whole tail.
        extra.append(((0, num_video), (num_video, num_valid)))
    if grid.seq_len > num_video:
        # Tail and pad queries attend densely; a fully masked pad row would be NaN.
        extra.append(((num_video, grid.seq_len), (0, num_valid)))
    for (q0, q1), (k0, k1) in extra:
        q_ranges.append(torch.tensor([[q0, q1]], dtype=torch.int32, device=device))
        k_ranges.append(torch.tensor([[k0, k1]], dtype=torch.int32, device=device))

    return torch.cat(q_ranges), torch.cat(k_ranges)


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


def _tile_near(*, grid: Magi2BlockGrid, device: torch.device) -> torch.Tensor:
    """Tile pairs whose video block ranges come within ``radius_thw`` on every axis."""
    lo, hi, _ = _tile_coord_bounds(grid=grid, device=device)
    near = torch.ones((grid.num_tiles, grid.num_tiles), dtype=torch.bool, device=device)
    for axis, radius in enumerate(grid.radius_thw):
        near &= (lo[axis][:, None] - radius <= hi[axis][None, :]) & (
            hi[axis][:, None] + radius >= lo[axis][None, :]
        )
    return near


def _tile_occupancy(*, grid: Magi2BlockGrid, device: torch.device) -> torch.Tensor:
    """A superset of the true mask, which ``mask_mod`` then masks elementwise."""
    _, _, has_tail = _tile_coord_bounds(grid=grid, device=device)
    near = _tile_near(grid=grid, device=device)
    return near | has_tail[:, None] | has_tail[None, :]


def _tile_kinds(
    *, grid: Magi2BlockGrid, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Which tiles hold one block of video, only tail, or any pad key."""
    lo, hi, has_tail = _tile_coord_bounds(grid=grid, device=device)
    positions = torch.arange(grid.num_tiles * _TILE, device=device)
    tiled = positions.view(grid.num_tiles, _TILE)
    has_video = (tiled < grid.num_video_tokens).any(-1)
    has_pad = (tiled >= grid.num_valid_tokens).any(-1)
    # lo == hi on every axis means all of the tile's video tokens sit in one block,
    # which holds for a full block but not for a ragged edge one.
    single_block = ((lo == hi).all(0)) & has_video
    return single_block & ~has_tail & ~has_pad, ~has_video & ~has_pad, has_pad


def _tile_full(*, grid: Magi2BlockGrid, device: torch.device) -> torch.Tensor:
    """Tile pairs with no masked element, so the kernel can skip ``mask_mod`` on them."""
    video, tail, has_pad = _tile_kinds(grid=grid, device=device)
    both_video = video[:, None] & video[None, :] & _tile_near(grid=grid, device=device)
    # A tail token on either side clears the video-only test for the whole pair.
    with_tail = (tail[:, None] & (video | tail)[None, :]) | (
        video[:, None] & tail[None, :]
    )
    return (both_video | with_tail) & ~has_pad[None, :]


def _rows_to_blockmask_args(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    counts = keep.sum(-1, dtype=torch.int32)
    # Keep all num_tiles columns rather than trimming to counts.max(): from_kv_blocks
    # sizes its transposed-mask buffer from kv_indices.shape[-1].
    indices = keep.to(torch.int8).argsort(dim=-1, descending=True, stable=True)
    return counts[None, None], indices.to(torch.int32)[None, None].contiguous()


@functools.lru_cache(maxsize=2)
def cached_block_mask(*, grid: Magi2BlockGrid, device: torch.device) -> BlockMask:
    """Built sparsely because ``create_block_mask`` materializes a dense [seq_len, seq_len] bool first."""
    allow = _tile_occupancy(grid=grid, device=device)
    full = _tile_full(grid=grid, device=device) & allow
    full_counts, full_indices = _rows_to_blockmask_args(full)
    # Only the ragged and pad-bearing pairs are left to mask elementwise.
    counts, indices = _rows_to_blockmask_args(allow & ~full)
    return BlockMask.from_kv_blocks(
        kv_num_blocks=counts,
        kv_indices=indices,
        full_kv_num_blocks=full_counts,
        full_kv_indices=full_indices,
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

    # Opaque to dynamo: the mask is compiled by _flex_attention above, and an outer
    # region would recompile it per sequence length instead of per bucket.
    @torch._dynamo.disable()
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

        if _HAS_FFA and not self.dense_fallback:
            # FFA takes the packed [S, H, D] layout directly, with no transpose.
            out = self._attend_ffa(q=q, k=k, v=v, grid=grid)
        else:
            out = self._attend(
                q=q.transpose(0, 1)[None],
                k=k.transpose(0, 1)[None],
                v=v.transpose(0, 1)[None],
                grid=grid,
            )[0].transpose(0, 1)

        if ulysses:
            out = _usp_output_all_to_all(out[None], head_dim=2)[0]
        return out

    def _attend_ffa(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid: Magi2BlockGrid,
    ) -> torch.Tensor:
        q_ranges, k_ranges = cached_ffa_ranges(grid=grid, device=q.device)
        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges,
            k_ranges,
            softmax_scale=self.softmax_scale,
            # Measured free (41.99 vs 41.06 ms) and keeps runs reproducible.
            deterministic=True,
            disable_fwd_atomic_reduction=True,
            auto_range_merge=True,
        )
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
