# SPDX-License-Identifier: Apache-2.0
"""Static-tile block-sparse window softmax for VDN-H3 on the VSA-H3 Triton kernel.

The chunk window mask is request-static, so instead of gathering [globals |
window frames | anchors] K/V per chunk (the decomposed path), tile the packed
sequence once in FRAME-MAJOR 1D order -- segment-pure prefix tiles for the
text / condition / audio rows, then per frame ceil(S / 64) tiles (15 full +
one 48-row ragged tile at S = 1008) -- and hand the in-tree tile-64
block-sparse kernel per-query-tile key-tile index lists derived from the
window: prefix tiles, the window frames' tiles and the anchor frames' tiles
(all tiles for anchor-row and global queries). Ragged tiles mask their pad
columns through ``variable_block_sizes``; pad query rows compute garbage that
the untile drops. No K/V gather, in-place reads, ~22% of the dense work at
the paper workload.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.vsa_h3_kernels import (
    VSA_H3_KERNEL_BLOCK,
    vsa_h3_block_sparse_attn_forward,
    vsa_h3_pack_tiles,
    vsa_h3_untile,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import VDNH3Layout

TILE = VSA_H3_KERNEL_BLOCK


def _segment_tiles(start: int, stop: int) -> list[tuple[int, int]]:
    """Split rows [start, stop) into 64-row tiles, the last one ragged."""
    tiles = []
    row = start
    while row < stop:
        size = min(TILE, stop - row)
        tiles.append((row, size))
        row += size
    return tiles


class WindowTilePlan:
    """Tile geometry + kernel index lists for one packed layout."""

    def __init__(
        self,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
    ) -> None:
        from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
            window_mask_frames,
        )

        used = layout.used
        num_frames = layout.num_frames
        bounds, dense_rows, dense_cols = window_mask_frames(hybrid, num_frames)

        # ---- tiles: prefix segments (segment-pure), then frames -----------
        tiles: list[tuple[int, int]] = []
        tile_frame: list[int] = []  # -1 for global rows
        for start, stop in layout.global_ranges:
            # global rows are dense queries, so one tiling per global range is
            # segment-pure enough
            segs = _segment_tiles(start, stop)
            tiles.extend(segs)
            tile_frame.extend([-1] * len(segs))
        prefix_tile_ids = list(range(len(tiles)))
        frame_tile_ids: list[list[int]] = []
        for f in range(num_frames):
            fs, fe = layout.frame_rows(f)
            segs = _segment_tiles(fs, fe)
            ids = list(range(len(tiles), len(tiles) + len(segs)))
            frame_tile_ids.append(ids)
            tiles.extend(segs)
            tile_frame.extend([f] * len(segs))
        n_tiles = len(tiles)
        self.num_tiles = n_tiles
        self.seq_pad = n_tiles * TILE
        sizes = torch.tensor([s for _, s in tiles], dtype=torch.int32)
        if int(sizes.sum()) != used:
            raise ValueError(
                f"tile plan covers {int(sizes.sum())} of {used} packed rows"
            )
        # padded position -> packed row (or -1); packed row -> padded position
        pack_index = torch.full((self.seq_pad,), -1, dtype=torch.int32)
        unpack_index = torch.empty(used, dtype=torch.int32)
        for t, (row, size) in enumerate(tiles):
            pos = torch.arange(t * TILE, t * TILE + size)
            rows = torch.arange(row, row + size)
            pack_index[pos] = rows.to(torch.int32)
            unpack_index[rows] = pos.to(torch.int32)
        self.variable_block_sizes = sizes.to(device)
        self.pack_index = pack_index.to(device)
        self.unpack_index = unpack_index.to(device)

        # ---- per query tile: key tile lists ------------------------------
        anchor_tiles = sorted(t for f in dense_cols for t in frame_tile_ids[f])
        lists: list[list[int]] = []
        all_tiles = list(range(n_tiles))
        for t in range(n_tiles):
            f = tile_frame[t]
            if f < 0 or f in dense_rows:
                lists.append(all_tiles)
                continue
            lo, hi = bounds[f]
            kv = set(prefix_tile_ids) | set(anchor_tiles)
            for wf in range(lo, hi + 1):
                kv.update(frame_tile_ids[wf])
            lists.append(sorted(kv))
        max_kv = max(len(l) for l in lists)
        q2k_index = torch.zeros((n_tiles, max_kv), dtype=torch.int32)
        q2k_num = torch.zeros((n_tiles,), dtype=torch.int32)
        for t, l in enumerate(lists):
            q2k_index[t, : len(l)] = torch.tensor(l, dtype=torch.int32)
            q2k_num[t] = len(l)
        self._q2k_index_1h = q2k_index.to(device)
        self._q2k_num_1h = q2k_num.to(device)
        self.max_kv = max_kv
        self._per_head: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._workspace: dict[tuple, dict[str, torch.Tensor]] = {}

    def index_lists(self, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """[1, H, n_tiles, max_kv] / [1, H, n_tiles] int32, head-major
        contiguous as the kernel addresses them."""
        cached = self._per_head.get(heads)
        if cached is None:
            cached = (
                self._q2k_index_1h[None].repeat(heads, 1, 1)[None].contiguous(),
                self._q2k_num_1h[None].repeat(heads, 1)[None].contiguous(),
            )
            self._per_head[heads] = cached
        return cached

    def workspace(
        self, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        key = (heads, head_dim, dtype, device)
        ws = self._workspace.get(key)
        if ws is None:
            ws = {
                "tiled": torch.empty(
                    (3, heads, self.seq_pad, head_dim), dtype=dtype, device=device
                ),
                "pooled": torch.empty(
                    (3, heads, self.num_tiles, head_dim),
                    dtype=torch.float32,
                    device=device,
                ),
                "out_tiled": torch.empty(
                    (heads, self.seq_pad, head_dim), dtype=dtype, device=device
                ),
            }
            self._workspace[key] = ws
        return ws


def build_window_tile_plan(
    layout: VDNH3Layout,
    hybrid: VDNHybridAttentionArchConfig,
    device: torch.device,
) -> WindowTilePlan:
    return WindowTilePlan(layout, hybrid, device)


def window_tile_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: WindowTilePlan,
    used: int,
) -> torch.Tensor:
    """query/key/value [T, H, D] bf16 (post-norm, post-RoPE) -> [T, H, D];
    rows at and past ``used`` are zero."""
    if query.dtype != torch.bfloat16:
        raise ValueError("the tile-64 window kernel serves bf16 only")
    heads, head_dim = query.shape[-2], query.shape[-1]
    ws = plan.workspace(heads, head_dim, query.dtype, query.device)
    vsa_h3_pack_tiles(
        query,
        key,
        value,
        None,
        plan.pack_index,
        plan.variable_block_sizes,
        ws["tiled"],
        ws["pooled"],
    )
    q2k_index, q2k_num = plan.index_lists(heads)
    vsa_h3_block_sparse_attn_forward(
        ws["tiled"][0:1],
        ws["tiled"][1:2],
        ws["tiled"][2:3],
        q2k_index,
        q2k_num,
        plan.variable_block_sizes,
        out=ws["out_tiled"][None],
    )
    result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    vsa_h3_untile(ws["out_tiled"], None, None, plan.unpack_index, used, result)
    return result


__all__ = ["WindowTilePlan", "build_window_tile_plan", "window_tile_attention"]
