"""Exact tile-max-gated paged top-k for the DeepSeek-V4.1 ROCm dense prefill indexer.

A drop-in for ``topk_transform_paged_hip`` given each row's 16-key tile maxima M_t (from the
scoring kernel). With T the k-th largest M_t, the k tiles at the top each hold an element
>= T, so the k-th largest element is >= T and the whole top-k (ties included) lies in tiles
with M_t >= T. Those tiles are compacted in order into a candidate row (at most CAP_TILES)
and top-k'd there; a row with more survivors (ties at T) takes the stock top-k. No host sync.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4 import plan_topk_v2, topk_transform_paged_v2

TILE = 16
# CAP_TILES * TILE = 16384 keeps the candidate top-k on top-k v2's register path
CAP_TILES = 1024
ROWS_PER_BLOCK = 2048
# measured on MI355X at 1M keys
_BT = 1024
_GA_TB = 64


@triton.jit
def _tile_fixup_kernel(
    scores_ptr,
    stride_s,
    lens_ptr,
    tmax_ptr,
    stride_t,
    ntiles_ptr,
):
    """Each row's tile count, and its last partial tile's max (the kernel's is unmasked)."""
    r = tl.program_id(0)
    L = tl.load(lens_ptr + r)
    nT = (L + 15) // 16
    tl.store(ntiles_ptr + r, nT)
    if (L % 16) != 0:
        j = tl.arange(0, 16)
        col = (nT - 1) * 16 + j
        v = tl.load(
            scores_ptr + r.to(tl.int64) * stride_s + col,
            mask=col < L,
            other=float("-inf"),
        )
        tl.store(tmax_ptr + r.to(tl.int64) * stride_t + (nT - 1), tl.max(v, axis=0))


@triton.jit
def _compact_kernel(
    tmax_ptr,
    stride_t,
    tsel_ptr,
    ntiles_ptr,
    lens_ptr,
    ctile_ptr,
    cand_len_ptr,
    fblen_ptr,
    count_ptr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    BT: tl.constexpr,
):
    """With T the min of the row's k best tile maxima: the ids of the tiles reaching T in
    order (the first CAP), and the row's candidate length, fallback length and count."""
    r = tl.program_id(0)
    r64 = r.to(tl.int64)
    sel = tl.load(tsel_ptr + r64 * K + tl.arange(0, K))
    sv = tl.load(tmax_ptr + r64 * stride_t + sel, mask=sel >= 0, other=float("inf"))
    thr = tl.min(sv, axis=0)
    n_tiles = tl.load(ntiles_ptr + r)
    total = tl.full((), 0, tl.int32)
    last = tl.full((), -1, tl.int32)
    for b in range(0, tl.cdiv(n_tiles, BT)):
        t = b * BT + tl.arange(0, BT)
        valid = t < n_tiles
        mv = tl.load(tmax_ptr + r64 * stride_t + t, mask=valid, other=float("-inf"))
        keep = (mv >= thr) & valid
        ki = keep.to(tl.int32)
        pos = tl.cumsum(ki, axis=0) - ki + total
        tl.store(ctile_ptr + r64 * CAP + pos, t, mask=keep & (pos < CAP))
        total += tl.sum(ki, axis=0)
        last = tl.maximum(last, tl.max(tl.where(keep, t, -1), axis=0))
    L = tl.load(lens_ptr + r)
    over = total > CAP
    cand_len = (total - 1) * 16 + tl.minimum(16, L - last * 16)
    tl.store(cand_len_ptr + r, tl.where(over | (total == 0), 0, cand_len))
    tl.store(fblen_ptr + r, tl.where(over, L, 0))
    tl.store(count_ptr + r, total)


@triton.jit
def _gather_kernel(
    scores_ptr,
    stride_s,
    ctile_ptr,
    count_ptr,
    fblen_ptr,
    cand_ptr,
    CAP: tl.constexpr,
    TB: tl.constexpr,
):
    """cand[r, 16 i + j] = scores[r, 16 ctile[r, i] + j]."""
    r = tl.program_id(0)
    pb = tl.program_id(1)
    n = tl.minimum(tl.load(count_ptr + r), CAP)
    i0 = pb * TB
    if (i0 < n) & (tl.load(fblen_ptr + r) == 0):
        i = i0 + tl.arange(0, TB)
        ok = i < n
        tile = tl.load(ctile_ptr + r.to(tl.int64) * CAP + i, mask=ok, other=0)
        j = tl.arange(0, 16)
        v = tl.load(
            scores_ptr + r.to(tl.int64) * stride_s + tile[:, None] * 16 + j[None, :],
            mask=ok[:, None],
        )
        tl.store(
            cand_ptr + r.to(tl.int64) * (CAP * 16) + i[:, None] * 16 + j[None, :],
            v,
            mask=ok[:, None],
        )


@triton.jit
def _finalize_kernel(
    csel_ptr,
    ctile_ptr,
    fblen_ptr,
    fbp_ptr,
    fbr_ptr,
    pt_ptr,
    stride_pt,
    out_page_ptr,
    stride_op,
    out_raw_ptr,
    stride_or,
    CAP: tl.constexpr,
    K: tl.constexpr,
    PS_BITS: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    r = tl.program_id(0)
    r64 = r.to(tl.int64)
    k = tl.arange(0, K)
    fb = tl.load(fblen_ptr + r) > 0
    if fb:
        page = tl.load(fbp_ptr + r64 * K + k)
        pos = tl.load(fbr_ptr + r64 * K + k)
    else:
        ci = tl.load(csel_ptr + r64 * K + k)
        good = ci >= 0
        cis = tl.where(good, ci, 0)
        tile = tl.load(ctile_ptr + r64 * CAP + cis // 16, mask=good, other=0)
        pos = tile * 16 + cis % 16
        pg = tl.load(pt_ptr + r64 * stride_pt + (pos >> PS_BITS), mask=good, other=0)
        page = (pg << PS_BITS) | (pos & ((1 << PS_BITS) - 1))
        page = tl.where(good, page, -1)
        pos = tl.where(good, pos, -1)
    tl.store(out_page_ptr + r64 * stride_op + k, page)
    if HAS_RAW:
        tl.store(out_raw_ptr + r64 * stride_or + k, pos)


def _view(buf: torch.Tensor, *shape: int) -> torch.Tensor:
    return buf[: shape[0] * (shape[1] if len(shape) > 1 else 1)].view(*shape)


class TileGateTopK:
    """The scratch is reserved at construction, so a prefill allocates none."""

    def __init__(self, *, topk: int, tile_max_elems: int, device: torch.device):
        assert topk & (topk - 1) == 0 and topk <= 2048, topk
        n = ROWS_PER_BLOCK
        self.topk = topk
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        self._tile_max = torch.empty(tile_max_elems, **f32)
        self._cand = torch.empty(n * CAP_TILES * TILE, **f32)
        self._cand_tiles = torch.empty(n * CAP_TILES, **i32)
        self._tile_sel, self._cand_sel, self._fb_page, self._fb_raw = (
            torch.empty(n * topk, **i32) for _ in range(4)
        )
        self._ntiles, self._cand_len, self._fb_len, self._count = (
            torch.empty(n, **i32) for _ in range(4)
        )
        self._plan = plan_topk_v2(torch.zeros(n, **i32))

    def tile_max(self, rows: int, width: int) -> Optional[torch.Tensor]:
        """[rows, width / 16] f32 for the scoring kernel, or None past the reserve."""
        if rows * (width // TILE) > self._tile_max.numel():
            return None
        return _view(self._tile_max, rows, width // TILE)

    def __call__(
        self,
        scores: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        page_indices: torch.Tensor,
        page_size: int,
        raw_indices: Optional[torch.Tensor],
        tile_max: torch.Tensor,
    ) -> None:
        """topk_transform_paged_hip(scores, ...), tile_max from self.tile_max()."""
        R, K = scores.shape[0], page_indices.shape[1]
        assert K == self.topk, (K, self.topk)
        assert page_size & (page_size - 1) == 0
        seq_lens = seq_lens.contiguous()
        ps_bits = page_size.bit_length() - 1
        for lo in range(0, R, ROWS_PER_BLOCK):
            hi = min(R, lo + ROWS_PER_BLOCK)
            self._select_block(
                scores[lo:hi],
                seq_lens[lo:hi],
                page_table[lo:hi],
                page_indices[lo:hi],
                page_size,
                ps_bits,
                None if raw_indices is None else raw_indices[lo:hi],
                tile_max[lo:hi],
            )

    def _topk(self, scores, lens, page_table, out, page_size, out_raw) -> None:
        plan = plan_topk_v2(lens, out=self._plan[: lens.shape[0] + 1])
        topk_transform_paged_v2(scores, lens, page_table, out, page_size, plan, out_raw)

    def _select_block(self, sc, ln, pt, op, page_size, ps_bits, raw, tmax) -> None:
        n, K = sc.shape[0], self.topk
        ntiles = self._ntiles[:n]
        _tile_fixup_kernel[(n,)](
            sc, sc.stride(0), ln, tmax, tmax.stride(0), ntiles, num_warps=1
        )
        tsel = _view(self._tile_sel, n, K)
        self._topk(tmax, ntiles, None, tsel, 64, None)
        ctile = self._cand_tiles[: n * CAP_TILES]
        cand = _view(self._cand, n, CAP_TILES * TILE)
        cand_len, fb_len, cnt = self._cand_len[:n], self._fb_len[:n], self._count[:n]
        _compact_kernel[(n,)](
            tmax, tmax.stride(0), tsel, ntiles, ln, ctile, cand_len, fb_len, cnt, K=K,
            CAP=CAP_TILES, BT=_BT, num_warps=4,
        )  # fmt: skip
        _gather_kernel[(n, triton.cdiv(CAP_TILES, _GA_TB))](
            sc, sc.stride(0), ctile, cnt, fb_len, cand, CAP=CAP_TILES, TB=_GA_TB,
            num_warps=4,
        )  # fmt: skip
        csel = _view(self._cand_sel, n, K)
        self._topk(cand, cand_len, None, csel, 64, None)
        # rows past CAP_TILES only (fb_len is 0 elsewhere)
        fbp = _view(self._fb_page, n, K)
        fbr = _view(self._fb_raw, n, K)
        self._topk(sc, fb_len, pt, fbp, page_size, fbr)
        orw = op if raw is None else raw
        _finalize_kernel[(n,)](
            csel, ctile, fb_len, fbp, fbr, pt, pt.stride(0), op, op.stride(0), orw,
            orw.stride(0), CAP=CAP_TILES, K=K, PS_BITS=ps_bits,
            HAS_RAW=raw is not None, num_warps=4,
        )  # fmt: skip
