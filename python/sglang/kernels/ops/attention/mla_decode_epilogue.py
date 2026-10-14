"""MLA decode epilogue in one launch: split-KV combine, value up-projection, output gate.

For aiter's persistent MLA decode stage 1 (head-padded, split KV), one program per
(query row, head) combines that row's split partials, projects the latent result
through W_vc and applies Kimi-K3's output gate:
    attn[t, h] = bf16(sum_j exp(lse_j - lse) * partial_j[t, h])   rows a reduce
                 tile covers; else stage 1's own output o[t, h]
    out[t, h]  = bf16(bf16(attn[t, h] @ w_vc[h]) [* bf16(sigmoid(gate[t, h]))])
with the rounding points of mla_split_reduce + mla_v_up_gate (the split
summation order differs). A 16-wave program loads every split of its row in
one [splits, latent] tile, so no latent intermediate is written or reread.

Partial layout (aiter mla_decode_fwd, persistent mode):
    partial_output [P * max_q, 1, HP, L] fp32,  partial_lse [P * max_q, 1, HP, 1] fp32
For reduce tile i the splits are reduce_partial_map[reduce_indptr[i]:reduce_indptr[i+1]];
row l of the tile (final row reduce_final_map[i][0] + l) reads partial row map + l.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

# Splits one program combines in a single tile (SGLANG_AITER_MLA_MAX_SPLITS_PER_BATCH=128).
MAX_SPLITS: int = 128
MIN_ROWS: int = 1
MAX_ROWS: int = 64


@triton.jit
def _global_lse(max_lse, sum_lse):
    # aiter's guard: an empty or NaN sum gives +inf, which zeroes the row.
    return tl.where(
        (sum_lse == 0.0) | (sum_lse != sum_lse), float("inf"), tl.log(sum_lse) + max_lse
    )


@triton.jit
def _mla_decode_epilogue_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    reduce_indptr_ptr,
    reduce_final_map_ptr,
    reduce_partial_map_ptr,
    num_tiles,
    o_ptr,  # [S, HP, L]: stage 1's output for rows no reduce tile covers
    stride_os,
    stride_oh,
    w_ptr,  # [H, L, V]
    stride_wh,
    stride_wl,
    stride_wv,
    gate_ptr,  # [S, H * V]
    stride_gt,
    out_ptr,  # [S, H * V]
    stride_ot,
    HP: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    v0 = tl.program_id(2) * BLOCK_V

    # The reduce tile covering row t, from one vector load of the tile map.
    offs_tile = tl.arange(0, BLOCK_TILES)
    tile_ok = offs_tile < num_tiles
    q_start = tl.load(reduce_final_map_ptr + offs_tile * 2, mask=tile_ok, other=0)
    q_end = tl.load(reduce_final_map_ptr + offs_tile * 2 + 1, mask=tile_ok, other=0)
    hit = tile_ok & (q_start <= t) & (t < q_end)
    tile = tl.max(tl.where(hit, offs_tile, -1), axis=0)
    l = t - tl.max(tl.where(hit, q_start, 0), axis=0)

    offs_l = tl.arange(0, L)
    if tile >= 0:
        start = tl.load(reduce_indptr_ptr + tile)
        n = tl.load(reduce_indptr_ptr + tile + 1) - start
        offs_s = tl.arange(0, BLOCK_S)
        if n <= BLOCK_S:
            # Every split of the row in one [BLOCK_S, L] tile.
            mask_s = offs_s < n
            rows = (
                tl.load(reduce_partial_map_ptr + start + offs_s, mask=mask_s, other=0)
                + l
            )
            lse = tl.load(
                partial_lse_ptr + rows * HP + h, mask=mask_s, other=-float("inf")
            )
            max_lse = tl.max(lse, axis=0)
            sum_lse = tl.sum(tl.where(mask_s, tl.exp(lse - max_lse), 0.0), axis=0)
            scale = tl.where(mask_s, tl.exp(lse - _global_lse(max_lse, sum_lse)), 0.0)
            part = tl.load(
                partial_out_ptr + (rows[:, None] * HP + h) * L + offs_l[None, :],
                mask=mask_s[:, None],
                other=0.0,
            )
            attn = tl.sum(scale[:, None] * part, axis=0)
        else:
            # More splits than the tile: the global LSE first, then BLOCK_S at a time.
            max_lse = -float("inf")
            for c0 in range(0, n, BLOCK_S):
                mask_s = c0 + offs_s < n
                rows = (
                    tl.load(
                        reduce_partial_map_ptr + start + c0 + offs_s,
                        mask=mask_s,
                        other=0,
                    )
                    + l
                )
                lse = tl.load(
                    partial_lse_ptr + rows * HP + h, mask=mask_s, other=-float("inf")
                )
                max_lse = tl.maximum(max_lse, tl.max(lse, axis=0))
            sum_lse = 0.0
            for c0 in range(0, n, BLOCK_S):
                mask_s = c0 + offs_s < n
                rows = (
                    tl.load(
                        reduce_partial_map_ptr + start + c0 + offs_s,
                        mask=mask_s,
                        other=0,
                    )
                    + l
                )
                lse = tl.load(
                    partial_lse_ptr + rows * HP + h, mask=mask_s, other=-float("inf")
                )
                sum_lse += tl.sum(tl.where(mask_s, tl.exp(lse - max_lse), 0.0), axis=0)
            global_lse = _global_lse(max_lse, sum_lse)
            attn = tl.zeros([L], dtype=tl.float32)
            for c0 in range(0, n, BLOCK_S):
                mask_s = c0 + offs_s < n
                rows = (
                    tl.load(
                        reduce_partial_map_ptr + start + c0 + offs_s,
                        mask=mask_s,
                        other=0,
                    )
                    + l
                )
                lse = tl.load(
                    partial_lse_ptr + rows * HP + h, mask=mask_s, other=-float("inf")
                )
                scale = tl.where(mask_s, tl.exp(lse - global_lse), 0.0)
                part = tl.load(
                    partial_out_ptr + (rows[:, None] * HP + h) * L + offs_l[None, :],
                    mask=mask_s[:, None],
                    other=0.0,
                )
                attn += tl.sum(scale[:, None] * part, axis=0)
    else:
        attn = tl.load(o_ptr + t * stride_os + h * stride_oh + offs_l).to(tl.float32)
    attn = attn.to(tl.bfloat16).to(tl.float32)

    offs_v = v0 + tl.arange(0, BLOCK_V)
    w = tl.load(
        w_ptr
        + h * stride_wh
        + offs_l[:, None] * stride_wl
        + offs_v[None, :] * stride_wv
    ).to(tl.float32)
    acc = tl.sum(attn[:, None] * w, axis=0)
    y = acc.to(tl.bfloat16)
    cols = h * V + offs_v
    if HAS_GATE:
        g = tl.load(gate_ptr + t * stride_gt + cols).to(tl.float32)
        sig = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16)
        y = (y.to(tl.float32) * sig.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + t * stride_ot + cols, y)


def covered(rows: int, w_vc: torch.Tensor, gate: Optional[torch.Tensor]) -> bool:
    """Whether mla_decode_epilogue takes rows query rows with w_vc [H, L, V]."""
    H, L, V = w_vc.shape
    return (
        MIN_ROWS <= rows <= MAX_ROWS
        and w_vc.dtype == torch.bfloat16
        and triton.next_power_of_2(L) == L
        and triton.next_power_of_2(V) == V
        and V % 4 == 0
        and (gate is None or (gate.shape == (rows, H * V) and gate.stride(-1) == 1))
    )


def _launch_config(rows: int, max_q: int, max_splits: int) -> tuple[int, int, int]:
    """(split tile, output column programs per (row, head), warps). Measured on
    MI350X (K3 TP8, 24 rotating layers) at the planner's split counts."""
    # aiter's persistent planner gives a request about CUs / requests + 1 splits
    # (capped by max_splits); a row with more is combined in chunks of the tile.
    requests = max(rows // max(max_q, 1), 1)
    expected = min(max_splits, _num_cus() // requests + 1)
    block_s = max(triton.next_power_of_2(min(expected, MAX_SPLITS)), 2)
    if block_s >= 128:
        # ~125 splits (1-2 requests) want the most programs; ~65 (4 requests) fewer.
        if expected > 96:
            return block_s, 4 if rows <= 4 else 2, 8
        return block_s, 2 if rows <= 4 else 1, 8
    if block_s == 64:
        return block_s, 2, 4
    return block_s, 1, 4


@functools.cache
def _num_cus() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def mla_decode_epilogue(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    o: torch.Tensor,
    w_vc: torch.Tensor,
    gate: Optional[torch.Tensor],
    *,
    max_q: int = 1,
    max_splits: int = MAX_SPLITS,
) -> torch.Tensor:
    """The output [S, H * V] bf16 of aiter's persistent stage 1 (partials + o
    [S, HP, L] holding the rows no reduce tile covers) through w_vc [H, L, V],
    gated by sigmoid(gate) [S, H * V] when given. max_splits bounds the splits of
    one row only to size the tile; more are combined in chunks. The caller
    checked covered()."""
    rows = o.shape[0]
    H, L, V = w_vc.shape
    HP = partial_output.shape[-2]
    num_tiles = reduce_indptr.shape[0] - 1
    out = torch.empty((rows, H * V), dtype=torch.bfloat16, device=o.device)
    assert reduce_final_map.is_contiguous() and reduce_final_map.shape[-1] == 2
    block_s, v_split, num_warps = _launch_config(rows, max_q, max_splits)
    gate_arg = gate if gate is not None else out
    _mla_decode_epilogue_kernel[(rows, H, v_split)](
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        num_tiles,
        o,
        o.stride(0),
        o.stride(1),
        w_vc,
        w_vc.stride(0),
        w_vc.stride(1),
        w_vc.stride(2),
        gate_arg,
        gate_arg.stride(0),
        out,
        out.stride(0),
        HP=HP,
        L=L,
        V=V,
        BLOCK_S=block_s,
        BLOCK_TILES=max(triton.next_power_of_2(max(num_tiles, 1)), 2),
        BLOCK_V=V // v_split,
        HAS_GATE=gate is not None,
        num_warps=num_warps,
    )
    return out
