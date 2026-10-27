"""DSpark kernels for gfx950 (MI350X) only."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.row_argmax import _argmax_pair


@triton.jit
def _fused_kv_gemma_norm_rope_write_kernel(
    kv_ptr,
    kv_row_stride,
    meta_ptr,
    knw_ptr,
    cos_sin_ptr,
    pos_ptr,
    loc_ptr,
    commit_lens_ptr,
    locs_row_width,
    eps,
    D: tl.constexpr,
    NH: tl.constexpr,
    ROT: tl.constexpr,
    HAS_COMMIT_LENS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    l = tl.program_id(1).to(tl.int64)
    if HAS_COMMIT_LENS:
        row_b = t // locs_row_width
        col = t - row_b * locs_row_width
        if col >= tl.load(commit_lens_ptr + row_b).to(tl.int64):
            return
    loc = tl.load(loc_ptr + t).to(tl.int64)
    if loc < 0:
        return
    pos = tl.load(pos_ptr + t).to(tl.int64)

    HALF: tl.constexpr = ROT // 2
    cols = tl.arange(0, D)
    half = tl.arange(0, HALF)
    cos = tl.load(cos_sin_ptr + pos * ROT + half).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * ROT + HALF + half).to(tl.float32)
    k_buf = tl.load(meta_ptr + l * 4 + 0).to(tl.pointer_type(tl.bfloat16))
    v_buf = tl.load(meta_ptr + l * 4 + 1).to(tl.pointer_type(tl.bfloat16))
    k_dst = k_buf + loc * tl.load(meta_ptr + l * 4 + 2)
    v_dst = v_buf + loc * tl.load(meta_ptr + l * 4 + 3)
    row = kv_ptr + t * kv_row_stride + l * (2 * NH * D)
    for h in tl.static_range(NH):
        k_src = row + h * D
        # Gemma RMSNorm as gemma_rmsnorm's one-row-per-program kernel computes it.
        x = tl.load(k_src + cols).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        w1 = tl.load(knw_ptr + l * D + half).to(tl.float32)
        w2 = tl.load(knw_ptr + l * D + HALF + half).to(tl.float32)
        x1 = tl.load(k_src + half).to(tl.float32) * rstd * (1.0 + w1)
        x2 = tl.load(k_src + HALF + half).to(tl.float32) * rstd * (1.0 + w2)
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
        # Neox rope in sgl_kernel.rotary_embedding's bf16 arithmetic: every product
        # and sum rounds to bf16.
        x1c = (x1 * cos).to(tl.bfloat16).to(tl.float32)
        x1s = (x1 * sin).to(tl.bfloat16).to(tl.float32)
        x2c = (x2 * cos).to(tl.bfloat16).to(tl.float32)
        x2s = (x2 * sin).to(tl.bfloat16).to(tl.float32)
        tl.store(k_dst + h * D + half, (x1c - x2s).to(tl.bfloat16))
        tl.store(k_dst + h * D + HALF + half, (x2c + x1s).to(tl.bfloat16))
        if ROT < D:
            tail = ROT + tl.arange(0, D - ROT)
            w3 = tl.load(knw_ptr + l * D + tail).to(tl.float32)
            x3 = tl.load(k_src + tail).to(tl.float32) * rstd * (1.0 + w3)
            tl.store(k_dst + h * D + tail, x3.to(tl.bfloat16))
        tl.store(v_dst + h * D + cols, tl.load(row + NH * D + h * D + cols))


def fused_kv_gemma_norm_rope_write(
    kv: torch.Tensor,
    meta: torch.Tensor,
    k_norm_weights: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    locs: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    commit_lens: Optional[torch.Tensor] = None,
    locs_row_width: Optional[int] = None,
) -> None:
    """``fused_kv_norm_rope_write`` for drafts with a Gemma k-norm (1 + weight) and
    a partial neox rope over the leading ``cos_sin_cache.shape[1]`` dims, rounding as
    GemmaRMSNorm's Triton kernel then ``sgl_kernel.rotary_embedding`` with a bf16
    cache do. ``kv`` is ``[tokens, layers * 2 * num_kv_heads * head_dim]`` bf16 (each
    layer's K heads, then its V heads); ``meta`` holds each layer's
    ``(k_buffer ptr, v_buffer ptr, k row stride, v row stride)``."""
    T = kv.shape[0]
    if T == 0:
        return
    has_commit_lens = commit_lens is not None
    if has_commit_lens != (locs_row_width is not None):
        raise ValueError("commit_lens and locs_row_width must be passed together.")
    if has_commit_lens:
        if commit_lens.numel() * locs_row_width != locs.numel():
            raise ValueError(
                f"locs must be a flattened [{commit_lens.numel()}, "
                f"{locs_row_width}] window, got numel={locs.numel()}."
            )
        commit_lens_arg = commit_lens.contiguous()
    else:
        locs_row_width = 1
        commit_lens_arg = locs
    assert kv.stride(1) == 1 and cos_sin_cache.dtype == torch.bfloat16
    _fused_kv_gemma_norm_rope_write_kernel[(T, meta.shape[0])](
        kv,
        kv.stride(0),
        meta,
        k_norm_weights,
        cos_sin_cache,
        positions.to(torch.int64).contiguous(),
        locs.to(torch.int64).contiguous(),
        commit_lens_arg,
        locs_row_width,
        eps,
        D=head_dim,
        NH=num_kv_heads,
        ROT=cos_sin_cache.shape[1],
        HAS_COMMIT_LENS=has_commit_lens,
        # gemma_rmsnorm's warp count below 1024 columns; it fixes the sum's order
        num_warps=4,
    )


# An index no vocab reaches; exact in fp32, so it rides in the float partials.
_NO_INDEX = 1 << 30


@triton.jit
def _shard_argmax_partial_kernel(
    X,
    P,
    N,
    SX,
    COL0,
    PSTRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    NO_INDEX: tl.constexpr,
):
    # One split of a row of this rank's vocab shard: its (max, lowest global index),
    # both as floats (exact below 2**24), values first in the row's PSTRIDE slots.
    row, part = tl.program_id(0), tl.program_id(1)
    ix = part * BLOCK + tl.arange(0, BLOCK)
    valid = ix < N
    v = tl.load(X + row * SX + ix, valid, float("-inf")).to(tl.float32)
    i = tl.where(valid, ix + COL0, NO_INDEX)
    best_v, best_i = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(P + row * PSTRIDE + part, best_v)
    tl.store(P + row * PSTRIDE + PSTRIDE // 2 + part, best_i.to(tl.float32))


@triton.jit
def _shard_argmax_final_kernel(
    P,
    OUT,
    ROWS,
    PSTRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    NO_INDEX: tl.constexpr,
):
    # Every rank's partials of one row, gathered rank-major: [WORLD, ROWS, PSTRIDE].
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < WORLD * SPLITS
    base = ((o // SPLITS) * ROWS + row) * PSTRIDE + o % SPLITS
    v = tl.load(P + base, m, float("-inf"))
    i = tl.load(P + base + PSTRIDE // 2, m, NO_INDEX).to(tl.int32)
    _, index = tl.reduce((v, i), 0, _argmax_pair)
    tl.store(OUT + row, index.to(tl.int64))


def tp_shard_row_argmax(
    x: torch.Tensor,
    *,
    col_start: int,
    num_valid: int,
    world_size: int,
    all_gather_into_tensor,
) -> torch.Tensor:
    """Row argmax of logits split by columns across a TP group, from each rank's
    ``[rows, shard]`` slice: ``torch.cat(shards, -1)[:, :vocab].float().argmax(-1)`` bit
    for bit (NaN rows and ties included). This rank's columns start at ``col_start``,
    the first ``num_valid`` of them are vocab; every rank passes the same ``x`` shape.
    ``all_gather_into_tensor(out, inp)`` gathers the small per-split partials
    rank-major along dim 0."""
    assert x.dim() == 2 and x.stride(1) == 1
    rows, shard = x.shape
    assert world_size * shard < 1 << 24, "the partials carry indices as fp32"
    block = 8192
    splits = triton.cdiv(shard, block)
    # values, then indices; 16-byte rows for the custom all-gather
    pstride = max(4, triton.next_power_of_2(2 * splits))
    partials = torch.empty((rows, pstride), dtype=torch.float32, device=x.device)
    _shard_argmax_partial_kernel[(rows, splits)](
        x,
        partials,
        num_valid,
        x.stride(0),
        col_start,
        pstride,
        block,
        _NO_INDEX,
        num_warps=4,
    )
    gathered = torch.empty(
        (world_size * rows, pstride), dtype=torch.float32, device=x.device
    )
    all_gather_into_tensor(gathered, partials)
    out = torch.empty((rows,), dtype=torch.int64, device=x.device)
    _shard_argmax_final_kernel[(rows,)](
        gathered,
        out,
        rows,
        pstride,
        splits,
        world_size,
        triton.next_power_of_2(world_size * splits),
        _NO_INDEX,
        num_warps=1,
    )
    return out
