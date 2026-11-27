# SPDX-License-Identifier: Apache-2.0
"""QK RMSNorm + NeoX RoPE written straight into the Ulysses exchange blocks (ROCm).

One pass replaces the per-tensor ``aten_head_rmsnorm`` (with RoPE), the
destination-major q|k|v pack and the copy of this rank's own blocks into its
receive slots: every (token, head) row of q and k gets exactly the arithmetic
of ``_aten_head_rmsnorm_kernel`` (same loads, the same fp32 lane sums and
shfl-down fold, ``rsqrt``, ``w * (rstd * x)``, the bf16-rounded RoPE products),
only stored at its block of the exchange instead of a contiguous copy; v rows
are moved unchanged. So the blocks hold the bytes the norm-then-pack path puts
there.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

_HEAD_DIM = 128
_ROWS_PER_PROG = 16


@triton.jit
def _norm_rope_rows(
    base,
    row_mask,
    tokens,
    weight_ptr,
    cache_ptr,
    positions_ptr,
    eps,
    cache_stride,
    head_dim: tl.constexpr,
    rows_per_prog: tl.constexpr,
    rope_half: tl.constexpr,
):
    # the body of _aten_head_rmsnorm_kernel (norm/aten_head_rmsnorm_triton.py)
    offs = tl.arange(0, head_dim)
    x = tl.load(base[:, None] + offs[None, :], mask=row_mask[:, None], other=0.0)
    x = x.to(tl.float32)
    lanes = tl.reshape(x, (rows_per_prog, head_dim // 4, 2, 2), can_reorder=False)
    even, odd = tl.split(lanes)
    x0, x2 = tl.split(even)
    x1, x3 = tl.split(odd)
    acc = x0 * x0
    acc = acc + x1 * x1
    acc = acc + x2 * x2
    acc = acc + x3 * x3
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 16), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 2), can_reorder=False), axis=1)
    mean = tl.sum(acc, axis=1) / head_dim
    rstd = tl.rsqrt(mean + eps)

    weight = tl.load(weight_ptr + offs).to(tl.float32)
    y = (weight[None, :] * (rstd[:, None] * x)).to(tl.bfloat16)
    if rope_half > 0:
        rotated = offs < 2 * rope_half
        first = offs < rope_half
        partner = tl.where(first, offs + rope_half, offs - rope_half)
        rope_mask = row_mask[:, None] & rotated[None, :]
        xp = tl.load(base[:, None] + partner[None, :], mask=rope_mask, other=0.0)
        wp = tl.load(weight_ptr + partner, mask=rotated, other=0.0).to(tl.float32)
        yp = (wp[None, :] * (rstd[:, None] * xp.to(tl.float32))).to(tl.bfloat16)
        cache_row = tl.load(positions_ptr + tokens, mask=row_mask) * cache_stride
        cos_offs = cache_row[:, None] + (offs % rope_half)[None, :]
        cos = tl.load(cache_ptr + cos_offs, mask=rope_mask, other=0.0)
        sin = tl.load(cache_ptr + cos_offs + rope_half, mask=rope_mask, other=0.0)
        a = (y.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        b = (yp.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        rope = tl.where(first[None, :], a - b, a + b).to(tl.bfloat16)
        y = tl.where(rotated[None, :], rope, y)
    return y


@triton.jit
def _qkv_fill_kernel(
    send_ptr,
    own_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    wq_ptr,
    wk_ptr,
    cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    v_stride,
    n_tokens,
    n_rows,
    eps_q,
    eps_k,
    cache_stride,
    rank,
    g0,
    world: tl.constexpr,
    groups: tl.constexpr,
    group_heads: tl.constexpr,
    launch_groups: tl.constexpr,
    head_dim: tl.constexpr,
    rows_per_prog: tl.constexpr,
    rope_half: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_prog + tl.arange(0, rows_per_prog)
    row_mask = rows < n_rows
    per_token: tl.constexpr = launch_groups * world * group_heads
    tokens = (rows // per_token).to(tl.int64)
    j = rows % per_token
    g = g0 + j // (world * group_heads)
    j2 = j % (world * group_heads)
    w = j2 // group_heads
    hl = j2 % group_heads
    head = (w * groups + g) * group_heads + hl
    offs = tl.arange(0, head_dim)

    yq = _norm_rope_rows(
        q_ptr + tokens * q_stride + head * head_dim,
        row_mask,
        tokens,
        wq_ptr,
        cache_ptr,
        positions_ptr,
        eps_q,
        cache_stride,
        head_dim,
        rows_per_prog,
        rope_half,
    )
    yk = _norm_rope_rows(
        k_ptr + tokens * k_stride + head * head_dim,
        row_mask,
        tokens,
        wk_ptr,
        cache_ptr,
        positions_ptr,
        eps_k,
        cache_stride,
        head_dim,
        rows_per_prog,
        rope_half,
    )
    vv = tl.load(
        v_ptr + (tokens * v_stride + head * head_dim)[:, None] + offs[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    # block (w, g) row `token`, head hl: [q | k | v] of head_dim each.
    # Peers' blocks are destination-major [world, groups, tokens, gh, 3 * hd]
    # in the send buffer; this rank's go to its receive slot [groups, world, ...]
    w64 = w.to(tl.int64)
    g64 = g.to(tl.int64)
    send_row = ((w64 * groups + g64) * n_tokens + tokens) * group_heads + hl
    own_row = ((g64 * world + w64) * n_tokens + tokens) * group_heads + hl
    mine = w == rank
    dst_send = send_ptr + send_row * (3 * head_dim)
    dst_own = own_ptr + own_row * (3 * head_dim)
    send_mask = row_mask & (w != rank)
    own_mask = row_mask & mine
    sm = send_mask[:, None]
    om = own_mask[:, None]
    tl.store(dst_send[:, None] + offs[None, :], yq, mask=sm)
    tl.store(dst_own[:, None] + offs[None, :], yq, mask=om)
    tl.store(dst_send[:, None] + (head_dim + offs)[None, :], yk, mask=sm)
    tl.store(dst_own[:, None] + (head_dim + offs)[None, :], yk, mask=om)
    tl.store(dst_send[:, None] + (2 * head_dim + offs)[None, :], vv, mask=sm)
    tl.store(dst_own[:, None] + (2 * head_dim + offs)[None, :], vv, mask=om)


def _token_stride(x: torch.Tensor) -> int | None:
    if x.dim() != 3 or x.stride(-1) != 1 or x.stride(-2) != _HEAD_DIM:
        return None
    return x.stride(0)


def can_fill(q, k, v, wq, wk, cos_sin_cache, positions) -> bool:
    return (
        torch.version.hip is not None
        and q.is_cuda
        and q.dim() == 3
        and q.shape == k.shape == v.shape
        and q.shape[-1] == _HEAD_DIM
        and q.dtype == k.dtype == v.dtype == wq.dtype == wk.dtype == torch.bfloat16
        and wq.shape == wk.shape == (_HEAD_DIM,)
        and wq.is_contiguous()
        and wk.is_contiguous()
        and all(_token_stride(t) is not None for t in (q, k, v))
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.dtype == q.dtype
        and cos_sin_cache.stride(-1) == 1
        and 0 < cos_sin_cache.shape[-1] <= _HEAD_DIM
        and cos_sin_cache.shape[-1] % 2 == 0
        and positions.dtype == torch.int64
        and positions.is_contiguous()
        and positions.numel() == q.shape[0]
    )


def qkv_fill(
    send: torch.Tensor,
    own: torch.Tensor,
    q,
    k,
    v,
    wq,
    wk,
    eps_q: float,
    eps_k: float,
    cos_sin_cache,
    positions,
    *,
    rank: int,
    world: int,
    groups: int,
    first_group: int,
    n_groups: int,
) -> None:
    """Write groups [first_group, first_group + n_groups) of q|k|v: peers' blocks
    into ``send`` ([world, groups, tokens, gh, 3 * hd]), this rank's into
    ``own`` ([groups, world, tokens, gh, 3 * hd], its receive slots)."""
    tokens, heads, hd = q.shape
    gh = heads // (world * groups)
    n_rows = tokens * n_groups * world * gh
    grid = (triton.cdiv(n_rows, _ROWS_PER_PROG),)
    with torch.get_device_module().device(q.device):
        _qkv_fill_kernel[grid](
            send,
            own,
            q,
            k,
            v,
            wq,
            wk,
            cos_sin_cache,
            positions,
            q.stride(0),
            k.stride(0),
            v.stride(0),
            tokens,
            n_rows,
            eps_q,
            eps_k,
            cos_sin_cache.stride(0),
            rank,
            first_group,
            world=world,
            groups=groups,
            group_heads=gh,
            launch_groups=n_groups,
            head_dim=_HEAD_DIM,
            rows_per_prog=_ROWS_PER_PROG,
            rope_half=cos_sin_cache.shape[-1] // 2,
            num_warps=4,
        )
