import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def _controller_norm(
    Dest,
    Message,
    Normed,
    T,
    DEST_STRIDE,
    MESSAGE_STRIDE,
    eps,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    valid = cols < W
    dest = tl.load(Dest + m * DEST_STRIDE + t * W + cols, valid, other=0)
    message = tl.load(Message + m * MESSAGE_STRIDE + t * W + cols, valid, other=0)
    dest_rstd = libdevice.rsqrt(tl.div_rn(tl.sum(dest * dest), W) + eps)
    message_rstd = libdevice.rsqrt(tl.div_rn(tl.sum(message * message), W) + eps)
    out = Normed + (m * T + t) * (2 * W)
    tl.store(out + cols, (dest * dest_rstd).to(tl.bfloat16), valid)
    tl.store(out + W + cols, (message * message_rstd).to(tl.bfloat16), valid)


@triton.jit
def _controller_down(
    Normed,
    Down,
    Partials,
    T,
    R: tl.constexpr,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Per 1024-wide K group partials of normed @ down.T, summed in group order
    unless SPLIT, where each program writes one group's partial."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    GROUPS: tl.constexpr = (K + 1023) // 1024
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    z = tl.program_id(2)
    if SPLIT:
        m = z // GROUPS
        first = z % GROUPS
        count = 1
    else:
        m = z
        first = 0
        count = GROUPS
    out = Partials + (z.to(tl.int64) * T) * R
    X = Normed + (m.to(tl.int64) * T) * K
    Wd = Down + m.to(tl.int64) * R * K
    ks = tl.arange(0, BLOCK_K)
    total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for group in range(first, first + count):
        partial = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for tile in range(1024 // BLOCK_K):
            k = group * 1024 + tile * BLOCK_K + ks
            x = tl.load(
                X + rows[:, None].to(tl.int64) * K + k[None, :],
                (rows[:, None] < T) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                Wd + cols[None, :] * K + k[:, None],
                (cols[None, :] < R) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        if group == first:
            total = partial
        else:
            total = total + partial
    tl.store(
        out + rows[:, None].to(tl.int64) * R + cols[None, :],
        total,
        (rows[:, None] < T) & (cols[None, :] < R),
    )


@triton.jit
def _controller_mix(
    Partials,
    Gain,
    Forget,
    Dest,
    Message,
    Out,
    T,
    DEST_STRIDE,
    MESSAGE_STRIDE,
    W: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COLS: tl.constexpr,
    STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Rows BLOCK_M x columns [pid2 * COLS, (pid2 + 1) * COLS) of controller pid1."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m = tl.program_id(1).to(tl.int64)
    first = tl.program_id(2) * COLS
    rank = tl.arange(0, BLOCK_R)
    row_ok = rows[:, None] < T
    hidden_ok = row_ok & (rank[None, :] < R)
    p = Partials + (m * GROUPS * T + rows[:, None]) * R + rank[None, :]
    hidden = tl.load(p, hidden_ok, other=0)
    for group in tl.static_range(1, GROUPS):
        hidden = hidden + tl.load(p + group * T * R, hidden_ok, other=0)
    hidden = hidden.to(tl.bfloat16)
    for start in tl.range(first, first + COLS, BLOCK_N, num_stages=STAGES):
        cols = start + tl.arange(0, BLOCK_N)
        col_ok = cols[None, :] < W
        weight_ok = col_ok & (rank[:, None] < R)
        weights = m * W * R + cols[None, :] * R + rank[:, None]
        offsets = rows[:, None].to(tl.int64) * W + cols[None, :]
        mask = row_ok & col_ok
        dest = tl.load(Dest + m * DEST_STRIDE + offsets, mask, other=0)
        message = tl.load(Message + m * MESSAGE_STRIDE + offsets, mask, other=0)
        gain = tl.dot(hidden, tl.load(Gain + weights, weight_ok, other=0))
        forget = tl.dot(hidden, tl.load(Forget + weights, weight_ok, other=0))
        output = (1.0 - libdevice.tanh(forget)) * dest + libdevice.tanh(gain) * message
        tl.store(Out + m * T * W + offsets, output, mask)


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _max_shared_mem(device_index: int) -> int:
    props = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return props["max_shared_mem"]


def _stacked(rows: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, int]:
    """Base tensor and per-module element stride of equally shaped fp32 rows,
    without a copy when they already sit at a constant stride (0 for repeats)."""
    base = rows[0]
    assert base.dtype == torch.float32 and base.ndim == 2 and base.is_contiguous()
    assert all(
        r.shape == base.shape and r.dtype == base.dtype and r.is_contiguous()
        for r in rows
    )
    if len(rows) == 1:
        return base, 0
    step = rows[1].data_ptr() - base.data_ptr()
    if step % 4 == 0 and all(
        r.data_ptr() == base.data_ptr() + i * step for i, r in enumerate(rows)
    ):
        return base, step // 4
    return torch.stack(rows), base.numel()


def message_controllers(dests, messages, down, gain, forget, eps=1e-6):
    """Mix messages[i] into dests[i] with controller i of the stacked weights into
    row i of the (len(dests), tokens, width) output:
    (1 - tanh(h @ forget.T)) * dest + tanh(h @ gain.T) * message, where h is the
    bf16 down projection of the bf16 RMS-normalized (dest, message) pair."""
    num, rank, k = down.shape
    assert len(dests) == len(messages) == num
    assert gain.shape == forget.shape == (num, k // 2, rank)
    assert down.dtype == gain.dtype == forget.dtype == torch.bfloat16
    assert down.is_contiguous() and gain.is_contiguous() and forget.is_contiguous()
    dest, dest_stride = _stacked(tuple(dests))
    message, message_stride = _stacked(tuple(messages))
    tokens, width = dests[0].shape
    assert width * 2 == k and messages[0].shape == dests[0].shape
    out = torch.empty((num, tokens, width), dtype=torch.float32, device=dest.device)
    if not tokens:
        return out
    pdl = is_arch_support_pdl()
    launch = {"launch_pdl": True} if pdl else {}
    normed = torch.empty((num, tokens, k), dtype=torch.bfloat16, device=dest.device)
    _controller_norm[(tokens, num)](
        dest,
        message,
        normed,
        tokens,
        dest_stride,
        message_stride,
        eps,
        width,
        triton.next_power_of_2(width),
        pdl,
        num_warps=8,
        enable_fp_fusion=False,
        **launch,
    )
    groups = triton.cdiv(k, 1024)
    decode = tokens <= 16
    split = tokens <= 4096
    block_m = 16 if decode else 64
    block_n = 16 if decode else 64
    block_k = 256 if tokens <= 1024 else 64
    # 64 x 64 tiles with 256-wide K stages need 128 KB of shared memory; SM120 has 99 KB.
    if not decode and block_k == 256 and _max_shared_mem(dest.device.index) < 1 << 17:
        block_k = 128
    partial_groups = groups if split else 1
    partials = torch.empty(
        (num, partial_groups, tokens, rank), dtype=torch.float32, device=dest.device
    )
    _controller_down[
        (triton.cdiv(tokens, block_m), triton.cdiv(rank, block_n), num * partial_groups)
    ](
        normed,
        down,
        partials,
        tokens,
        rank,
        k,
        split,
        block_m,
        block_n,
        block_k,
        pdl,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
        **launch,
    )
    mix_m, mix_n = (16, 64) if decode else (32, 64)
    row_blocks, col_tiles = triton.cdiv(tokens, mix_m), triton.cdiv(width, mix_n)
    tiles_per_program = (
        1
        if decode
        else max(1, col_tiles * row_blocks * num // (12 * _sm_count(dest.device.index)))
    )
    _controller_mix[(row_blocks, num, triton.cdiv(col_tiles, tiles_per_program))](
        partials,
        gain,
        forget,
        dest,
        message,
        out,
        tokens,
        dest_stride,
        message_stride,
        width,
        rank,
        max(16, triton.next_power_of_2(rank)),
        partial_groups,
        mix_m,
        mix_n,
        tiles_per_program * mix_n,
        1 if decode else 3,
        pdl,
        num_warps=4 if decode else 8,
        enable_fp_fusion=False,
        **launch,
    )
    return out
