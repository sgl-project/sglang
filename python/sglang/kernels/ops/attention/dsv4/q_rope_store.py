"""Rotate Q while writing the padded buffer consumed by sparse attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _q_rope_store(X, Y, F, POS, SX: tl.constexpr, SY: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    r = tl.arange(0, 512)
    value = tl.load(X + row * SX + head * 512 + r).to(tl.float32)
    partner = tl.gather(value, r ^ 1, 0)
    position = tl.load(POS + row)
    cos = tl.load(F + position * 64 + (r - 448) // 2 * 2, r >= 448, 0)
    sin = tl.load(F + position * 64 + (r - 448) // 2 * 2 + 1, r >= 448, 0)
    # Match the operation order of deepseek_rope_kernel before BF16 rounding.
    even = tl.fma(value, cos, -partner * sin)
    odd = tl.fma(partner, sin, value * cos)
    rotated = tl.where((r & 1) == 0, even, odd)
    tl.store(Y + row * SY + head * 512 + r, tl.where(r >= 448, rotated, value))


@triton.jit
def _q_rope_store_prefill(
    X,
    Y,
    F,
    POS,
    M,
    H: tl.constexpr,
    SX: tl.constexpr,
    SY: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    # Keep token count dynamic to avoid compiling every prefill batch length.
    heads = tl.program_id(0) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    row, head = heads // H, heads % H
    r = tl.arange(0, 512)
    # The padded output can exceed 2 GiB at the 64K prefill ceiling.
    x_offset = row[:, None].to(tl.int64) * SX + head[:, None] * 512 + r[None, :]
    value = tl.load(X + x_offset, row[:, None] < M, 0).to(tl.float32)
    partner = tl.gather(value, tl.broadcast_to((r ^ 1)[None, :], (BLOCK_HEADS, 512)), 1)
    position = tl.load(POS + row, row < M, 0)
    freq_offset = position[:, None].to(tl.int64) * 64 + (r[None, :] - 448) // 2 * 2
    rope_mask = (row[:, None] < M) & (r[None, :] >= 448)
    cos = tl.load(F + freq_offset, rope_mask, 0)
    sin = tl.load(F + freq_offset + 1, rope_mask, 0)
    # Keep the same arithmetic and BF16 rounding as the decode kernel.
    even = tl.fma(value, cos, -partner * sin)
    odd = tl.fma(partner, sin, value * cos)
    rotated = tl.where((r[None, :] & 1) == 0, even, odd)
    y_offset = row[:, None].to(tl.int64) * SY + head[:, None] * 512 + r[None, :]
    tl.store(
        Y + y_offset,
        tl.where(r[None, :] >= 448, rotated, value),
        row[:, None] < M,
    )


def q_rope_store(
    q: torch.Tensor,
    output: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Apply 64-wide forward RoPE to 512-wide heads without changing Q padding."""
    assert q.shape == output.shape and q.ndim == 3 and q.shape[2] == 512
    assert q.dtype == output.dtype == torch.bfloat16
    assert q.stride(2) == output.stride(2) == 1
    assert q.stride(1) == output.stride(1) == 512
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous()
    assert freqs_cis.shape[1] == 32 and positions.shape == (q.shape[0],)
    assert positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
    if q.shape[0] >= 4096 and q.shape[1] == 16:
        _q_rope_store_prefill[(triton.cdiv(q.shape[0] * q.shape[1], 4),)](
            q,
            output,
            torch.view_as_real(freqs_cis),
            positions,
            q.shape[0],
            q.shape[1],
            q.stride(0),
            output.stride(0),
            BLOCK_HEADS=4,
            num_warps=4,
        )
        return
    _q_rope_store[(q.shape[0], q.shape[1])](
        q,
        output,
        torch.view_as_real(freqs_cis),
        positions,
        q.stride(0),
        output.stride(0),
        num_warps=4,
    )
