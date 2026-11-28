"""gfx950 query RoPE fused into the query's MXFP8 quant for the V4.1 sparse attention: q is read
once instead of roped in place by the K store launch and then re-read by fp8_grid_quantize.

The RoPE is bit-equal to the K launch's in-place query rope (``main_norm_rope.cuh``, kRopeQ):
interleaved pairs on the trailing ``ROPE`` dims, the sine products rounded, then one fma with the
cosine, the result rounded to bf16. The quant is ``fp8_grid_quant`` on that bf16 head."""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_quant

D = 512
GROUPS = D // 32


@triton.jit
def _rope_tail(q_ptr, freqs_ptr, pos, offs_tail, rmask, stride_f, ROPE: tl.constexpr):
    """The rope tail [BLOCK_M, ROPE] of each row rotated (kRopeQ order), as fp32 on the bf16 grid."""
    x = tl.load(q_ptr + offs_tail, mask=rmask[:, None], other=0.0).to(tl.float32)
    re, im = tl.split(tl.reshape(x, (x.shape[0], ROPE // 2, 2)))
    f = tl.load(
        freqs_ptr + pos[:, None] * stride_f + tl.arange(0, ROPE)[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    cos, sin = tl.split(tl.reshape(f, (x.shape[0], ROPE // 2, 2)))
    out_re = tl.fma(re, cos, -(im * sin)).to(tl.bfloat16).to(tl.float32)
    out_im = tl.fma(im, cos, re * sin).to(tl.bfloat16).to(tl.float32)
    return tl.reshape(tl.join(out_re, out_im), (x.shape[0], ROPE))


@triton.jit
def _q_rope_quant_kernel(
    q_ptr,
    freqs_ptr,
    pos_ptr,
    q8_ptr,
    scale_ptr,
    rows,
    num_heads,
    stride_qt,
    stride_qh,
    stride_f,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROPE_ONLY: tl.constexpr,
):
    """Program: BLOCK_M (token, head) rows. The whole row is quantized as is and stored up to the
    rope tail; the tail is reloaded, rotated, quantized (its own two-plus groups) and stored.
    ROPE_ONLY: the rotated tail goes back into q as bf16."""
    r = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = r < rows
    t = r // num_heads
    base = t.to(tl.int64) * stride_qt + (r % num_heads).to(tl.int64) * stride_qh
    pos = tl.load(pos_ptr + t, mask=rmask, other=0).to(tl.int64)
    tcol = (D - ROPE) + tl.arange(0, ROPE)
    offs_tail = base[:, None] + tcol[None, :]
    y = _rope_tail(q_ptr, freqs_ptr, pos, offs_tail, rmask, stride_f, ROPE)
    if ROPE_ONLY:
        tl.store(q_ptr + offs_tail, y.to(tl.bfloat16), mask=rmask[:, None])
    else:
        r64 = r.to(tl.int64)
        col = tl.arange(0, D)
        x = tl.load(
            q_ptr + base[:, None] + col[None, :], mask=rmask[:, None], other=0.0
        )
        q8, e8 = fp8_grid_quant(tl.reshape(x.to(tl.float32), (BLOCK_M * (D // 32), 32)))
        head_cols = (col < D - ROPE)[None, :]
        tl.store(
            q8_ptr + r64[:, None] * D + col[None, :],
            tl.reshape(q8, (BLOCK_M, D)),
            mask=rmask[:, None] & head_cols,
        )
        g = tl.arange(0, D // 32)
        tl.store(
            scale_ptr + r64[:, None] * (D // 32) + g[None, :],
            tl.reshape(e8, (BLOCK_M, D // 32)).to(tl.uint8),
            mask=rmask[:, None] & (g < (D - ROPE) // 32)[None, :],
        )
        tq8, te8 = fp8_grid_quant(tl.reshape(y, (BLOCK_M * (ROPE // 32), 32)))
        tl.store(
            q8_ptr + r64[:, None] * D + tcol[None, :],
            tl.reshape(tq8, (BLOCK_M, ROPE)),
            mask=rmask[:, None],
        )
        tg = (D - ROPE) // 32 + tl.arange(0, ROPE // 32)
        tl.store(
            scale_ptr + r64[:, None] * (D // 32) + tg[None, :],
            tl.reshape(te8, (BLOCK_M, ROPE // 32)).to(tl.uint8),
            mask=rmask[:, None],
        )


def _check(q, freqs_real, positions):
    n, h, d = q.shape
    assert d == D and q.dtype == torch.bfloat16 and q.stride(2) == 1, (q.shape, q.dtype)
    assert freqs_real.dtype == torch.float32 and freqs_real.stride(-1) == 1
    rope = freqs_real.shape[-1]
    assert rope % 32 == 0 and rope <= d, rope
    assert positions.shape[0] >= n, (positions.shape, n)
    return n, h, rope


def rope_fp8_grid_quantize(
    q: torch.Tensor,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    block_m: int = 64,
    num_warps: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [n, h, 512] bf16 with its trailing rope dims not yet rotated, freqs_real [max_pos, rope]
    fp32 (cos / sin interleaved), positions [n] -> fp8_grid_quantize(rope(q).reshape(n * h, 512)):
    (e4m3 [n * h, 512], ue8m0 [n * h, 16] uint8). q is left as it was."""
    n, h, rope = _check(q, freqs_real, positions)
    q8 = torch.empty((n * h, D), dtype=torch.float8_e4m3fn, device=q.device)
    scale = torch.empty((n * h, GROUPS), dtype=torch.uint8, device=q.device)
    if n:
        _q_rope_quant_kernel[(triton.cdiv(n * h, block_m),)](
            q,
            freqs_real,
            positions,
            q8,
            scale,
            n * h,
            h,
            q.stride(0),
            q.stride(1),
            freqs_real.stride(0),
            D=D,
            ROPE=rope,
            BLOCK_M=block_m,
            ROPE_ONLY=False,
            num_warps=num_warps,
        )
    return q8, scale


def rope_q_inplace(
    q: torch.Tensor, freqs_real: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Rotate q's trailing rope dims in place, bit-equal to the K store launch's query rope."""
    n, h, rope = _check(q, freqs_real, positions)
    if n:
        block_m = 64
        _q_rope_quant_kernel[(triton.cdiv(n * h, block_m),)](
            q,
            freqs_real,
            positions,
            q,
            q,
            n * h,
            h,
            q.stride(0),
            q.stride(1),
            freqs_real.stride(0),
            D=D,
            ROPE=rope,
            BLOCK_M=block_m,
            ROPE_ONLY=True,
            num_warps=4,
        )
    return q
