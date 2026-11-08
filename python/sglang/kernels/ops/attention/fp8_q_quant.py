"""Per-tensor FP8 query quantization for the Muse Spark extend attention, in two launches instead of the torch chain

    q_descale = (q.abs().amax().float() / 448.0).clamp_min(1e-12).view(1)
    q_fp8 = (q.float() / q_descale).to(torch.float8_e4m3fn)

(abs, amax, casts, divisions, clamp and the FP8 cast: ~8 launches and ~5 full passes over q).
Launch 1 writes one |q| max per block; launch 2 reduces the block maxima (a max is exact and order-free), forms the
scale with the chain's own fp32 operations (ATen divides by the Python scalar 448 as a multiplication by its
correctly rounded fp32 reciprocal; clamp at the fp32 value of 1e-12) and quantizes its block with a correctly
rounded division and the RNE e4m3fn cast. Same bytes as the chain.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 8192
_PART_CHUNK = 1024


@triton.jit(do_not_specialize=["N"])
def _q_absmax_kernel(Q, PART, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Q + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(PART + pid, tl.max(tl.abs(x), axis=0))


@triton.jit(do_not_specialize=["N", "NPART"])
def _q_quant_kernel(
    Q, PART, OUT, DESCALE, N, NPART, BLOCK: tl.constexpr, PART_CHUNK: tl.constexpr
):
    pid = tl.program_id(0)
    amax = tl.zeros((), dtype=tl.float32)
    for start in range(0, NPART, PART_CHUNK):
        p = start + tl.arange(0, PART_CHUNK)
        part = tl.load(PART + p, mask=p < NPART, other=0.0)
        amax = tl.maximum(amax, tl.max(part, axis=0))
    descale = tl.maximum(amax * tl.math.div_rn(1.0, 448.0), 1e-12)
    if pid == 0:
        tl.store(DESCALE, descale)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(Q + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + offs, tl.math.div_rn(x, descale).to(OUT.dtype.element_ty), mask=m)


def fp8_q_quant_supported(q: torch.Tensor) -> bool:
    return q.dtype == torch.bfloat16 and q.is_contiguous() and 0 < q.numel() < 2**31


def fp8_q_quant(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(q as float8_e4m3fn, descale [1] fp32): the bytes of the torch chain in the module docstring."""
    n = q.numel()
    parts = triton.cdiv(n, _BLOCK)
    part = torch.empty(parts, dtype=torch.float32, device=q.device)
    out = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    descale = torch.empty(1, dtype=torch.float32, device=q.device)
    _q_absmax_kernel[(parts,)](q, part, n, BLOCK=_BLOCK, num_warps=8)
    _q_quant_kernel[(parts,)](
        q,
        part,
        out,
        descale,
        n,
        parts,
        BLOCK=_BLOCK,
        PART_CHUNK=_PART_CHUNK,
        num_warps=8,
    )
    return out, descale
