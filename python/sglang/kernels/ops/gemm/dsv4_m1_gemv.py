# SPDX-License-Identifier: Apache-2.0
"""SM80 (A100) M=1 FP8-E4M3 group-32 GEMV over raw row-major weights.

One token, one dense FP8 projection, no repack: the kernel streams the RAW E4M3
bytes (N, K) row-major plus the PLAIN bf16 group-32 model scales (N, K // 32)
and writes a bf16 (N,) vector. Single token (M == 1) only.

NUMERICAL CONTRACT (identical to the SM80 FP8 Marlin path)
----------------------------------------------------------
For output column n and k group g = k // 32:

    w_eff[n, k] = bf16( bf16(fp8_value[n, k]) * bf16(scale[n, g]) )
    out[n]      = bf16( sum_k fp32( w_eff[n, k] * x[k] ) )

Both operands are BF16, the product is rounded to BF16 exactly once, the
accumulation is FP32 and the result is stored as BF16. ``scale`` is the plain
model scale, NOT the 2**120-folded copy Marlin stores.

Triton has no native sm80 fp8 cast, so the payload is rebuilt bit-exactly from
the raw byte with an exponent-bias fold:

    bits = ((raw << 8) & 0x8000) | ((raw << 4) & 0x07F0)
    wv   = bits.to(int16).to(bf16, bitcast=True)      # == fp8_value * 2**-120

That shift is exact for every finite E4M3 code, subnormals included: fp8's
exponent bias 7 and bf16's 127 differ by exactly 120, and E4M3 -> BF16 is a
pure bit-field move (0x7F / 0xFF land on the two bf16 NaN patterns). Scaling
the BF16 model scale by 2**120 cancels the fold, so

    wk = bf16( wv * bf16(scale * 2**120) )

is the single contract rounding: the fold is a power-of-two rescale of each
BF16 factor, so it changes neither rounding. The fold exists because a literal
per-element ``* 2**120`` pushes most payloads into the bf16 subnormal range,
where Triton's fp32 arithmetic (compiled with FTZ) would flush them to zero.

FOLD RANGE: ``bf16(scale * 2**120)`` is exact for ``|scale| < 2**8``; at or
above that the folded scale overflows to inf in bf16. The SM80 Marlin path
applies the identical fold in ``fp8_fused_exponent_bias_into_scales``, so this
is the same bound the production path already carries, not a new restriction.

SHAPES AND CONFIGS
------------------
The (NCOL, BKK, num_warps) triples in ``_EXACT_CONFIGS`` are the measured
winners on A100-SXM4-80GB for the five dense DeepSeek-V4.1-Flash projections
(one program owns NCOL output columns, keeps an (NCOL, BKK) fp32 register
accumulator and reduces once at the end). The weight tile is reshaped to
(NCOL, BKK // 32, 32) so the group-32 scale is loaded as (NCOL, BKK // 32) and
broadcast 32-wide, i.e. read once per group instead of once per element.

Any other (N, K) with N % ncol == 0 and K % bkk == 0 can use the generic
fallback below; when neither the measured nor the generic tiling divides the
shape, ``dsv41_m1_gemv_eligible`` declines so the caller stays on Marlin.

This module is inert on its own: the production gate lives in
``sglang.srt.layers.quantization.marlin_utils_fp8`` and requires both
``SGLANG_DSV41_SM80_M1_GEMV=1`` and ``SGLANG_DSV41_SM80_EXPERIMENTAL=1``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

__all__ = [
    "dsv41_m1_gemv",
    "dsv41_m1_gemv_config",
    "dsv41_m1_gemv_eligible",
]

# Group-32 block quantization along K: one bf16 scale per 32 raw FP8 weights.
GROUP = 32

# (N, K) -> (NCOL, BKK, num_warps), measured on A100-SXM4-80GB (SM80) for the
# five dense DeepSeek-V4.1-Flash shapes. (5120, 320) is the shared-expert down
# projection with its K padded 288 -> 320 by the SM80 Marlin K-tail padding.
#
# Selection rule: among the configurations whose isolated time is within a few
# percent of the shape's best, take the one with the SMALLEST NCOL * BKK
# accumulator. The accumulator lives in registers for the whole K sweep, so its
# size drives the kernel's per-CTA register demand and therefore how much room
# is left for the allreduce blocks to co-reside. Measured end-to-end, a
# low-register choice with the same isolated speed preserved the overlap that
# Marlin's one-CTA-per-SM tile leaves, while the fastest isolated configurations
# (4096-float accumulators) cost more in lost concurrency than they saved.
_EXACT_CONFIGS: Dict[Tuple[int, int], Tuple[int, int, int]] = {
    (1792, 5120): (4, 128, 8),
    (4096, 1280): (4, 64, 4),
    (5120, 1024): (8, 64, 8),
    (576, 5120): (4, 128, 8),
    (5120, 320): (8, 64, 8),
}

# Safe generic tiling for any other divisible shape.
_GENERIC_CONFIG = (4, 64, 4)


@triton.jit
def _dsv41_m1_gemv_kernel(
    w_ptr,  # uint8 (N, K) raw E4M3 bytes, row major
    s_ptr,  # bf16  (N, K // 32) plain model scales, contiguous
    x_ptr,  # bf16  (K,) activation vector
    out_ptr,  # bf16 (N,) result
    K: tl.constexpr,
    KS: tl.constexpr,  # K // 32
    NCOL: tl.constexpr,
    BKK: tl.constexpr,  # multiple of 32; N % NCOL == 0 and K % BKK == 0
):
    """One program owns NCOL output columns and streams all of K.

    The (NCOL, BKK) fp32 accumulator lives in registers for the whole K sweep,
    so the only reduction is the final one over the accumulator tile. Weights
    are read with the default cache policy: nothing in the tile is reused, and
    the tiling below is the one the (NCOL, BKK) timings were measured with.
    """
    pid = tl.program_id(0)
    cols = pid * NCOL + tl.arange(0, NCOL)
    # Exact bf16 constant, as a tensor so the multiply cannot be promoted.
    two120 = tl.full([1], 2.0**120, dtype=tl.bfloat16)
    acc = tl.zeros([NCOL, BKK], dtype=tl.float32)
    for kb in range(0, K, BKK):
        k = kb + tl.arange(0, BKK)
        # Streaming weight bytes are never reused, so keep them from displacing
        # the small working sets of the latency-sensitive kernels that follow
        # this one in the decode step (the allreduce push and the MoE router).
        raw = tl.load(
            w_ptr + cols[:, None] * K + k[None, :], eviction_policy="evict_first"
        )
        # Bit-exact E4M3 -> BF16 payload: fp8_value * 2**-120, all finite codes.
        b = raw.to(tl.int32)
        bits = ((b << 8) & 0x8000) | ((b << 4) & 0x07F0)
        wv = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
        # One scale per 32-wide group, broadcast over the reshaped weight tile.
        g = kb // 32 + tl.arange(0, BKK // 32)
        sc = tl.load(s_ptr + cols[:, None] * KS + g[None, :])
        w3 = tl.reshape(wv, (NCOL, BKK // 32, 32))
        # The single contract rounding: bf16 product of the two bf16 operands.
        wk = (w3 * (sc * two120)[:, :, None]).to(tl.bfloat16)
        xv = tl.load(x_ptr + k).to(tl.float32)
        acc += tl.reshape(wk, (NCOL, BKK)).to(tl.float32) * xv[None, :]
    tl.store(out_ptr + cols, tl.sum(acc, axis=1).to(tl.bfloat16))


def _select_config(n: int, k: int) -> Optional[Tuple[int, int, int]]:
    """Measured config for this exact shape, else the generic one, else None."""
    if n <= 0 or k <= 0 or k % GROUP != 0:
        return None
    exact = _EXACT_CONFIGS.get((n, k))
    if exact is not None:
        ncol, bkk, _ = exact
        if n % ncol == 0 and k % bkk == 0:
            return exact
    ncol, bkk, _ = _GENERIC_CONFIG
    if n % ncol == 0 and k % bkk == 0:
        return _GENERIC_CONFIG
    return None


def dsv41_m1_gemv_config(n: int, k: int) -> Optional[Tuple[int, int, int]]:
    """The (NCOL, BKK, num_warps) this kernel would use, or None if it declines."""
    return _select_config(int(n), int(k))


def dsv41_m1_gemv_eligible(n: int, k: int) -> bool:
    """Whether an (N, K) single-token group-32 GEMV has a legal tiling.

    Cheap and side-effect free: the caller uses this as the last gate before
    falling back to the Marlin kernel.
    """
    return _select_config(int(n), int(k)) is not None


def dsv41_m1_gemv(
    w: torch.Tensor,
    s: torch.Tensor,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out[n] = bf16(sum_k bf16(bf16(fp8(w[n, k])) * s[n, k // 32]) * x[k]).

    w: uint8 (N, K) contiguous raw E4M3 bytes (float8_e4m3fn is accepted and
       reinterpreted); s: bf16 (N, K // 32) contiguous plain scales; x: bf16
       with K elements. Returns a new bf16 (N,) tensor unless ``out`` is given.
    """
    if w.dim() != 2:
        raise ValueError(f"dsv41_m1_gemv expects a 2-D weight, got {tuple(w.shape)}")
    n, k = int(w.shape[0]), int(w.shape[1])
    config = _select_config(n, k)
    if config is None:
        raise ValueError(
            f"dsv41_m1_gemv has no tiling for (N={n}, K={k}); "
            "use dsv41_m1_gemv_eligible() and fall back to Marlin"
        )
    ncol, bkk, warps = config

    if w.dtype == torch.float8_e4m3fn:
        w = w.view(torch.uint8)
    if w.dtype != torch.uint8:
        raise ValueError(f"dsv41_m1_gemv expects uint8 weights, got {w.dtype}")
    if not w.is_contiguous():
        raise ValueError("dsv41_m1_gemv expects a contiguous row-major weight")
    if s.dtype != torch.bfloat16:
        raise ValueError(f"dsv41_m1_gemv expects bf16 scales, got {s.dtype}")
    if not s.is_contiguous():
        raise ValueError("dsv41_m1_gemv expects contiguous scales")
    if tuple(s.shape) != (n, k // GROUP):
        raise ValueError(
            f"dsv41_m1_gemv expects scales of shape {(n, k // GROUP)}, "
            f"got {tuple(s.shape)}"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError(f"dsv41_m1_gemv expects a bf16 activation, got {x.dtype}")
    if x.numel() != k:
        raise ValueError(f"dsv41_m1_gemv expects {k} activations, got {x.numel()}")
    if not (w.device == s.device == x.device):
        raise ValueError(
            "dsv41_m1_gemv expects weight, scales and activation on one device"
        )
    x = x.reshape(-1)
    if not x.is_contiguous():
        x = x.contiguous()

    if out is None:
        out = torch.empty(n, dtype=torch.bfloat16, device=w.device)
    elif (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (n,)
        or out.device != w.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"dsv41_m1_gemv expects a contiguous bf16 out of shape {(n,)}, "
            f"got {out.dtype} {tuple(out.shape)}"
        )

    _dsv41_m1_gemv_kernel[(n // ncol,)](
        w,
        s,
        x,
        out,
        K=k,
        KS=k // GROUP,
        NCOL=ncol,
        BKK=bkk,
        num_warps=warps,
    )
    return out
