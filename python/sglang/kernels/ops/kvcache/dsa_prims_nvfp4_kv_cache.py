"""KV writer for the 352 B/token DSA cache row that prims_ts reads with
FlashInfer Prims-TS ``kv_cache_format="nvfp4.fp8rope"``.

Bytes 0:256 hold the 512 NoPE features as E2M1, even features in the low
nibble; bytes 256:320 the 64 RoPE features as E4M3; block b of 16 NoPE features
has its E4M3 scale at byte 320 + 8*(b % 4) + b//4. There is no global scale.
The block scale is RN_E4M3(amax / 6), and E2M1 rounds to nearest, ties to even.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

PRIMS_NVFP4_BYTES_PER_TOKEN = 352


@triton.jit
def _finite(x, bound: tl.constexpr):
    return tl.where(x != x, 0.0, tl.minimum(tl.maximum(x, -bound), bound))


@triton.jit
def _fp8(x):
    return _finite(x, 448.0).to(tl.float8e4nv, fp_downcast_rounding="rtne")


@triton.jit
def _encode_e2m1(x, sf):
    # Midpoints of the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}; > or >= picks the
    # even code on ties.
    a = tl.abs(x)
    c = (
        (a > sf * 0.25).to(tl.uint8)
        + (a >= sf * 0.75).to(tl.uint8)
        + (a > sf * 1.25).to(tl.uint8)
        + (a >= sf * 1.75).to(tl.uint8)
        + (a > sf * 2.5).to(tl.uint8)
        + (a >= sf * 3.5).to(tl.uint8)
        + (a > sf * 5.0).to(tl.uint8)
    )
    sign = ((x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)) << 3
    return tl.where(sf > 0, c | sign, 0).to(tl.uint8)


@triton.jit
def _write_kernel(
    N,
    R,
    C,
    L,
    SKIP,
    NS: tl.constexpr,
    NF: tl.constexpr,
    RS: tl.constexpr,
    RF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    dst = tl.load(L + t).to(tl.int64)
    valid = dst != SKIP
    safe = tl.where(valid, dst, 0)
    b = tl.arange(0, 32).to(tl.int64)
    f = tl.arange(0, 16).to(tl.int64)
    x = tl.load(
        N + t * NS + (b[:, None] * 16 + f[None, :]) * NF, mask=valid, other=0
    ).to(tl.float32)
    x = _finite(x, 2688.0)  # 6 x 448, the largest representable value
    sf = _fp8(tl.div_rn(tl.max(tl.abs(x), 1), 6.0))
    c = _encode_e2m1(x, sf.to(tl.float32)[:, None])
    lo, hi = tl.split(tl.reshape(c, (32, 8, 2)))
    p = tl.arange(0, 8).to(tl.int64)
    tl.store(C + safe * 352 + b[:, None] * 8 + p[None, :], lo | (hi << 4), mask=valid)
    tl.store(
        C + safe * 352 + 320 + 8 * (b % 4) + b // 4,
        sf.to(tl.uint8, bitcast=True),
        mask=valid,
    )
    r = tl.arange(0, 64).to(tl.int64)
    rope = tl.load(R + t * RS + r * RF, mask=valid, other=0).to(tl.float32)
    tl.store(
        C + safe * 352 + 256 + r, _fp8(rope).to(tl.uint8, bitcast=True), mask=valid
    )


def set_prims_nvfp4_kv_buffer(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    *,
    reserved_skip_index: int = 0,
) -> None:
    """Quantize K into uint8 ``kv_buffer`` [slots, 1, 352] rows at ``loc``.

    ``cache_k_nope`` is [n, (1,) 512] and ``cache_k_rope`` [n, (1,) 64]. As in
    ``set_mla_kv_buffer_triton``, writes to ``reserved_skip_index`` are skipped.
    """
    nope, rope = cache_k_nope.squeeze(1), cache_k_rope.squeeze(1)
    if not nope.shape[0]:
        return
    _write_kernel[(nope.shape[0],)](
        nope,
        rope,
        kv_buffer,
        loc,
        reserved_skip_index,
        nope.stride(0),
        nope.stride(1),
        rope.stride(0),
        rope.stride(1),
        num_warps=4,
        enable_fp_fusion=False,
    )
