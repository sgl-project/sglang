"""Packed V4 page loads shared by the CUDA and HIP small-head kernels."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _load_v4(
    CACHE, ids, valid, PAGE: gl.constexpr, STRIDE: gl.constexpr, KV_LAYOUT: gl.constexpr
):
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * 576 + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * 576 + slot * 8 + d[None, :] // 64, mask, 0
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * 576 + 448 + (d[None, :] - 448) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)
