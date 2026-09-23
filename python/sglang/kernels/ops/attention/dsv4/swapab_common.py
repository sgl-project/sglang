"""Packed V4 page loads shared by the CUDA and HIP small-head kernels."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def load_v4(
    CACHE,
    ids,
    valid,
    PAGE: gl.constexpr,
    STRIDE: gl.constexpr,
    KV_LAYOUT: gl.constexpr,
    DATA_BYTES: gl.constexpr,
    SCALE_BYTES: gl.constexpr,
    TILE: gl.constexpr,
):
    # V4 row: 448 fp8 nope + 64 bf16 rope = DATA_BYTES, plus one ue8m0 scale per
    # TILE values in the page's scale rows.
    d = gl.arange(0, 512, gl.SliceLayout(0, KV_LAYOUT))
    base = (ids // PAGE).to(gl.int64)[:, None] * STRIDE
    slot = (ids % PAGE)[:, None]
    mask = valid[:, None] & (d[None, :] < 448)
    bits = gl.load(CACHE + base + slot * DATA_BYTES + d[None, :], mask, 0)
    fp8 = bits.to(gl.float8e4nv, bitcast=True).to(gl.float32)
    exponent = gl.load(
        CACHE + base + PAGE * DATA_BYTES + slot * SCALE_BYTES + d[None, :] // TILE,
        mask,
        0,
    ).to(gl.int32)
    scale = gl.where(exponent == 0, 0x00400000, exponent << 23).to(
        gl.float32, bitcast=True
    )
    rope_ptr = (CACHE + base + slot * DATA_BYTES + 448 + (d[None, :] - 448) * 2).to(
        gl.pointer_type(gl.bfloat16)
    )
    rope = gl.load(rope_ptr, valid[:, None] & (d[None, :] >= 448), 0)
    return gl.where(d[None, :] < 448, fp8 * scale, rope.to(gl.float32)).to(gl.bfloat16)
