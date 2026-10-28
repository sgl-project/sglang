"""Gluon NVFP4 dequantization for the gfx950 index-K kernels (MSA-v1 cache layout).

The payload is read as u16, two bytes holding four e2m1 codes, so one register feeds one
conversion and the even (low-nibble) and odd elements come out as separate fp16 tiles,
the split `nvfp4_load_even_odd` makes for the Triton kernels. A dot over the head dim
is then the even-dim dot accumulated before the odd-dim one, as those kernels chain it.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .nvfp4 import Nvfp4Cache


def fits_buffer_offsets(cache: Nvfp4Cache) -> bool:
    """Whether 32-bit byte offsets from each buffer's base reach its last element; a
    pool page-row view spans more bytes than it holds."""
    return all(
        buf.element_size()
        * (1 + sum((n - 1) * s for n, s in zip(buf.shape, buf.stride())))
        < 2**31
        for buf in (cache.payload, cache.scales)
    )


def nvfp4_buffer_loadable(cache: Nvfp4Cache) -> bool:
    """Whether the kernels can read ``cache``: u16 payload view, 32-bit buffer offsets."""
    payload, scales = cache.payload, cache.scales
    return (
        payload.stride(-1) == 1
        and scales.stride(-1) == 1
        and all(s % 2 == 0 for s in (*payload.stride()[:-1], payload.storage_offset()))
        and fits_buffer_offsets(cache)
    )


@gluon.jit
def _fp4x4_to_f16(payload_u16, scale, PURE: gl.constexpr):
    """Two payload u16 and their fp16 block scale -> fp16 pairs (e0, e1), (e2, e3) of
    the even and (o0, o1), (o2, o3) of the odd elements of the four bytes.

    The conversion's f32 scale only applies an exponent, so it gets 1.0 and the E4M3
    scale is multiplied after; e2m1 times E4M3 is exact in fp16 (normal range, at most
    6 significant bits), so these are the values of `nvfp4_kv_dequantize`.
    """
    e01, e23, o01, o23, _ = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $1, $5, 1.0\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $4, $5, 1.0 op_sel:[1,0,0]\n\t"
        "v_pk_mul_f16 $1, $1, $6\n\t"
        "v_pk_mul_f16 $4, $4, $6\n\t"
        "v_pack_b32_f16 $0, $1, $4\n\t"
        "v_pack_b32_f16 $2, $1, $4 op_sel:[1,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $1, $5, 1.0 op_sel:[0,1,0]\n\t"
        "v_cvt_scalef32_pk_f16_fp4 $4, $5, 1.0 op_sel:[1,1,0]\n\t"
        "v_pk_mul_f16 $1, $1, $6\n\t"
        "v_pk_mul_f16 $4, $4, $6\n\t"
        "v_pack_b32_f16 $3, $1, $4 op_sel:[1,1,0]\n\t"
        "v_pack_b32_f16 $1, $1, $4",
        "=&v,=&v,=&v,=&v,=&v,v,v",
        [payload_u16, scale],
        dtype=(gl.float16, gl.float16, gl.float16, gl.float16, gl.float16),
        is_pure=PURE,
        pack=2,
    )
    return e01, e23, o01, o23


@gluon.jit
def _interleave_pairs(lo, hi):
    """[G, 4, N] pairs of a 16-element group -> [8 * G, N]: u16 r of group g gives rows
    8g + 4 * (r // 2) + r % 2 (``lo``) and those + 2 (``hi``)."""
    G: gl.constexpr = lo.shape[0]
    N: gl.constexpr = lo.shape[2]
    x = gl.join(gl.reshape(lo, [G, 2, 2, N]), gl.reshape(hi, [G, 2, 2, N]))
    return gl.reshape(gl.permute(x, [0, 1, 4, 2, 3]), [G * 8, N])


@gluon.jit
def _interleave_pairs_w(lo, hi):
    """`_interleave_pairs` with a leading warp dim: [W, G, 4, N] -> [W, 8 * G, N]."""
    W: gl.constexpr = lo.shape[0]
    G: gl.constexpr = lo.shape[1]
    N: gl.constexpr = lo.shape[3]
    x = gl.join(gl.reshape(lo, [W, G, 2, 2, N]), gl.reshape(hi, [W, G, 2, 2, N]))
    return gl.reshape(gl.permute(x, [0, 1, 2, 5, 3, 4]), [W, G * 8, N])


@gluon.jit
def nvfp4_even_odd(payload, scales, GROUP_LAYOUT: gl.constexpr):
    """Payload [D // 4, N] u16 and E4M3 scales [D // 16, N] (dims x tokens) -> the even
    and odd dims, fp16 [D // 2, N] each.

    ``GROUP_LAYOUT`` is the [D // 16, 4, N] view of the payload's layout, with each
    thread's u16 of a group contiguous; ``scales`` is in its slice along dim 1.
    """
    U: gl.constexpr = payload.shape[0]
    N: gl.constexpr = payload.shape[1]
    codes = gl.convert_layout(gl.reshape(payload, [U // 4, 4, N]), GROUP_LAYOUT)
    scale = gl.expand_dims(scales.to(gl.float8e4nv, bitcast=True).to(gl.float16), 1)
    e01, e23, o01, o23 = _fp4x4_to_f16(codes, scale, True)
    return _interleave_pairs(e01, e23), _interleave_pairs(o01, o23)


@gluon.jit
def nvfp4_even_odd_w(payload, scales, GROUP_LAYOUT: gl.constexpr):
    """`nvfp4_even_odd` per warp: [W, D // 4, N] u16 and [W, D // 16, N] scales ->
    fp16 [W, D // 2, N] even and odd dims; ``GROUP_LAYOUT`` is the [W, D // 16, 4, N]
    view."""
    W: gl.constexpr = payload.shape[0]
    U: gl.constexpr = payload.shape[1]
    N: gl.constexpr = payload.shape[2]
    codes = gl.convert_layout(gl.reshape(payload, [W, U // 4, 4, N]), GROUP_LAYOUT)
    scale = gl.expand_dims(scales.to(gl.float8e4nv, bitcast=True).to(gl.float16), 2)
    # Not pure: the side effects keep the next block's prefetch, issued before it in
    # program order, from being scheduled after the dequant, where it hides no latency.
    e01, e23, o01, o23 = _fp4x4_to_f16(codes, scale, False)
    return _interleave_pairs_w(e01, e23), _interleave_pairs_w(o01, o23)
