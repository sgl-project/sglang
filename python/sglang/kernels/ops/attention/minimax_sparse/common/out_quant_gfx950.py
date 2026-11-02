"""Gluon epilogue of the gfx950 sparse attention kernels: O as the fp8 o_proj GEMM input.

The bytes and scales are those of aiter's separate quant.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_FP8_E4M3_MAX = gl.constexpr(448.0)


@gluon.jit
def _rcp_f32(x):
    # aiter's per-token kernel multiplies by the approximate v_rcp_f32, not 1 / scale.
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def store_fp8_out(x, q8_ptrs, scale_ptr, scale_offs, OUT_QUANT: gl.constexpr):
    """Quantize the bf16-rounded O [H, D] (fp32 ``x``) like aiter's separate o_proj quant.

    ``OUT_QUANT`` 1 is per token over the row and 2 per 1 x 128 head, the arithmetic of
    ``fp8_per_token_quant`` / ``fp8_group_quant``.
    """
    if OUT_QUANT == 1:
        scale = gl.max(gl.max(gl.abs(x), axis=1), axis=0) * (1.0 / _FP8_E4M3_MAX)
        q = gl.clamp(x * _rcp_f32(scale), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    else:
        scale = gl.maximum(gl.max(gl.abs(x), axis=1), 1e-10) * (1.0 / _FP8_E4M3_MAX)
        inv = gl.div_rn(gl.full(scale.shape, 1.0, gl.float32, scale.type.layout), scale)
        q = gl.clamp(x * inv[:, None], -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    gl.store(q8_ptrs, q.to(gl.float8e4nv))
    gl.store(scale_ptr + scale_offs, scale)
