"""Helion fused RMSNorm and NVFP4 quantization for Muse Spark v1.2."""

from __future__ import annotations

import helion
import helion.language as hl
import torch
from helion.autotuner import BooleanFragment

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.muse_spark.helion_runtime import (
    token_count_specialization_key,
)

# The reduction is intentionally one 256-lane row (one block, eight warps).
# Changing this topology changes accumulation order and breaks byte equivalence
# with the production CUDA kernel.
NORM_FP4_QUANT_CONFIGS: dict[str, helion.Config] = {
    "short": helion.Config(
        block_sizes=[1],
        num_stages=7,
        num_warps=8,
        pid_type="flat",
        predicated_encode=True,
    ),
    "long": helion.Config(
        block_sizes=[1],
        num_stages=1,
        num_warps=8,
        pid_type="flat",
        predicated_encode=False,
    ),
}


def _fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Use the same FP32 fused multiply-add as the CUDA row reduction."""
    return hl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _reduce_sum_256(value: torch.Tensor) -> torch.Tensor:
    """Reduce across eight warps, reserving disjoint sum/max scratch space."""
    return hl.inline_asm_elementwise(
        """
          .shared .align 4 .b8 norm_smem[64];
          .reg .pred lane_zero, lane_valid;
          .reg .b32 thread, lane, warp, address;
          .reg .f32 reduced, other;
          mov.f32 reduced, $1;
          shfl.sync.bfly.b32 other, reduced, 16, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 8, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 4, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 2, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 1, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          mov.u32 thread, %tid.x;
          and.b32 lane, thread, 31;
          shr.u32 warp, thread, 5;
          setp.eq.u32 lane_zero, lane, 0;
          mov.u32 address, norm_smem;
          mad.lo.u32 address, warp, 4, address;
          @lane_zero st.shared.f32 [address], reduced;
          bar.sync 0;
          setp.lt.u32 lane_valid, lane, 8;
          mov.u32 address, norm_smem;
          mad.lo.u32 address, lane, 4, address;
          @lane_valid ld.shared.f32 reduced, [address];
          @!lane_valid mov.f32 reduced, 0f00000000;
          shfl.sync.bfly.b32 other, reduced, 16, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 8, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 4, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 2, 31, 0xffffffff;
          add.f32 reduced, reduced, other;
          shfl.sync.bfly.b32 other, reduced, 1, 31, 0xffffffff;
          add.f32 $0, reduced, other;
        """,
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=False,
        pack=1,
    )


def _reduce_max_256(value: torch.Tensor) -> torch.Tensor:
    """Reduce one FP32 maximum per lane in the second half of CTA scratch."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .pred lane_zero, lane_valid;
          .reg .b32 thread, lane, warp, address;
          .reg .f32 warp_max, warp_input;
          mov.u32 thread, %tid.x;
          and.b32 lane, thread, 31;
          shr.u32 warp, thread, 5;
          redux.sync.max.f32 warp_max, $1, 0xffffffff;
          setp.eq.u32 lane_zero, lane, 0;
          mov.u32 address, norm_smem;
          mad.lo.u32 address, warp, 4, address;
          @lane_zero st.shared.f32 [address + 32], warp_max;
          bar.sync 0;
          setp.lt.u32 lane_valid, lane, 8;
          mov.u32 address, norm_smem;
          mad.lo.u32 address, lane, 4, address;
          @lane_valid ld.shared.f32 warp_input, [address + 32];
          @!lane_valid mov.f32 warp_input, 0f00000000;
          redux.sync.max.f32 $0, warp_input, 0xffffffff;
        }
        """,
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=False,
        pack=1,
    )


def _load_bfloat16_x16_to_f32(
    storage: torch.Tensor,
    blocks: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Vector-load one contiguous 16-value BF16 group per Helion lane."""
    like = blocks.to(torch.float32)
    return hl.inline_triton(
        """
        pointer = {storage} + {blocks} * 16
        tl.inline_asm_elementwise(
            '''
            {{
              .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
              .reg .b16 h8, h9, h10, h11, h12, h13, h14, h15;
              .reg .b32 r0, r1, r2, r3, r4, r5, r6, r7;
              ld.global.v4.b32 {{r0, r1, r2, r3}}, [$16];
              ld.global.v4.b32 {{r4, r5, r6, r7}}, [$16 + 16];
              mov.b32 {{h0, h1}}, r0;
              mov.b32 {{h2, h3}}, r1;
              mov.b32 {{h4, h5}}, r2;
              mov.b32 {{h6, h7}}, r3;
              mov.b32 {{h8, h9}}, r4;
              mov.b32 {{h10, h11}}, r5;
              mov.b32 {{h12, h13}}, r6;
              mov.b32 {{h14, h15}}, r7;
              cvt.f32.bf16 $0, h0;
              cvt.f32.bf16 $1, h1;
              cvt.f32.bf16 $2, h2;
              cvt.f32.bf16 $3, h3;
              cvt.f32.bf16 $4, h4;
              cvt.f32.bf16 $5, h5;
              cvt.f32.bf16 $6, h6;
              cvt.f32.bf16 $7, h7;
              cvt.f32.bf16 $8, h8;
              cvt.f32.bf16 $9, h9;
              cvt.f32.bf16 $10, h10;
              cvt.f32.bf16 $11, h11;
              cvt.f32.bf16 $12, h12;
              cvt.f32.bf16 $13, h13;
              cvt.f32.bf16 $14, h14;
              cvt.f32.bf16 $15, h15;
            }}
            ''',
            '=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l',
            [pointer],
            dtype=(tl.float32,) * 16,
            is_pure=False,
            pack=1,
        )
        """,
        args={"storage": storage, "blocks": blocks},
        output_like=(
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
        ),
    )


def _load_row_bfloat16_x16_to_f32(
    storage: torch.Tensor,
    row: torch.Tensor,
    blocks: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Vector-load one 16-value group from a row-major 4096-wide tensor."""
    like = blocks.to(torch.float32)
    return hl.inline_triton(
        """
        pointer = {storage} + {row} * 4096 + {blocks} * 16
        tl.inline_asm_elementwise(
            '''
            {{
              .reg .b16 h0, h1, h2, h3, h4, h5, h6, h7;
              .reg .b16 h8, h9, h10, h11, h12, h13, h14, h15;
              .reg .b32 r0, r1, r2, r3, r4, r5, r6, r7;
              ld.global.v4.b32 {{r0, r1, r2, r3}}, [$16];
              ld.global.v4.b32 {{r4, r5, r6, r7}}, [$16 + 16];
              mov.b32 {{h0, h1}}, r0;
              mov.b32 {{h2, h3}}, r1;
              mov.b32 {{h4, h5}}, r2;
              mov.b32 {{h6, h7}}, r3;
              mov.b32 {{h8, h9}}, r4;
              mov.b32 {{h10, h11}}, r5;
              mov.b32 {{h12, h13}}, r6;
              mov.b32 {{h14, h15}}, r7;
              cvt.f32.bf16 $0, h0;
              cvt.f32.bf16 $1, h1;
              cvt.f32.bf16 $2, h2;
              cvt.f32.bf16 $3, h3;
              cvt.f32.bf16 $4, h4;
              cvt.f32.bf16 $5, h5;
              cvt.f32.bf16 $6, h6;
              cvt.f32.bf16 $7, h7;
              cvt.f32.bf16 $8, h8;
              cvt.f32.bf16 $9, h9;
              cvt.f32.bf16 $10, h10;
              cvt.f32.bf16 $11, h11;
              cvt.f32.bf16 $12, h12;
              cvt.f32.bf16 $13, h13;
              cvt.f32.bf16 $14, h14;
              cvt.f32.bf16 $15, h15;
            }}
            ''',
            '=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,l',
            [pointer],
            dtype=(tl.float32,) * 16,
            is_pure=False,
            pack=1,
        )
        """,
        args={"storage": storage, "row": row, "blocks": blocks},
        output_like=(
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
            like,
        ),
    )


def _norm_bf16_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMS scale and weight, then round one pair through BF16."""
    return hl.inline_triton(
        """
        tl.inline_asm_elementwise(
            '''
            {{
              .reg .f32 n0, n1;
              .reg .b32 pair;
              .reg .b16 lo, hi;
              mul.f32 n0, $2, $6;
              mul.f32 n0, n0, $4;
              mul.f32 n1, $3, $6;
              mul.f32 n1, n1, $5;
              cvt.rn.bf16x2.f32 pair, n1, n0;
              mov.b32 {{lo, hi}}, pair;
              cvt.f32.bf16 $0, lo;
              cvt.f32.bf16 $1, hi;
            }}
            ''',
            '=f,=f,f,f,f,f,f',
            [{x0}, {x1}, {w0}, {w1}, {rstd}],
            dtype=(tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )
        """,
        args={"x0": x0, "x1": x1, "w0": w0, "w1": w1, "rstd": rstd},
        output_like=(x0, x1),
    )


def _encode_block_scale_predicated(
    token_scale: torch.Tensor,
    block_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the E4M3 byte while guarding the reciprocal on zero rows."""
    return hl.inline_triton(
        """
        tl.inline_asm_elementwise(
            '''
            {{
              .reg .pred nonzero, scale_nonzero;
              .reg .b16 fp8_pair;
              .reg .b32 h2_32;
              .reg .b16 h_lo, h_hi;
              .reg .f32 encode_scale, block_scaled, scale_f;
              .reg .f32 scale_f32, rcp_gs, product, result;
              setp.ne.f32 nonzero, $2, 0f00000000;
              @nonzero rcp.approx.ftz.f32 encode_scale, $2;
              @!nonzero mov.b32 encode_scale, 0x7f7fffff;
              mul.f32 block_scaled, $3, 0f3e2aaaab;
              mul.f32 scale_f, encode_scale, block_scaled;
              cvt.rn.satfinite.e4m3x2.f32 fp8_pair, 0f00000000, scale_f;
              cvt.u32.u16 $0, fp8_pair;
              cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
              mov.b32 {{h_lo, h_hi}}, h2_32;
              cvt.f32.f16 scale_f32, h_lo;
              rcp.approx.ftz.f32 rcp_gs, encode_scale;
              mul.f32 product, scale_f32, rcp_gs;
              rcp.approx.ftz.f32 result, product;
              setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;
              selp.f32 $1, result, 0f00000000, scale_nonzero;
            }}
            ''',
            '=r,=f,f,f',
            [{token_scale}, {block_amax}],
            dtype=(tl.int32, tl.float32),
            is_pure=True,
            pack=1,
        )
        """,
        args={"token_scale": token_scale, "block_amax": block_amax},
        output_like=(block_amax.to(torch.int32), block_amax),
    )


def _encode_block_scale_unpredicated(
    token_scale: torch.Tensor,
    block_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the E4M3 byte with the prefill-friendly straight-line path."""
    return hl.inline_triton(
        """
        tl.inline_asm_elementwise(
            '''
            {{
              .reg .pred token_zero, scale_nonzero;
              .reg .b16 fp8_pair;
              .reg .b32 h2_32;
              .reg .b16 h_lo, h_hi;
              .reg .f32 encode_scale, block_scaled, scale_f;
              .reg .f32 scale_f32, rcp_gs, product, result;
              rcp.approx.ftz.f32 encode_scale, $2;
              setp.eq.f32 token_zero, $2, 0f00000000;
              selp.f32 encode_scale, 0f7f7fffff, encode_scale, token_zero;
              mul.f32 block_scaled, $3, 0f3e2aaaab;
              mul.f32 scale_f, encode_scale, block_scaled;
              cvt.rn.satfinite.e4m3x2.f32 fp8_pair, 0f00000000, scale_f;
              cvt.u32.u16 $0, fp8_pair;
              cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
              mov.b32 {{h_lo, h_hi}}, h2_32;
              cvt.f32.f16 scale_f32, h_lo;
              rcp.approx.ftz.f32 rcp_gs, encode_scale;
              mul.f32 product, scale_f32, rcp_gs;
              rcp.approx.ftz.f32 result, product;
              setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;
              selp.f32 $1, result, 0f00000000, scale_nonzero;
            }}
            ''',
            '=r,=f,f,f',
            [{token_scale}, {block_amax}],
            dtype=(tl.int32, tl.float32),
            is_pure=True,
            pack=1,
        )
        """,
        args={"token_scale": token_scale, "block_amax": block_amax},
        output_like=(block_amax.to(torch.int32), block_amax),
    )


def _maximum_abs_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Use Triton's finite-input maximum without PyTorch NaN propagation."""
    return hl.inline_triton(
        "tl.maximum(tl.abs({a}), tl.abs({b}))",
        args={"a": a, "b": b},
        output_like=a,
    )


def _maximum_abs_next(current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return hl.inline_triton(
        "tl.maximum({current}, tl.abs({value}))",
        args={"current": current, "value": value},
        output_like=current,
    )


def _pack_e2m1x16(
    value_0: torch.Tensor,
    value_1: torch.Tensor,
    value_2: torch.Tensor,
    value_3: torch.Tensor,
    value_4: torch.Tensor,
    value_5: torch.Tensor,
    value_6: torch.Tensor,
    value_7: torch.Tensor,
    value_8: torch.Tensor,
    value_9: torch.Tensor,
    value_10: torch.Tensor,
    value_11: torch.Tensor,
    value_12: torch.Tensor,
    value_13: torch.Tensor,
    value_14: torch.Tensor,
    value_15: torch.Tensor,
) -> torch.Tensor:
    """Pack one 16-value NVFP4 scale group into one 64-bit word."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;
          .reg .b32 lo, hi;
          cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
          cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;
          cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;
          cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;
          cvt.rn.satfinite.e2m1x2.f32 b4, $10, $9;
          cvt.rn.satfinite.e2m1x2.f32 b5, $12, $11;
          cvt.rn.satfinite.e2m1x2.f32 b6, $14, $13;
          cvt.rn.satfinite.e2m1x2.f32 b7, $16, $15;
          mov.b32 lo, {b0, b1, b2, b3};
          mov.b32 hi, {b4, b5, b6, b7};
          mov.b64 $0, {lo, hi};
        }
        """,
        "=l,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        [
            value_0,
            value_1,
            value_2,
            value_3,
            value_4,
            value_5,
            value_6,
            value_7,
            value_8,
            value_9,
            value_10,
            value_11,
            value_12,
            value_13,
            value_14,
            value_15,
        ],
        dtype=torch.int64,
        is_pure=True,
        pack=1,
    )


@helion.kernel(
    backend="triton",
    key=token_count_specialization_key,
    static_shapes=False,
)
def _norm_fp4_quant_helion_kernel(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: hl.constexpr,
    global_scale_inv: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse RMSNorm and NVFP4 quantization in one Helion-generated kernel."""
    num_tokens, hidden_dim = hidden_states.size()
    hidden_dim = hl.specialize(hidden_dim)
    hl.specialize(hidden_states.stride())
    hl.specialize(weight.stride())
    num_scale_blocks = hidden_dim // 16
    predicated_encode = hl.register_tunable("predicated_encode", BooleanFragment())

    fp4_output = torch.empty(
        (num_tokens, hidden_dim // 2),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    fp4_output_i64 = fp4_output.view(torch.int64)
    scale_output = torch.empty(
        (num_tokens, num_scale_blocks),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    per_token_scale = torch.empty(
        num_tokens, dtype=torch.float32, device=hidden_states.device
    )

    for tile_tokens in hl.tile(num_tokens):
        hl.inline_triton("tl.extra.cuda.gdc_wait()", args=(), output_like=None)
        row = tile_tokens.begin
        blocks = hl.arange(num_scale_blocks)
        (
            x0,
            x1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15,
        ) = _load_row_bfloat16_x16_to_f32(hidden_states, row, blocks)

        local_sum = hl.zeros([num_scale_blocks], dtype=torch.float32)
        local_sum = _fma_f32(x0, x0, local_sum)
        local_sum = _fma_f32(x1, x1, local_sum)
        local_sum = _fma_f32(x2, x2, local_sum)
        local_sum = _fma_f32(x3, x3, local_sum)
        local_sum = _fma_f32(x4, x4, local_sum)
        local_sum = _fma_f32(x5, x5, local_sum)
        local_sum = _fma_f32(x6, x6, local_sum)
        local_sum = _fma_f32(x7, x7, local_sum)
        local_sum = _fma_f32(x8, x8, local_sum)
        local_sum = _fma_f32(x9, x9, local_sum)
        local_sum = _fma_f32(x10, x10, local_sum)
        local_sum = _fma_f32(x11, x11, local_sum)
        local_sum = _fma_f32(x12, x12, local_sum)
        local_sum = _fma_f32(x13, x13, local_sum)
        local_sum = _fma_f32(x14, x14, local_sum)
        local_sum = _fma_f32(x15, x15, local_sum)
        sum_sq = _reduce_sum_256(local_sum)
        rstd = torch.rsqrt(sum_sq * (1.0 / hidden_dim) + eps)

        (
            w0,
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
        ) = _load_bfloat16_x16_to_f32(weight, blocks)

        n0, n1 = _norm_bf16_pair(x0, x1, w0, w1, rstd)
        n2, n3 = _norm_bf16_pair(x2, x3, w2, w3, rstd)
        n4, n5 = _norm_bf16_pair(x4, x5, w4, w5, rstd)
        n6, n7 = _norm_bf16_pair(x6, x7, w6, w7, rstd)
        n8, n9 = _norm_bf16_pair(x8, x9, w8, w9, rstd)
        n10, n11 = _norm_bf16_pair(x10, x11, w10, w11, rstd)
        n12, n13 = _norm_bf16_pair(x12, x13, w12, w13, rstd)
        n14, n15 = _norm_bf16_pair(x14, x15, w14, w15, rstd)

        block_amax = _maximum_abs_pair(n0, n1)
        block_amax = _maximum_abs_next(block_amax, n2)
        block_amax = _maximum_abs_next(block_amax, n3)
        block_amax = _maximum_abs_next(block_amax, n4)
        block_amax = _maximum_abs_next(block_amax, n5)
        block_amax = _maximum_abs_next(block_amax, n6)
        block_amax = _maximum_abs_next(block_amax, n7)
        block_amax = _maximum_abs_next(block_amax, n8)
        block_amax = _maximum_abs_next(block_amax, n9)
        block_amax = _maximum_abs_next(block_amax, n10)
        block_amax = _maximum_abs_next(block_amax, n11)
        block_amax = _maximum_abs_next(block_amax, n12)
        block_amax = _maximum_abs_next(block_amax, n13)
        block_amax = _maximum_abs_next(block_amax, n14)
        block_amax = _maximum_abs_next(block_amax, n15)
        row_amax = _reduce_max_256(block_amax)
        token_scale = row_amax * global_scale_inv
        if predicated_encode:
            fp8, output_scale = _encode_block_scale_predicated(token_scale, block_amax)
        else:
            fp8, output_scale = _encode_block_scale_unpredicated(
                token_scale, block_amax
            )

        packed = _pack_e2m1x16(
            n0 * output_scale,
            n1 * output_scale,
            n2 * output_scale,
            n3 * output_scale,
            n4 * output_scale,
            n5 * output_scale,
            n6 * output_scale,
            n7 * output_scale,
            n8 * output_scale,
            n9 * output_scale,
            n10 * output_scale,
            n11 * output_scale,
            n12 * output_scale,
            n13 * output_scale,
            n14 * output_scale,
            n15 * output_scale,
        )
        hl.inline_triton(
            """
            tl.store({fp4_output} + {row} * 256 + {blocks}, {packed})
            tl.store({scale_output} + {row} * 256 + {blocks}, {fp8}.to(tl.uint8))
            tl.store(
                {per_token_scale} + {row} + {blocks} * 0,
                {token_scale},
                mask={blocks} == 0,
            )
            """,
            args={
                "fp4_output": fp4_output_i64,
                "scale_output": scale_output,
                "per_token_scale": per_token_scale,
                "row": row,
                "blocks": blocks,
                "packed": packed,
                "fp8": fp8,
                "token_scale": token_scale,
            },
            output_like=None,
        )
        hl.inline_triton(
            "tl.extra.cuda.gdc_launch_dependents()", args=(), output_like=None
        )

    return fp4_output, scale_output, per_token_scale


def fused_norm_fp4_quant_helion(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    global_scale_inv: float,
    *,
    quantize: bool = True,
    want_normed: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run Helion at the fused production norm/quant boundary."""
    assert x.shape[0] > 0
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
    assert x.shape[1] == 4096
    assert quantize and not want_normed
    assert weight is not None
    assert weight.dtype == torch.bfloat16 and weight.shape == (x.shape[1],)

    weight = weight.contiguous()
    from sglang.kernels.ops.muse_spark.helion_runtime import run_pinned_helion

    fp4, sf, per_token_scale = run_pinned_helion(
        _norm_fp4_quant_helion_kernel,
        NORM_FP4_QUANT_CONFIGS,
        (x, weight, float(eps), float(global_scale_inv)),
        num_tokens=x.shape[0],
        launch_pdl=is_arch_support_pdl(),
    )
    return fp4, sf, per_token_scale, None


__all__ = [
    "NORM_FP4_QUANT_CONFIGS",
    "fused_norm_fp4_quant_helion",
    "_norm_fp4_quant_helion_kernel",
]
