"""NVFP4 PTX helpers for the Helion verify megakernels."""

from __future__ import annotations

import helion.language as hl
import torch


def encode_block_scale(
    token_scale: torch.Tensor,
    block_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One E4M3 block scale byte and its quant multiplier; a zero token scale
    yields FlashInfer's zero-row result."""

    return hl.inline_asm_elementwise(
        """
        {
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
          @!nonzero mov.b16 fp8_pair, 0x0000;
          cvt.u32.u16 $0, fp8_pair;
          cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
          mov.b32 {h_lo, h_hi}, h2_32;
          cvt.f32.f16 scale_f32, h_lo;
          rcp.approx.ftz.f32 rcp_gs, encode_scale;
          mul.f32 product, scale_f32, rcp_gs;
          rcp.approx.ftz.f32 result, product;
          setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;
          selp.f32 $1, result, 0f00000000, scale_nonzero;
        }
        """,
        "=r,=f,f,f",
        [token_scale, block_amax],
        dtype=(torch.int32, torch.float32),
        is_pure=True,
        pack=1,
    )


def pack_e2m1x2(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """One e2m1x2 byte (as int32) per (low, high) FP32 pair, RN saturating."""

    return hl.inline_asm_elementwise(
        """
        {
          .reg .b8 b;
          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.u32.u8 $0, b;
        }
        """,
        "=r,f,f",
        [low, high],
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    )


__all__ = ["encode_block_scale", "pack_e2m1x2"]
