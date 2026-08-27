"""Pure-torch oracle helpers for residue NVFP4 kernel tests.

Shared by the quantization parity tests and the fold GEMM tests: FP4 e2m1
nibble decoding, the swizzled 128x4 scale-factor layout (both directions),
mext_r1 layout row mapping, and a simple NVFP4 weight quantizer.
"""

from __future__ import annotations

import torch

E2M1_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)

# mext_r1 layout modes (must match the kernel's enum).
LAYOUT_CONCAT = 0
LAYOUT_ROW_PAIR = 1
LAYOUT_CONCAT_K = 3


def decode_fp4(packed: torch.Tensor) -> torch.Tensor:
    """[R, K/2] packed uint8 -> [R, K] float32 (low nibble = even element)."""
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return E2M1_LUT.to(packed.device)[codes]


def sf_unswizzle(sf_bytes: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Swizzled 128x4-atom SF bytes -> row-major (rows, cols // 16) float32."""
    device = sf_bytes.device
    m_i = torch.arange(rows, device=device).view(-1, 1)
    ks = torch.arange(cols // 16, device=device).view(1, -1)
    n_kt = (cols + 63) // 64
    idx = (
        (m_i // 128) * (n_kt * 512)
        + (ks // 4) * 512
        + (m_i % 32) * 16
        + ((m_i % 128) // 32) * 4
        + (ks % 4)
    ).reshape(-1)
    picked = sf_bytes.reshape(-1)[idx].reshape(rows, cols // 16)
    return picked.view(torch.float8_e4m3fn).float()


def swizzle_scale(scale: torch.Tensor) -> torch.Tensor:
    """Row-major [R, K/16] fp8 scale -> swizzled 128x4 tiled layout bytes."""
    r, k_groups = scale.shape
    r_pad = (r + 127) // 128 * 128
    k_pad = (k_groups + 3) // 4 * 4
    padded = torch.zeros(r_pad, k_pad, dtype=scale.dtype, device=scale.device)
    padded[:r, :k_groups] = scale
    out = (
        padded.reshape(r_pad // 128, 4, 32, k_pad // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
    )
    return out.reshape(r_pad, k_pad)


def base_row(r: torch.Tensor, out_m: int, mode: int) -> torch.Tensor:
    """mext_r1: output row index of token r's BASE row for a layout mode."""
    if mode == LAYOUT_ROW_PAIR:
        return 2 * r
    return r


def residue_row(r: torch.Tensor, out_m: int, mode: int) -> torch.Tensor:
    """mext_r1: output row index of token r's RESIDUE row for a layout mode."""
    if mode == LAYOUT_ROW_PAIR:
        return 2 * r + 1
    return r + out_m


def quantize_nvfp4_weight(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple per-16-group NVFP4 weight quantization (torch oracle).

    Returns (packed uint8 [N, K/2], swizzled fp8-e4m3 scale bytes,
    w_global_scale). Round-to-nearest against the e2m1 grid; tie behavior is
    bucketize's, which is fine for decomposition-tolerance tests.
    """
    n, k = w.shape
    wf = w.float().reshape(n, k // 16, 16)
    w_global = (448.0 * 6.0) / wf.abs().max()
    sf = (wf.abs().amax(dim=-1) / 6.0) * w_global  # [N, K/16]
    sf_fp8 = sf.to(torch.float8_e4m3fn)
    sf_dec = sf_fp8.float()
    scale = torch.where(sf_dec > 0, w_global / sf_dec, torch.zeros_like(sf_dec))
    scaled = wf * scale.unsqueeze(-1)

    lut = E2M1_LUT[:8].to(w.device)
    mags = scaled.abs().clamp(max=6.0)
    idx = torch.bucketize(mags, (lut[1:] + lut[:-1]) / 2)
    codes = torch.where(scaled < 0, idx + 8, idx).to(torch.uint8).reshape(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, swizzle_scale(sf_fp8), w_global


def dequant_nvfp4_weight(
    packed: torch.Tensor, sf_swizzled: torch.Tensor, w_global: torch.Tensor
) -> torch.Tensor:
    """Invert quantize_nvfp4_weight back to float32 [N, K]."""
    n, k_half = packed.shape
    k = k_half * 2
    vals = decode_fp4(packed)
    sf = sf_unswizzle(sf_swizzled, (n + 127) // 128 * 128, k)[:n]
    return vals * sf.repeat_interleave(16, dim=1) / w_global
