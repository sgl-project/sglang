# SPDX-License-Identifier: Apache-2.0
"""SGLang's tensors in the layouts the DeepSeek-V4.1 launches read."""

import torch


def e8m0(scale: torch.Tensor, tag: str) -> torch.Tensor:
    """f32 32 x 32 block scales as E8M0 codes: a power of two's exponent bits."""
    if scale.dtype != torch.float32:
        raise RuntimeError(f"{tag}: needs f32 block scales, got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32)
    if not bool(((bits & 0x807FFFFF) == 0).all()):
        raise RuntimeError(f"{tag}: needs power-of-two (ue8m0) block scales")
    return (bits >> 23).to(torch.uint8)


def mxfp8_exact(w: torch.Tensor, tag: str):
    """A bf16 weight [N, K] as e4m3 with E8M0 32 x 32 block codes, bit for bit: a
    checkpoint FP8 weight that SGLang dequantized with power-of-two block scales."""
    n, k = w.shape
    blocks = w.float().view(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3))
    # the smallest power of two that keeps each block within e4m3's 448
    e = torch.ceil(torch.log2(amax.clamp_min(2.0**-126) / 448.0)).clamp(-127, 127)
    scale = torch.exp2(e)[:, None, :, None]
    q = (blocks / scale).to(torch.float8_e4m3fn)
    if not torch.equal(
        (q.float() * scale).to(w.dtype), w.view(n // 32, 32, k // 32, 32)
    ):
        raise RuntimeError(f"{tag}: the bf16 weight is not MXFP8 at 32 x 32 blocks")
    return q.view(n, k).contiguous(), (e + 127).to(torch.uint8).contiguous()


def rope_table(freqs_cis: torch.Tensor, tables: list) -> torch.Tensor:
    """K1's and K2's RoPE table [positions, 64] f32 (32 cos, then 32 sin); equal
    tables share one copy (``tables``: the (freqs_cis, table) pairs so far)."""
    for f, t in tables:
        if f.shape == freqs_cis.shape and f.device == freqs_cis.device:
            if torch.equal(f, freqs_cis):
                return t
    t = torch.cat([freqs_cis.real, freqs_cis.imag], -1).float().contiguous()
    tables.append((freqs_cis, t))
    return t
