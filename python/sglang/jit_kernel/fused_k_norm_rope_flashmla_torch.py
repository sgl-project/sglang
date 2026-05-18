"""
Pure-PyTorch reference implementation of FusedKNormRopeFlashMLAKernel.

Matches the CUDA kernel in:
  python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh

Algorithm per token (head_dim=512, rope_dim=64, nope_dim=448):
  1.  RMSNorm with kv_weight over the full 512-element vector.
  2.  RoPE on the last 64 elements  → stored as BF16 in the cache.
  3.  Per-warp (64-element) abs-max UE8M0 quantisation of the first 448
      nope elements  → stored as FP8 E4M3 in the cache.
  4.  Pack everything into the FlashMLA paged kvcache byte tensor:

      Per token slot  (576 bytes at  page_ptr + offset*576):
        bytes   0 .. 447   FP8 E4M3 nope  (7 warps × 64 elems, 1 B each)
        bytes 448 .. 575   BF16 rope      (64 elems × 2 B each)

      Per scale slot  (8 bytes at  page_ptr + page_size*576 + offset*8):
        bytes 0 .. 6       UE8M0 scale per nope warp (7 bytes)
        byte  7            padding

UE8M0 format:
  An unsigned 8-bit biased exponent — effectively the exponent byte of a
  float32 — representing a power-of-two scale factor.
    encode: ue8m0 = (float32_bits(x) >> 23) & 0xFF
    decode: inv_scale = float32_from_bits((254 - ue8m0) << 23)
              i.e. 2^(127 − ue8m0)
"""

from __future__ import annotations

import math
import struct

import torch

# Maximum representable magnitude in float8_e4m3fn  (1.1111111 × 2^7)
_FP8_E4M3_MAX: float = 448.0

# Fixed dimensions required by the FlashMLA cache layout
_HEAD_DIM   = 512
_ROPE_DIM   = 64
_NOPE_DIM   = _HEAD_DIM - _ROPE_DIM   # 448
_NOPE_WARPS = 7                        # warps 0-6 own the nope region
_EPW        = _NOPE_DIM // _NOPE_WARPS  # elements per nope warp = 64


# ---------------------------------------------------------------------------
# UE8M0 helpers  (scalar, matching CUDA cast_to_ue8m0 / inv_scale_ue8m0)
# ---------------------------------------------------------------------------

def _cast_to_ue8m0(x: float) -> int:
    """
    Encode a positive float as UE8M0:
    extract the biased IEEE-754 exponent byte  →  floor(log2(x)) + 127.
    """
    bits: int = struct.unpack("I", struct.pack("f", float(max(x, 1e-38))))[0]
    return (bits >> 23) & 0xFF


def _inv_scale_ue8m0(ue8m0: int) -> float:
    """
    Decode UE8M0 → 1/scale:
      scale     = 2^(ue8m0 − 127)
      inv_scale = 2^(127 − ue8m0)
    Reconstructed as a float32 with biased exponent (254 − ue8m0) and zero mantissa.
    """
    inv_exp = 254 - int(ue8m0)
    if inv_exp <= 0:       # ue8m0 >= 254 → scale >= 2^127, inv_scale → 0
        return 0.0
    bits = inv_exp << 23   # sign=0, mantissa=0 → exact power of two
    return struct.unpack("f", struct.pack("I", bits))[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fused_k_norm_rope_flashmla_torch(
    kv: torch.Tensor,        # (B, 512)             input dtype (bf16/fp16/fp32)
    kv_weight: torch.Tensor, # (512,)               same dtype as kv
    freqs_cis: torch.Tensor, # (max_pos, 64) fp32   interleaved [re0,im0,re1,im1,…]
    positions: torch.Tensor, # (B,)                 int32 or int64
    out_loc: torch.Tensor,   # (B,)                 int32 cache-slot indices
    kvcache: torch.Tensor,   # (npages, kPageBytes) uint8
    eps: float,
    page_size: int,          # must be a power of 2  (e.g. 1, 2, 4, …)
) -> None:
    """
    In-place fused K-norm + RoPE + FlashMLA cache store.
    Writes directly into *kvcache*; returns nothing.
    """
    assert kv.shape[-1] == _HEAD_DIM, \
        f"FlashMLA layout requires head_dim={_HEAD_DIM}, got {kv.shape[-1]}"
    assert kv_weight.shape == (_HEAD_DIM,)
    assert freqs_cis.shape[-1] == _ROPE_DIM, \
        f"FlashMLA layout requires rope_dim={_ROPE_DIM}, got {freqs_cis.shape[-1]}"
    assert (page_size & (page_size - 1)) == 0, "page_size must be a power of 2"

    B = kv.shape[0]
    if B == 0:
        return

    page_bits = int(math.log2(page_size))

    # ------------------------------------------------------------------
    # Step 1: RMSNorm with kv_weight
    #   norm_factor = rsqrt(mean(x²) + eps)
    #   out = x * norm_factor * kv_weight
    # Matches the CUDA block-wide sum-of-squares + weight multiply.
    # ------------------------------------------------------------------
    x = kv.float()                                          # (B, 512)
    rms = x.pow(2).mean(dim=-1, keepdim=True)               # (B,   1)
    norm_factor = torch.rsqrt(rms + eps)                    # (B,   1)
    x = x * norm_factor * kv_weight.float().unsqueeze(0)    # (B, 512)

    # ------------------------------------------------------------------
    # Step 2: RoPE on the last rope_dim=64 elements  (warp 7 in CUDA)
    # freqs_cis row: [re0, im0, re1, im1, …]  length 64 → 32 complex pairs
    # ------------------------------------------------------------------
    freq = freqs_cis[positions.long()]       # (B, 64)  fp32
    freq_re = freq[:, 0::2]                 # (B, 32)  cosines
    freq_im = freq[:, 1::2]                 # (B, 32)  sines

    x_nope = x[:, :_NOPE_DIM]              # (B, 448)
    x_rope = x[:, _NOPE_DIM:]              # (B,  64)
    xr = x_rope[:, 0::2]                   # (B,  32)
    xi = x_rope[:, 1::2]                   # (B,  32)

    # (x_re + j·x_im) × (freq_re + j·freq_im)
    rotated_re = xr * freq_re - xi * freq_im   # (B, 32)
    rotated_im = xr * freq_im + xi * freq_re   # (B, 32)
    # Re-interleave → (B, 64)
    rope_out = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    # ------------------------------------------------------------------
    # Step 3: Pack into the FlashMLA paged kvcache
    # Work on CPU to allow byte-level tensor slicing; copy result back.
    # ------------------------------------------------------------------
    nope_cpu  = x_nope.cpu()
    rope_cpu  = rope_out.cpu()
    cache_cpu = kvcache.cpu().clone()    # clone so we can index-assign freely

    for b in range(B):
        loc    = int(out_loc[b].item())
        page   = loc >> page_bits           # which page
        offset = loc & (page_size - 1)      # slot within page

        vbase = offset * 576                # byte offset of value slot in page
        sbase = page_size * 576 + offset * 8  # byte offset of scale slot in page

        # ---- BF16 rope region: value[448..576) ----
        # Matches: reinterpret_cast<bf16x2_t*>(rope_ptr)[lane_id] = result
        rope_bf16  = rope_cpu[b].to(torch.bfloat16).contiguous()
        rope_bytes = rope_bf16.view(torch.uint8)           # 64 × 2B = 128 bytes
        cache_cpu[page, vbase + 448 : vbase + 576] = rope_bytes

        # ---- FP8 nope region + UE8M0 scales: value[0..448) + scale[0..7) ----
        # One iteration = one CUDA nope warp (lanes 0-31 of warps 0-6)
        for w in range(_NOPE_WARPS):
            warp_data = nope_cpu[b, w * _EPW : (w + 1) * _EPW]   # (64,) fp32

            # Per-warp abs-max → UE8M0 scale
            # Matches: abs_max = warp::reduce_max(…); scale_raw = max(1e-4, abs_max) / FP8_E4M3_MAX
            abs_max   = warp_data.abs().max().item()
            scale_raw = max(1e-4, abs_max) / _FP8_E4M3_MAX
            ue8m0     = _cast_to_ue8m0(scale_raw)
            inv_scale = _inv_scale_ue8m0(ue8m0)

            # Quantise → FP8 E4M3  then reinterpret as uint8
            # Matches: result = pack_fp8(x * inv_scale, y * inv_scale)
            fp8   = (warp_data * inv_scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            fp8_bytes = fp8.view(torch.uint8)

            # Write FP8 values
            bstart = vbase + w * _EPW
            cache_cpu[page, bstart : bstart + _EPW] = fp8_bytes

            # Write per-warp UE8M0 scale byte
            # Matches: static_cast<uint8_t*>(scale_ptr)[warp_id] = scale_ue8m0
            cache_cpu[page, sbase + w] = ue8m0

    kvcache.copy_(cache_cpu)
