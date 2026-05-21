"""
Optimized pure-PyTorch implementation of FusedKNormRopeFlashMLAKernel::forward.

Key optimizations over the naive version:
  1. No Python loop over batch tokens — all ops are fully vectorized.
  2. No .cpu() / .clone() round-trips — stays on the input device (GPU).
  3. UE8M0 encode/decode via integer bit manipulation on tensors (no struct.unpack).
  4. Per-warp FP8 quant via a single (B, 7, 64) reshape — no Python warp loop.
  5. Cache scatter via flat index_put_ — one fused write per region.

Fixed FlashMLA cache layout (head_dim=512, rope_dim=64, nope_dim=448):
  value slot  @ page_ptr + offset * 576  (576 bytes):
    [  0.. 447]  FP8 E4M3  nope  (7 warps × 64 elements, 1 byte each)
    [448.. 575]  BF16      rope  (64 elements × 2 bytes)
  scale slot  @ page_ptr + page_size * 576 + offset * 8  (8 bytes):
    [0..6]       UE8M0 exponent byte per nope warp
    [7]          padding
"""

from __future__ import annotations

import math

import torch

_FP8_E4M3_MAX: float = 448.0

# Fixed dimensions required by the FlashMLA layout
_HEAD_DIM = 512
_ROPE_DIM = 64
_NOPE_DIM = _HEAD_DIM - _ROPE_DIM  # 448
_NOPE_WARPS = 7  # warps 0-6  (64 elems each)
_EPW = _NOPE_DIM // _NOPE_WARPS  # elements per nope warp = 64
_VALUE_BYTES = 576  # bytes per value slot


# ---------------------------------------------------------------------------
# Vectorized UE8M0 helpers  (no Python loops, no struct.unpack)
# ---------------------------------------------------------------------------


def _cast_to_ue8m0_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized UE8M0 encode: extract the biased IEEE-754 exponent byte.
    Equivalent to (float32_bits(max(x, 1e-38)) >> 23) & 0xFF.

    Input:  any shape, positive float32.
    Output: same shape, int32 (values 0-255).
    """
    bits = x.float().clamp(min=1e-38).view(torch.int32)
    return (bits >> 23) & 0xFF  # (…) int32, values in [0,255]


def _inv_scale_ue8m0_tensor(ue8m0: torch.Tensor) -> torch.Tensor:
    """
    Vectorized UE8M0 decode → inv_scale = 2^(127 − ue8m0).
    Reconstructs a float32 with biased exponent (254 − ue8m0) and zero mantissa.

    Input:  any shape, int32 (UE8M0 exponent values).
    Output: same shape, float32.
    """
    inv_exp = (254 - ue8m0).clamp(min=0)  # (…) int32; 0 when ue8m0 >= 254
    inv_bits = (inv_exp << 23).to(torch.int32)
    inv_f = inv_bits.view(torch.float32)
    # ue8m0 >= 254 → scale ≥ 2^127 (overflow) → inv_scale = 0
    return torch.where(ue8m0 >= 254, torch.zeros_like(inv_f), inv_f)


# ---------------------------------------------------------------------------
# Main entry point  (fully vectorized, GPU-resident)
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
    Fused K-norm + RoPE + FlashMLA paged cache store.
    All operations stay on the same device as `kv`; no host round-trips.
    """
    assert kv.shape[-1] == _HEAD_DIM, f"head_dim must be {_HEAD_DIM}"
    assert freqs_cis.shape[-1] == _ROPE_DIM, f"rope_dim must be {_ROPE_DIM}"
    assert (page_size & (page_size - 1)) == 0, "page_size must be a power of 2"

    B = kv.shape[0]
    if B == 0:
        return

    device = kv.device
    page_bits = int(math.log2(page_size))
    page_bytes = kvcache.shape[1]  # kPageBytes

    # ------------------------------------------------------------------
    # Step 1: Block-wide RMSNorm with kv_weight  (B, 512)
    #   norm_factor = rsqrt(mean(x²) + eps)
    #   out = x * norm_factor * kv_weight
    # ------------------------------------------------------------------
    x = kv.float()  # (B, 512)
    rms = x.pow(2).mean(dim=-1, keepdim=True)  # (B,   1)
    norm_factor = torch.rsqrt(rms + eps)  # (B,   1)
    x = x * norm_factor * kv_weight.float().unsqueeze(0)  # (B, 512)

    # ------------------------------------------------------------------
    # Step 2: RoPE on the last rope_dim=64 elements
    # freqs_cis row: [re0, im0, re1, im1, …]  length 64 → 32 complex pairs
    # ------------------------------------------------------------------
    freq = freqs_cis[positions.long()]  # (B,  64)
    freq_re = freq[:, 0::2]  # (B,  32) cosines
    freq_im = freq[:, 1::2]  # (B,  32) sines

    x_nope = x[:, :_NOPE_DIM]  # (B, 448)
    x_rope = x[:, _NOPE_DIM:]  # (B,  64)
    xr = x_rope[:, 0::2]  # (B,  32)
    xi = x_rope[:, 1::2]  # (B,  32)

    rotated_re = xr * freq_re - xi * freq_im  # (B, 32)
    rotated_im = xr * freq_im + xi * freq_re  # (B, 32)
    rope_out = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)  # (B, 64)

    # ------------------------------------------------------------------
    # Step 3: Per-warp FP8 E4M3 quantisation of the nope region
    #
    # Reshape to (B, 7, 64) so each [b, w, :] slice is one CUDA warp's
    # 64 elements. All warp-level ops become dim=-1 reductions — no loop.
    #
    # Matches CUDA warps 0-6:
    #   abs_max   = warp::reduce_max(fmaxf(fabs(x), fabs(y)))
    #   scale_raw = fmaxf(1e-4, abs_max) / FP8_E4M3_MAX
    # ------------------------------------------------------------------
    x_warps = x_nope.reshape(B, _NOPE_WARPS, _EPW)  # (B, 7, 64)

    abs_max = x_warps.abs().amax(dim=-1)  # (B, 7)
    scale_raw = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX  # (B, 7)

    ue8m0 = _cast_to_ue8m0_tensor(scale_raw)  # (B, 7) int32 [0,255]
    inv_scale = _inv_scale_ue8m0_tensor(ue8m0)  # (B, 7) float32

    fp8_nope = (
        (x_warps * inv_scale.unsqueeze(-1))
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )  # (B, 7, 64)

    # ------------------------------------------------------------------
    # Step 4: Compute flat byte addresses for all scatter writes
    #
    #   page   = out_loc >> page_bits
    #   offset = out_loc & (page_size - 1)
    #
    #   value_base[b] = page[b] * page_bytes + offset[b] * 576
    #   scale_base[b] = page[b] * page_bytes + page_size*576 + offset[b]*8
    # ------------------------------------------------------------------
    pages = out_loc.long() >> page_bits  # (B,)
    offsets = out_loc.long() & (page_size - 1)  # (B,)

    value_base = pages * page_bytes + offsets * _VALUE_BYTES  # (B,)
    scale_base = pages * page_bytes + page_size * _VALUE_BYTES + offsets * 8  # (B,)

    flat_cache = kvcache.view(-1)  # (npages*page_bytes,) uint8

    # ------------------------------------------------------------------
    # Step 5a: Scatter FP8 nope bytes → value[0..447]
    #
    # fp8_nope: (B, 448) reinterpreted as uint8 (1 byte per element).
    # nope_idx: (B, 448) flat byte addresses.
    # ------------------------------------------------------------------
    fp8_bytes = fp8_nope.reshape(B, _NOPE_DIM).view(torch.uint8)  # (B, 448)
    nope_cols = torch.arange(_NOPE_DIM, device=device)  # (448,)
    nope_idx = value_base.unsqueeze(1) + nope_cols.unsqueeze(0)  # (B, 448)
    flat_cache.index_put_((nope_idx.reshape(-1),), fp8_bytes.reshape(-1))

    # ------------------------------------------------------------------
    # Step 5b: Scatter BF16 rope bytes → value[448..575]
    #
    # rope_out: (B, 64) fp32 → bf16 → (B, 128) uint8  (2 bytes per elem).
    # ------------------------------------------------------------------
    rope_bf16 = rope_out.to(torch.bfloat16)  # (B,  64)
    rope_bytes = rope_bf16.view(torch.uint8)  # (B, 128)
    rope_cols = torch.arange(128, device=device)  # (128,)
    rope_idx = (value_base + _NOPE_DIM).unsqueeze(1) + rope_cols.unsqueeze(
        0
    )  # (B, 128)
    flat_cache.index_put_((rope_idx.reshape(-1),), rope_bytes.reshape(-1))

    # ------------------------------------------------------------------
    # Step 5c: Scatter UE8M0 scale bytes → scale[0..6]
    #
    # ue8m0: (B, 7) int32 → uint8.
    # ------------------------------------------------------------------
    ue8m0_bytes = ue8m0.to(torch.uint8)  # (B, 7)
    scale_cols = torch.arange(_NOPE_WARPS, device=device)  # (7,)
    scale_idx = scale_base.unsqueeze(1) + scale_cols.unsqueeze(0)  # (B, 7)
    flat_cache.index_put_((scale_idx.reshape(-1),), ue8m0_bytes.reshape(-1))
