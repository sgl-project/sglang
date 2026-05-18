"""
Pure-PyTorch reference implementation of FusedQIndexerRopeHadamardQuantKernel::forward.

Matches the CUDA kernel in:
  python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh

Algorithm per (token, head) work item — fixed head_dim=128, rope_dim=64:
  1. Load input as fp32  (no RMSNorm — this is the indexer path).
  2. RoPE on the last 64 elements (2 complex pairs per lane in CUDA).
  3. 128-point Walsh-Hadamard Transform (WHT):
       - 2 local butterfly stages  (bits 0 and 1 within each 4-elem pack)
       - 5 cross-lane shfl_xor butterfly stages  (masks 1,2,4,8,16)
       Together these form the standard iterative WHT on 128 elements.
  4. Per-(token,head) FP8 E4M3 activation quantisation:
       scale      = max(1e-4, abs_max) / FP8_E4M3_MAX
       q_fp8      = clamp(x / scale)  cast to fp8_e4m3fn
  5. weights_out = weight * weight_scale * scale  (fused scale absorption).
"""

from __future__ import annotations

import math

import torch

# Maximum representable value in float8_e4m3fn
_FP8_E4M3_MAX: float = 448.0

# Fixed dimensions (enforced by static_assert in the CUDA kernel)
_HEAD_DIM: int = 128
_ROPE_DIM: int = 64
_NOPE_DIM: int = _HEAD_DIM - _ROPE_DIM  # 64


# ---------------------------------------------------------------------------
# 128-point Walsh-Hadamard Transform (iterative, in-place on last dim)
# ---------------------------------------------------------------------------

def _fwht128(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalised 128-point Walsh-Hadamard Transform on the last dimension.

    Implements the same butterfly decomposition as the CUDA kernel:
      - h=1  : local stage 1  (butterfly on adjacent pairs  → bit 0)
      - h=2  : local stage 2  (butterfly on stride-2 pairs  → bit 1)
      - h=4  : cross-lane mask 1  (→ bit 2)
      - h=8  : cross-lane mask 2  (→ bit 3)
      - h=16 : cross-lane mask 4  (→ bit 4)
      - h=32 : cross-lane mask 8  (→ bit 5)
      - h=64 : cross-lane mask 16 (→ bit 6)

    At each step the reshape groups the N elements into (N/2h, 2, h) blocks.
    The butterfly is: (u, v) → (u+v, u-v) across the middle dimension.
    This is numerically identical to the CUDA __shfl_xor_sync butterflies.
    """
    N = x.shape[-1]
    assert N == 128, f"Expected last dim 128, got {N}"
    leading = x.shape[:-1]

    h = 1
    while h < N:
        # Reshape: (..., N/2h, 2, h)
        x = x.reshape(*leading, N // (2 * h), 2, h)
        u = x[..., 0, :]   # (..., N/2h, h)
        v = x[..., 1, :]   # (..., N/2h, h)
        x = torch.stack([u + v, u - v], dim=-2)   # (..., N/2h, 2, h)
        x = x.reshape(*leading, N)
        h *= 2

    return x


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fused_q_indexer_rope_hadamard_quant_torch(
    q_input: torch.Tensor,      # (B, H, 128)   input dtype (bf16/fp16/fp32)
    q_fp8: torch.Tensor,        # (B, H, 128)   fp8_e4m3fn, pre-allocated
    weight: torch.Tensor,       # (B, H)        same dtype as q_input
    weights_out: torch.Tensor,  # (B, H, 1)     fp32, pre-allocated
    weight_scale: float,        # scalar  c4_indexer.weight_scale
    freqs_cis: torch.Tensor,    # (max_pos, 64) fp32, interleaved [re0,im0,re1,im1,…]
    positions: torch.Tensor,    # (B,)          int32 or int64
) -> None:
    """
    In-place fused RoPE + 128-pt WHT + FP8 quantisation for the C4 indexer Q path.
    Writes into q_fp8 and weights_out; returns nothing.

    Call-site pre-processing (matches deepseek_v4.py):
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # (max_pos, 64)
        fused_q_indexer_rope_hadamard_quant_torch(..., freqs_real, positions)
    """
    B, H, head_dim = q_input.shape
    assert head_dim == _HEAD_DIM, f"head_dim must be {_HEAD_DIM}, got {head_dim}"
    assert freqs_cis.shape[-1] == _ROPE_DIM, \
        f"freqs_cis last dim must be {_ROPE_DIM}, got {freqs_cis.shape[-1]}"
    assert q_fp8.shape == (B, H, _HEAD_DIM)
    assert weight.shape == (B, H)
    assert weights_out.shape == (B, H, 1)

    if B == 0:
        return

    # ------------------------------------------------------------------
    # Part 1: Load — promote to fp32, no normalisation.
    # Matches: input_vec.load(input_ptr, lane_id); data[i] = cast<float>(…)
    # ------------------------------------------------------------------
    x = q_input.float()                              # (B, H, 128)

    # ------------------------------------------------------------------
    # Part 2: RoPE on the last rope_dim=64 elements.
    #
    # CUDA: the last kRopeSize=16 lanes each hold 4 elements covering
    # positions [64..128).  Each lane holds 2 complex pairs.
    # freqs layout: [re0, im0, re1, im1, …]  along the last axis.
    #
    # In PyTorch we gather the per-token freq row and apply the rotation
    # vectorised over all heads simultaneously.
    # ------------------------------------------------------------------
    freq = freqs_cis[positions.long()]               # (B,    64)  fp32
    freq = freq.unsqueeze(1)                         # (B,  1, 64) broadcast over H

    freq_re = freq[..., 0::2]                        # (B, 1, 32)  cosines
    freq_im = freq[..., 1::2]                        # (B, 1, 32)  sines

    x_nope = x[..., :_NOPE_DIM]                     # (B, H, 64)  pass-through
    x_rope = x[..., _NOPE_DIM:]                     # (B, H, 64)

    xr = x_rope[..., 0::2]                          # (B, H, 32)  real parts
    xi = x_rope[..., 1::2]                          # (B, H, 32)  imag parts

    # (x_re + j·x_im) × (freq_re + j·freq_im)
    # Matches CUDA:
    #   data[0] = x_real * fxr - x_imag * fxi
    #   data[1] = x_real * fxi + x_imag * fxr   (two pairs per rope lane)
    rotated_re = xr * freq_re - xi * freq_im         # (B, H, 32)
    rotated_im = xr * freq_im + xi * freq_re         # (B, H, 32)

    # Re-interleave [re0, im0, re1, im1, …] → (B, H, 64)
    rotated = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    x = torch.cat([x_nope, rotated], dim=-1)         # (B, H, 128)

    # ------------------------------------------------------------------
    # Part 3: 128-point Walsh-Hadamard Transform + normalise.
    #
    # The CUDA kernel performs:
    #   • 2 local butterfly stages on each 4-elem pack  (bits 0, 1)
    #   • 5 cross-lane shfl_xor stages  (masks 1,2,4,8,16 = bits 2-6)
    # which together constitute the standard iterative WHT on 128 elements.
    # _fwht128 implements exactly the same butterfly sequence.
    #
    # Matches CUDA: kHadamardScale = rsqrt(128.0f)
    # ------------------------------------------------------------------
    x = _fwht128(x)                                  # (B, H, 128)  unnormalised
    x = x * (1.0 / math.sqrt(_HEAD_DIM))             # normalise

    # ------------------------------------------------------------------
    # Part 4: Per-(token, head) FP8 E4M3 quantisation.
    #
    # CUDA (warp-level):
    #   abs_max   = warp::reduce_max(local_max)       ← over all 128 elems
    #   scale     = max(1e-4, abs_max) / FP8_E4M3_MAX
    #   inv_scale = 1 / scale
    #   q_fp8[…]  = pack_fp8(data[i] * inv_scale, …)
    # ------------------------------------------------------------------
    abs_max = x.abs().amax(dim=-1, keepdim=True)     # (B, H, 1)
    scale   = abs_max.clamp(min=1e-4) / _FP8_E4M3_MAX  # (B, H, 1)
    inv_scale = 1.0 / scale                          # (B, H, 1)

    x_scaled = (x * inv_scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    q_fp8.copy_(x_scaled.to(torch.float8_e4m3fn))

    # ------------------------------------------------------------------
    # Part 5: Fused weight absorption.
    #
    # CUDA: params.weights_out[work_id] = weight_val * weight_scale * scale
    # where work_id = b*H + h, so weights_out has shape (B*H,) ≡ (B, H, 1).
    # ------------------------------------------------------------------
    w = weight.float()                               # (B, H)
    # scale is (B, H, 1); squeeze to (B, H) for element-wise multiply
    weights_out.copy_((w * float(weight_scale) * scale.squeeze(-1)).unsqueeze(-1))
