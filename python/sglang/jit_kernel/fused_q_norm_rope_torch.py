"""
Pure-PyTorch reference implementation of FusedQNormRopeKernel.

Matches the CUDA kernel in:
  python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh

Algorithm (per token, per head):
  1. RMSNorm-self  – normalize the full head_dim vector (no learned weight).
  2. NoPE region   – first (head_dim - rope_dim) elements written as-is.
  3. RoPE region   – last rope_dim elements rotated with freqs_cis.

freqs_cis contract (from the call-site):
  torch.view_as_real(freqs_cis).flatten(-2)  →  (max_pos, rope_dim) fp32
  Layout is interleaved [re0, im0, re1, im1, ...] along the last axis,
  so rope_dim pairs map to rope_dim/2 complex rotations.
"""

from __future__ import annotations

import torch


def fused_q_norm_rope_torch(
    q_input: torch.Tensor,   # (B, num_q_heads, head_dim)  any float dtype
    q_output: torch.Tensor,  # (B, num_q_heads, head_dim)  same dtype, pre-allocated
    eps: float,
    freqs_cis: torch.Tensor, # (max_pos, rope_dim) fp32, interleaved re/im
                             # — pass the *already-flattened* tensor, i.e.
                             #   torch.view_as_real(freqs_cis).flatten(-2)
    positions: torch.Tensor, # (B,) int32 or int64
) -> None:
    """
    In-place fused warp-per-(token, head) RMSNorm-self + RoPE.
    Writes result into q_output (same shape as q_input).
    """
    # ------------------------------------------------------------------ #
    # shapes / constants
    # ------------------------------------------------------------------ #
    B, H, head_dim = q_input.shape
    rope_dim = freqs_cis.shape[-1]           # interleaved: rope_dim = 2 * num_pairs
    nope_dim = head_dim - rope_dim

    assert rope_dim % 2 == 0, "rope_dim must be even (interleaved re/im)"
    assert freqs_cis.shape[-1] == rope_dim

    # ------------------------------------------------------------------ #
    # part 1: RMSNorm-self  (no learned weight)
    #   norm_factor = rsqrt(mean(x^2) + eps)
    # ------------------------------------------------------------------ #
    x = q_input.float()                       # (B, H, head_dim)

    # mean of squares over the head dimension
    rms = x.pow(2).mean(dim=-1, keepdim=True)  # (B, H, 1)
    norm_factor = torch.rsqrt(rms + eps)       # (B, H, 1)
    x = x * norm_factor                        # (B, H, head_dim)  – normalised

    # ------------------------------------------------------------------ #
    # part 2: RoPE on the last rope_dim elements
    #   freqs row for each token: shape (B, rope_dim)  [re0,im0,re1,im1,...]
    # ------------------------------------------------------------------ #
    # gather the per-token frequency rows
    freq_rows = freqs_cis[positions.long()]    # (B, rope_dim)
    # broadcast over heads: (B, 1, rope_dim)
    freq_rows = freq_rows.unsqueeze(1)

    # split interleaved [re, im, re, im, ...] into separate tensors
    # shape of each: (B, 1, rope_dim/2)
    freq_re = freq_rows[..., 0::2]             # cosines
    freq_im = freq_rows[..., 1::2]             # sines

    # rope tail from the normalised vector
    x_rope = x[..., nope_dim:]                # (B, H, rope_dim)
    x_re   = x_rope[..., 0::2]               # (B, H, rope_dim/2)
    x_im   = x_rope[..., 1::2]               # (B, H, rope_dim/2)

    # complex multiply: (x_re + i*x_im) * (freq_re + i*freq_im)
    rotated_re = x_re * freq_re - x_im * freq_im   # (B, H, rope_dim/2)
    rotated_im = x_re * freq_im + x_im * freq_re   # (B, H, rope_dim/2)

    # re-interleave back to (B, H, rope_dim)
    rotated = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    # ------------------------------------------------------------------ #
    # part 3: assemble and write to q_output
    # ------------------------------------------------------------------ #
    out = torch.cat([x[..., :nope_dim], rotated], dim=-1)  # (B, H, head_dim)
    q_output.copy_(out.to(q_input.dtype))
