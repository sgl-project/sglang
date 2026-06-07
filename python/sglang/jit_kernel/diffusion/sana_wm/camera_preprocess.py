# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.diffusion.sana_wm.qkv_preprocess import (
    prepare_sana_wm_rope_tables,
)


@triton.jit
def _sana_wm_cam_prep_kernel(
    q_raw_ptr,
    k_raw_ptr,
    v_raw_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    q_weight_ptr,
    k_weight_ptr,
    proj_q_ptr,
    proj_kv_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    inflation_sq_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_D_ROPE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx - b_idx * N

    row_base = b_idx * (N * H * D) + n_idx * (H * D) + h_idx * D
    weight_base = h_idx * D
    proj_base = (b_idx * N + n_idx) * 16

    q_inv_rms = tl.load(q_inv_rms_ptr + bn_idx).to(tl.float32)
    k_inv_rms = tl.load(k_inv_rms_ptr + bn_idx).to(tl.float32)

    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    proj_q = tl.load(
        proj_q_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]
    ).to(tl.float32)
    proj_kv = tl.load(
        proj_kv_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]
    ).to(tl.float32)

    # First half: UCPE 4x4 projection over channel groups.
    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]
    mask_gj = mask_g[:, None]

    q_half = tl.load(q_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    k_half = tl.load(k_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    v_half = tl.load(v_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    q_w_half = tl.load(
        q_weight_ptr + weight_base + offs_gj, mask=mask_gj, other=0.0
    ).to(tl.float32)
    k_w_half = tl.load(
        k_weight_ptr + weight_base + offs_gj, mask=mask_gj, other=0.0
    ).to(tl.float32)

    q_half = tl.maximum(q_half * q_inv_rms * q_w_half, 0.0)
    k_half = tl.maximum(k_half * k_inv_rms * k_w_half, 0.0) * K_SCALE
    k_pre_half_sq = tl.sum(tl.where(mask_gj, k_half * k_half, 0.0))

    # out[g, i] = sum_j P[i, j] * in[g, j]
    q_half_out = tl.sum(q_half[:, None, :] * proj_q[None, :, :], axis=-1)
    k_half_out = tl.sum(k_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    v_half_out = tl.sum(v_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    k_post_half_sq = tl.sum(tl.where(mask_gj, k_half_out * k_half_out, 0.0))

    # Second half: interleaved-pair RoPE.
    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF
    rope_base = row_base + D_HALF
    rope_row = n_idx * D_HALF

    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(
        tl.float32
    )
    sin_v = tl.load(rope_sin_ptr + rope_row + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )

    q_r = tl.load(q_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    k_r = tl.load(k_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    v_r = tl.load(v_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    q_pair = tl.load(
        q_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)
    k_pair = tl.load(
        k_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)
    v_pair = tl.load(
        v_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)

    q_w = tl.load(
        q_weight_ptr + weight_base + D_HALF + offs_r, mask=mask_r, other=0.0
    ).to(tl.float32)
    k_w = tl.load(
        k_weight_ptr + weight_base + D_HALF + offs_r, mask=mask_r, other=0.0
    ).to(tl.float32)
    q_pair_w = tl.load(
        q_weight_ptr + weight_base + D_HALF + offs_r_pair,
        mask=mask_r_pair,
        other=0.0,
    ).to(tl.float32)
    k_pair_w = tl.load(
        k_weight_ptr + weight_base + D_HALF + offs_r_pair,
        mask=mask_r_pair,
        other=0.0,
    ).to(tl.float32)

    q_r = tl.maximum(q_r * q_inv_rms * q_w, 0.0)
    k_r = tl.maximum(k_r * k_inv_rms * k_w, 0.0) * K_SCALE
    q_pair = tl.maximum(q_pair * q_inv_rms * q_pair_w, 0.0)
    k_pair = tl.maximum(k_pair * k_inv_rms * k_pair_w, 0.0) * K_SCALE
    k_pre_rope_sq = tl.sum(tl.where(mask_r, k_r * k_r, 0.0))

    q_rope_out = q_r * cos_v + q_pair * sin_v
    k_rope_out = k_r * cos_v + k_pair * sin_v
    v_rope_out = v_r * cos_v + v_pair * sin_v
    k_post_rope_sq = tl.sum(tl.where(mask_r, k_rope_out * k_rope_out, 0.0))

    pre_sq = k_pre_half_sq + k_pre_rope_sq
    post_sq = k_post_half_sq + k_post_rope_sq
    inflation = tl.maximum(post_sq, 1.0e-12) / tl.maximum(pre_sq, 1.0e-12)

    out_base = b_idx * (H * D * N) + h_idx * (D * N) + n_idx
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]
    tl.store(q_out_ptr + out_base + offs_d_half * N, q_half_out, mask=mask_gj)
    tl.store(k_out_ptr + out_base + offs_d_half * N, k_half_out, mask=mask_gj)
    tl.store(v_out_ptr + out_base + offs_d_half * N, v_half_out, mask=mask_gj)

    offs_d_rope = D_HALF + offs_r
    tl.store(q_out_ptr + out_base + offs_d_rope * N, q_rope_out, mask=mask_r)
    tl.store(k_out_ptr + out_base + offs_d_rope * N, k_rope_out, mask=mask_r)
    tl.store(v_out_ptr + out_base + offs_d_rope * N, v_rope_out, mask=mask_r)
    tl.store(inflation_sq_ptr + (b_idx * H + h_idx) * N + n_idx, inflation)


@triton.jit
def _sana_wm_cam_qk_inv_rms_kernel(
    q_raw_ptr,
    k_raw_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bn_idx = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    row_base = bn_idx * C

    q = tl.load(q_raw_ptr + row_base + offs_c, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_raw_ptr + row_base + offs_c, mask=mask, other=0.0).to(tl.float32)

    inv_c = 1.0 / C
    q_inv_rms = tl.rsqrt(tl.sum(q * q, axis=0) * inv_c + EPS)
    k_inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) * inv_c + EPS)

    tl.store(q_inv_rms_ptr + bn_idx, q_inv_rms)
    tl.store(k_inv_rms_ptr + bn_idx, k_inv_rms)


def sana_wm_cam_qk_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute camera-branch Q/K inverse RMS in one Triton pass.

    Inputs are contiguous ``(B, N, H, D)`` tensors. Outputs are fp32
    ``(B, N)`` tensors consumed by the fused camera UCPE preprocess kernel.
    """
    if q_raw.shape != k_raw.shape:
        raise ValueError(f"q/k shape mismatch: {q_raw.shape}, {k_raw.shape}.")
    if q_raw.dim() != 4:
        raise ValueError(f"Expected q/k shape (B, N, H, D), got {q_raw.shape}.")
    if not q_raw.is_cuda:
        raise ValueError("sana_wm_cam_qk_inv_rms requires CUDA tensors.")
    q_raw = q_raw.contiguous()
    k_raw = k_raw.contiguous()

    B, N, H, D = q_raw.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), device=q_raw.device, dtype=torch.float32)
    k_inv_rms = torch.empty_like(q_inv_rms)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(q_raw.device):
        _sana_wm_cam_qk_inv_rms_kernel[(B * N,)](
            q_raw,
            k_raw,
            q_inv_rms,
            k_inv_rms,
            C=C,
            EPS=eps,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_inv_rms, k_inv_rms


@triton.jit
def _sana_wm_cam_softmax_prep_kernel(
    q_raw_ptr,
    k_raw_ptr,
    v_raw_ptr,
    q_inv_rms_ptr,
    k_inv_rms_ptr,
    q_weight_ptr,
    k_weight_ptr,
    proj_q_ptr,
    proj_kv_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    N_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D_ROPE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx - b_idx * N

    row_base = b_idx * (N * H * D) + n_idx * (H * D) + h_idx * D
    weight_base = h_idx * D
    proj_base = (b_idx * N + n_idx) * 16

    q_inv_rms = tl.load(q_inv_rms_ptr + bn_idx).to(tl.float32)
    k_inv_rms = tl.load(k_inv_rms_ptr + bn_idx).to(tl.float32)

    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    proj_q = tl.load(
        proj_q_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]
    ).to(tl.float32)
    proj_kv = tl.load(
        proj_kv_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]
    ).to(tl.float32)

    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]
    mask_gj = mask_g[:, None]

    q_half = tl.load(q_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    k_half = tl.load(k_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    v_half = tl.load(v_raw_ptr + row_base + offs_gj, mask=mask_gj, other=0.0).to(
        tl.float32
    )
    q_w_half = tl.load(
        q_weight_ptr + weight_base + offs_gj, mask=mask_gj, other=0.0
    ).to(tl.float32)
    k_w_half = tl.load(
        k_weight_ptr + weight_base + offs_gj, mask=mask_gj, other=0.0
    ).to(tl.float32)

    q_half = q_half * q_inv_rms * q_w_half
    k_half = k_half * k_inv_rms * k_w_half
    q_pre_half_sq = tl.sum(tl.where(mask_gj, q_half * q_half, 0.0))
    k_pre_half_sq = tl.sum(tl.where(mask_gj, k_half * k_half, 0.0))
    v_pre_half_sq = tl.sum(tl.where(mask_gj, v_half * v_half, 0.0))

    q_half_out = tl.sum(q_half[:, None, :] * proj_q[None, :, :], axis=-1)
    k_half_out = tl.sum(k_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    v_half_out = tl.sum(v_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    q_post_half_sq = tl.sum(tl.where(mask_gj, q_half_out * q_half_out, 0.0))
    k_post_half_sq = tl.sum(tl.where(mask_gj, k_half_out * k_half_out, 0.0))
    v_post_half_sq = tl.sum(tl.where(mask_gj, v_half_out * v_half_out, 0.0))

    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF
    rope_base = row_base + D_HALF
    rope_row = n_idx * D_HALF

    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(
        tl.float32
    )
    sin_v = tl.load(rope_sin_ptr + rope_row + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )

    q_r = tl.load(q_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    k_r = tl.load(k_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    v_r = tl.load(v_raw_ptr + rope_base + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    q_pair = tl.load(
        q_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)
    k_pair = tl.load(
        k_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)
    v_pair = tl.load(
        v_raw_ptr + rope_base + offs_r_pair, mask=mask_r_pair, other=0.0
    ).to(tl.float32)

    q_w = tl.load(
        q_weight_ptr + weight_base + D_HALF + offs_r, mask=mask_r, other=0.0
    ).to(tl.float32)
    k_w = tl.load(
        k_weight_ptr + weight_base + D_HALF + offs_r, mask=mask_r, other=0.0
    ).to(tl.float32)
    q_pair_w = tl.load(
        q_weight_ptr + weight_base + D_HALF + offs_r_pair,
        mask=mask_r_pair,
        other=0.0,
    ).to(tl.float32)
    k_pair_w = tl.load(
        k_weight_ptr + weight_base + D_HALF + offs_r_pair,
        mask=mask_r_pair,
        other=0.0,
    ).to(tl.float32)

    q_r = q_r * q_inv_rms * q_w
    k_r = k_r * k_inv_rms * k_w
    q_pair = q_pair * q_inv_rms * q_pair_w
    k_pair = k_pair * k_inv_rms * k_pair_w
    q_pre_rope_sq = tl.sum(tl.where(mask_r, q_r * q_r, 0.0))
    k_pre_rope_sq = tl.sum(tl.where(mask_r, k_r * k_r, 0.0))
    v_pre_rope_sq = tl.sum(tl.where(mask_r, v_r * v_r, 0.0))

    q_rope_out = q_r * cos_v + q_pair * sin_v
    k_rope_out = k_r * cos_v + k_pair * sin_v
    v_rope_out = v_r * cos_v + v_pair * sin_v
    q_post_rope_sq = tl.sum(tl.where(mask_r, q_rope_out * q_rope_out, 0.0))
    k_post_rope_sq = tl.sum(tl.where(mask_r, k_rope_out * k_rope_out, 0.0))
    v_post_rope_sq = tl.sum(tl.where(mask_r, v_rope_out * v_rope_out, 0.0))

    inv_d = 1.0 / D
    q_scale = tl.sqrt(
        (q_pre_half_sq + q_pre_rope_sq) * inv_d + EPS
    ) / tl.sqrt((q_post_half_sq + q_post_rope_sq) * inv_d + EPS)
    k_scale = tl.sqrt(
        (k_pre_half_sq + k_pre_rope_sq) * inv_d + EPS
    ) / tl.sqrt((k_post_half_sq + k_post_rope_sq) * inv_d + EPS)
    v_scale = tl.sqrt(
        (v_pre_half_sq + v_pre_rope_sq) * inv_d + EPS
    ) / tl.sqrt((v_post_half_sq + v_post_rope_sq) * inv_d + EPS)
    q_scale = tl.minimum(q_scale, 1.0)
    k_scale = tl.minimum(k_scale, 1.0)
    v_scale = tl.minimum(v_scale, 1.0)

    out_base = b_idx * (H * D * N) + h_idx * (D * N) + n_idx
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]
    tl.store(q_out_ptr + out_base + offs_d_half * N, q_half_out * q_scale, mask=mask_gj)
    tl.store(k_out_ptr + out_base + offs_d_half * N, k_half_out * k_scale, mask=mask_gj)
    tl.store(v_out_ptr + out_base + offs_d_half * N, v_half_out * v_scale, mask=mask_gj)

    offs_d_rope = D_HALF + offs_r
    tl.store(q_out_ptr + out_base + offs_d_rope * N, q_rope_out * q_scale, mask=mask_r)
    tl.store(k_out_ptr + out_base + offs_d_rope * N, k_rope_out * k_scale, mask=mask_r)
    tl.store(v_out_ptr + out_base + offs_d_rope * N, v_rope_out * v_scale, mask=mask_r)


@triton.jit
def _sana_wm_cam_output_apply_o_kernel(
    x_ptr,
    proj_o_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    out_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    N_GROUPS: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_N: tl.constexpr,
    STRIDE_D: tl.constexpr,
    OUT_STRIDE_B: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_D: tl.constexpr,
    BLOCK_D_ROPE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    h_idx = pid % H
    bn_idx = pid // H
    b_idx = bn_idx // N
    n_idx = bn_idx - b_idx * N

    in_base = (
        b_idx * STRIDE_B
        + h_idx * STRIDE_H
        + n_idx * STRIDE_N
    )
    proj_base = (b_idx * N + n_idx) * 16

    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, 4)
    proj_o = tl.load(
        proj_o_ptr + proj_base + offs_i[:, None] * 4 + offs_j[None, :]
    ).to(tl.float32)

    offs_g = tl.arange(0, BLOCK_GROUPS)
    mask_g = offs_g < N_GROUPS
    offs_gj = offs_g[:, None] * 4 + offs_j[None, :]
    mask_gj = mask_g[:, None]
    x_half = tl.load(
        x_ptr + in_base + offs_gj * STRIDE_D,
        mask=mask_gj,
        other=0.0,
    ).to(tl.float32)
    x_half_out = tl.sum(x_half[:, None, :] * proj_o[None, :, :], axis=-1)

    offs_r = tl.arange(0, BLOCK_D_ROPE)
    mask_r = offs_r < D_HALF
    offs_r_pair = offs_r ^ 1
    mask_r_pair = offs_r_pair < D_HALF
    rope_base = in_base + D_HALF * STRIDE_D
    rope_row = n_idx * D_HALF

    cos_v = tl.load(rope_cos_ptr + rope_row + offs_r, mask=mask_r, other=1.0).to(
        tl.float32
    )
    sin_v = tl.load(rope_sin_ptr + rope_row + offs_r, mask=mask_r, other=0.0).to(
        tl.float32
    )
    x_r = tl.load(
        x_ptr + rope_base + offs_r * STRIDE_D,
        mask=mask_r,
        other=0.0,
    ).to(tl.float32)
    x_pair = tl.load(
        x_ptr + rope_base + offs_r_pair * STRIDE_D,
        mask=mask_r_pair,
        other=0.0,
    ).to(tl.float32)
    x_rope_out = x_r * cos_v - x_pair * sin_v

    out_base = (
        b_idx * OUT_STRIDE_B
        + h_idx * OUT_STRIDE_H
        + n_idx * OUT_STRIDE_N
    )
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]
    tl.store(
        out_ptr + out_base + offs_d_half * OUT_STRIDE_D,
        x_half_out,
        mask=mask_gj,
    )
    tl.store(
        out_ptr + out_base + (D_HALF + offs_r) * OUT_STRIDE_D,
        x_rope_out,
        mask=mask_r,
    )


def sana_wm_cam_output_apply_o(
    x: torch.Tensor,
    proj_o: torch.Tensor,
    rotary_emb_cam: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply SANA-WM camera output inverse UCPE transform in Triton.

    ``x`` is ``(B, H, N, D)`` and may be non-contiguous. The output keeps
    the same shape and stride contract as ``x``.
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x shape (B, H, N, D), got {x.shape}.")
    if not x.is_cuda:
        raise ValueError("sana_wm_cam_output_apply_o requires CUDA tensors.")
    B, H, N, D = x.shape
    D_HALF = D // 2
    if D % 2 != 0 or D_HALF % 4 != 0:
        raise ValueError(f"SANA-WM camera output requires D/2 divisible by 4, got D={D}.")
    if proj_o.shape != (B, N, 4, 4):
        raise ValueError(
            f"SANA-WM camera output proj_o must be shaped (B, N, 4, 4), got {proj_o.shape}."
        )

    rope_cos, rope_sin = prepare_sana_wm_rope_tables(
        rotary_emb_cam,
        N,
        D_HALF,
        x.device,
    )
    out = torch.empty_strided(
        (B, H, N, D),
        x.stride(),
        device=x.device,
        dtype=x.dtype,
    )
    block_d_rope = triton.next_power_of_2(D_HALF)
    block_groups = triton.next_power_of_2(D_HALF // 4)
    stride_b, stride_h, stride_n, stride_d = x.stride()
    out_stride_b, out_stride_h, out_stride_n, out_stride_d = out.stride()

    with torch.get_device_module().device(x.device):
        _sana_wm_cam_output_apply_o_kernel[(B * N * H,)](
            x,
            proj_o.contiguous(),
            rope_cos,
            rope_sin,
            out,
            H=H,
            N=N,
            D=D,
            D_HALF=D_HALF,
            N_GROUPS=D_HALF // 4,
            STRIDE_B=stride_b,
            STRIDE_H=stride_h,
            STRIDE_N=stride_n,
            STRIDE_D=stride_d,
            OUT_STRIDE_B=out_stride_b,
            OUT_STRIDE_H=out_stride_h,
            OUT_STRIDE_N=out_stride_n,
            OUT_STRIDE_D=out_stride_d,
            BLOCK_D_ROPE=block_d_rope,
            BLOCK_GROUPS=block_groups,
            num_warps=1,
        )
    return out


def sana_wm_cam_gdn_preprocess(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: Optional[torch.Tensor],
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SANA-WM camera-branch RMSNorm, UCPE, RoPE, and layout conversion.

    Returns ``(q, k, v, inflation_sq)`` where ``q/k/v`` use contiguous
    ``(B, H, D, N)`` layout and ``inflation_sq`` has shape ``(B, H, N)``.
    """
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(
            f"q/k/v shape mismatch: {q_raw.shape}, {k_raw.shape}, {v_raw.shape}."
        )
    if q_raw.dim() != 4:
        raise ValueError(f"Expected q/k/v shape (B, N, H, D), got {q_raw.shape}.")
    if not q_raw.is_cuda:
        raise ValueError("sana_wm_cam_gdn_preprocess requires CUDA tensors.")

    q_inv_rms, k_inv_rms = sana_wm_cam_qk_inv_rms(q_raw, k_raw, eps=eps)
    return sana_wm_cam_gdn_preprocess_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        k_scale=k_scale,
        eps=eps,
    )


def sana_wm_cam_gdn_preprocess_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: Optional[torch.Tensor],
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Camera preprocess with caller-provided Q/K inverse RMS.

    TP keeps camera Q/K heads sharded but RMSNorm must still use the full
    hidden dimension. The SGLang runtime computes that cross-rank inv-RMS and
    passes the local norm-weight shard through this entry point.
    """
    del eps
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(
            f"q/k/v shape mismatch: {q_raw.shape}, {k_raw.shape}, {v_raw.shape}."
        )
    if q_raw.dim() != 4:
        raise ValueError(f"Expected q/k/v shape (B, N, H, D), got {q_raw.shape}.")
    if not q_raw.is_cuda:
        raise ValueError("sana_wm_cam_gdn_preprocess requires CUDA tensors.")
    if not q_inv_rms.is_cuda or not k_inv_rms.is_cuda:
        raise ValueError("SANA-WM camera q/k inv-RMS tensors must be CUDA tensors.")
    q_raw = q_raw.contiguous()
    k_raw = k_raw.contiguous()
    v_raw = v_raw.contiguous()

    B, N, H, D = q_raw.shape
    D_HALF = D // 2
    if D % 2 != 0 or D_HALF % 4 != 0:
        raise ValueError(f"SANA-WM camera prep requires D/2 divisible by 4, got D={D}.")
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        raise ValueError(
            "SANA-WM camera q/k norm weights must match H*D, got "
            f"{q_weight.numel()} and {k_weight.numel()} for H*D={H * D}."
        )
    if tuple(q_inv_rms.shape) != (B, N) or tuple(k_inv_rms.shape) != (B, N):
        raise ValueError(
            "SANA-WM camera q/k inv-RMS must be shaped (B, N), got "
            f"{q_inv_rms.shape} and {k_inv_rms.shape}."
        )
    if proj_q.shape != (B, N, 4, 4) or proj_kv.shape != (B, N, 4, 4):
        raise ValueError(
            "SANA-WM camera proj matrices must be shaped (B, N, 4, 4), got "
            f"{proj_q.shape} and {proj_kv.shape}."
        )

    rope_cos, rope_sin = prepare_sana_wm_rope_tables(
        rotary_emb_cam,
        N,
        D_HALF,
        q_raw.device,
    )

    q_out = torch.empty((B, H, D, N), device=q_raw.device, dtype=q_raw.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    inflation_sq = torch.empty((B, H, N), device=q_raw.device, dtype=torch.float32)

    block_d_rope = triton.next_power_of_2(D_HALF)
    block_groups = triton.next_power_of_2(D_HALF // 4)
    with torch.get_device_module().device(q_raw.device):
        _sana_wm_cam_prep_kernel[(B * N * H,)](
            q_raw,
            k_raw,
            v_raw,
            q_inv_rms.contiguous(),
            k_inv_rms.contiguous(),
            q_weight.contiguous(),
            k_weight.contiguous(),
            proj_q.contiguous(),
            proj_kv.contiguous(),
            rope_cos,
            rope_sin,
            q_out,
            k_out,
            v_out,
            inflation_sq,
            H=H,
            N=N,
            D=D,
            D_HALF=D_HALF,
            N_GROUPS=D_HALF // 4,
            K_SCALE=k_scale,
            BLOCK_D_ROPE=block_d_rope,
            BLOCK_GROUPS=block_groups,
            num_warps=1,
        )
    return q_out, k_out, v_out, inflation_sq


def sana_wm_cam_softmax_preprocess(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: Optional[torch.Tensor],
    *,
    norm_eps: float,
    downscale_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse softmax camera-branch RMSNorm, UCPE, RoPE, downscale, and layout.

    Returns ``(q, k, v)`` in contiguous ``(B, H, D, N)`` layout. Unlike the GDN
    camera branch, this path intentionally skips ReLU and key scaling because
    softmax attention applies its own scale.
    """
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(
            f"q/k/v shape mismatch: {q_raw.shape}, {k_raw.shape}, {v_raw.shape}."
        )
    if q_raw.dim() != 4:
        raise ValueError(f"Expected q/k/v shape (B, N, H, D), got {q_raw.shape}.")
    if not q_raw.is_cuda:
        raise ValueError("sana_wm_cam_softmax_preprocess requires CUDA tensors.")

    q_inv_rms, k_inv_rms = sana_wm_cam_qk_inv_rms(q_raw, k_raw, eps=norm_eps)
    return sana_wm_cam_softmax_preprocess_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
        downscale_eps=downscale_eps,
    )


def sana_wm_cam_softmax_preprocess_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rotary_emb_cam: Optional[torch.Tensor],
    *,
    downscale_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Softmax camera preprocess with caller-provided Q/K inverse RMS."""
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(
            f"q/k/v shape mismatch: {q_raw.shape}, {k_raw.shape}, {v_raw.shape}."
        )
    if q_raw.dim() != 4:
        raise ValueError(f"Expected q/k/v shape (B, N, H, D), got {q_raw.shape}.")
    if not q_raw.is_cuda:
        raise ValueError("sana_wm_cam_softmax_preprocess requires CUDA tensors.")
    if not q_inv_rms.is_cuda or not k_inv_rms.is_cuda:
        raise ValueError("SANA-WM camera q/k inv-RMS tensors must be CUDA tensors.")
    q_raw = q_raw.contiguous()
    k_raw = k_raw.contiguous()
    v_raw = v_raw.contiguous()

    B, N, H, D = q_raw.shape
    D_HALF = D // 2
    if D % 2 != 0 or D_HALF % 4 != 0:
        raise ValueError(f"SANA-WM camera prep requires D/2 divisible by 4, got D={D}.")
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        raise ValueError(
            "SANA-WM camera q/k norm weights must match H*D, got "
            f"{q_weight.numel()} and {k_weight.numel()} for H*D={H * D}."
        )
    if tuple(q_inv_rms.shape) != (B, N) or tuple(k_inv_rms.shape) != (B, N):
        raise ValueError(
            "SANA-WM camera q/k inv-RMS must be shaped (B, N), got "
            f"{q_inv_rms.shape} and {k_inv_rms.shape}."
        )
    if proj_q.shape != (B, N, 4, 4) or proj_kv.shape != (B, N, 4, 4):
        raise ValueError(
            "SANA-WM camera proj matrices must be shaped (B, N, 4, 4), got "
            f"{proj_q.shape} and {proj_kv.shape}."
        )

    rope_cos, rope_sin = prepare_sana_wm_rope_tables(
        rotary_emb_cam,
        N,
        D_HALF,
        q_raw.device,
    )

    q_out = torch.empty((B, H, D, N), device=q_raw.device, dtype=q_raw.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    block_d_rope = triton.next_power_of_2(D_HALF)
    block_groups = triton.next_power_of_2(D_HALF // 4)
    with torch.get_device_module().device(q_raw.device):
        _sana_wm_cam_softmax_prep_kernel[(B * N * H,)](
            q_raw,
            k_raw,
            v_raw,
            q_inv_rms.contiguous(),
            k_inv_rms.contiguous(),
            q_weight.contiguous(),
            k_weight.contiguous(),
            proj_q.contiguous(),
            proj_kv.contiguous(),
            rope_cos,
            rope_sin,
            q_out,
            k_out,
            v_out,
            H=H,
            N=N,
            D=D,
            D_HALF=D_HALF,
            N_GROUPS=D_HALF // 4,
            EPS=downscale_eps,
            BLOCK_D_ROPE=block_d_rope,
            BLOCK_GROUPS=block_groups,
            num_warps=1,
        )
    return q_out, k_out, v_out


def can_use_sana_wm_cam_output_apply_o(
    x: torch.Tensor,
    proj_o: Optional[torch.Tensor],
    rotary_emb_cam: Optional[torch.Tensor],
) -> bool:
    if x.dim() != 4 or not x.is_cuda:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    B, _, N, D = x.shape
    if D % 2 != 0 or (D // 2) % 4 != 0:
        return False
    if proj_o is None or tuple(proj_o.shape) != (B, N, 4, 4):
        return False
    if rotary_emb_cam is None:
        return True
    if not rotary_emb_cam.is_cuda:
        return False
    return tuple(rotary_emb_cam.squeeze(0).squeeze(0).shape) == (N, D // 4)


def can_use_sana_wm_cam_gdn_preprocess(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: Optional[torch.Tensor],
    proj_kv: Optional[torch.Tensor],
    rotary_emb_cam: Optional[torch.Tensor],
) -> bool:
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        return False
    if q_raw.dim() != 4 or not q_raw.is_cuda:
        return False
    if q_raw.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    B, N, H, D = q_raw.shape
    if D % 2 != 0 or (D // 2) % 4 != 0:
        return False
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        return False
    if proj_q is None or proj_kv is None:
        return False
    if tuple(proj_q.shape) != (B, N, 4, 4) or tuple(proj_kv.shape) != (B, N, 4, 4):
        return False
    if rotary_emb_cam is None:
        return True
    if not rotary_emb_cam.is_cuda:
        return False
    return tuple(rotary_emb_cam.squeeze(0).squeeze(0).shape) == (N, D // 4)


def can_use_sana_wm_cam_gdn_preprocess_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: Optional[torch.Tensor],
    proj_kv: Optional[torch.Tensor],
    rotary_emb_cam: Optional[torch.Tensor],
) -> bool:
    if not can_use_sana_wm_cam_gdn_preprocess(
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    ):
        return False
    B, N, _, _ = q_raw.shape
    if tuple(q_inv_rms.shape) != (B, N) or tuple(k_inv_rms.shape) != (B, N):
        return False
    return q_inv_rms.is_cuda and k_inv_rms.is_cuda


def can_use_sana_wm_cam_softmax_preprocess(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: Optional[torch.Tensor],
    proj_kv: Optional[torch.Tensor],
    rotary_emb_cam: Optional[torch.Tensor],
) -> bool:
    return can_use_sana_wm_cam_gdn_preprocess(
        q_raw,
        k_raw,
        v_raw,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    )


def can_use_sana_wm_cam_softmax_preprocess_with_inv_rms(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    proj_q: Optional[torch.Tensor],
    proj_kv: Optional[torch.Tensor],
    rotary_emb_cam: Optional[torch.Tensor],
) -> bool:
    return can_use_sana_wm_cam_gdn_preprocess_with_inv_rms(
        q_raw,
        k_raw,
        v_raw,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        proj_q,
        proj_kv,
        rotary_emb_cam,
    )
