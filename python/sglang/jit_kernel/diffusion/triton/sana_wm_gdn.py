# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Inference-side helpers for the bidirectional fused GDN path.

Precision knob: env var ``FUSED_GDN_PRECISION`` or ``PRECISION_OVERRIDE``:
  0=IEEE fp32 dots, 1=TF32, 2=bf16 TC + fp32 state [default], 3=bf16 TC + bf16 state.
"""

# ruff: noqa: E501

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

# =====================================================================
#  GPU-adaptive kernel config
# =====================================================================


def _get_kernel_config() -> dict:
    """Return optimal kernel parameters for the current GPU.

    STATE_FP32 (fp32 state_prev) needs ~128KB SRAM (H100 228KB), vs ~96KB for
    bf16 state_prev (fits GB10's 101KB).
    """
    if not torch.cuda.is_available():
        return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 4, "STATE_FP32": False}
    smem = torch.cuda.get_device_properties(0).shared_memory_per_multiprocessor
    state_fp32 = smem >= 150 * 1024  # H100 (228KB) yes, GB10 (101KB) no
    return {"BLOCK_S": 64, "num_stages": 1, "num_warps": 8, "STATE_FP32": state_fp32}


_KCFG = None


def _kcfg():
    global _KCFG
    if _KCFG is None:
        _KCFG = _get_kernel_config()
    return _KCFG


# precision=0 → IEEE fp32 dots + fp32 state  (DOT_PRECISION=2, STATE_FP32=1)
# precision=1 → TF32  dots   + fp32 state    (DOT_PRECISION=1, STATE_FP32=1)
# precision=2 → bf16  dots   + fp32 state    (DOT_PRECISION=0, STATE_FP32=1) [default]
# precision=3 → bf16  dots   + bf16 state    (DOT_PRECISION=0, STATE_FP32=0)
def _precision_params(precision: int) -> tuple:
    if precision == 0:
        return 2, True
    elif precision == 1:
        return 1, True
    elif precision == 3:
        return 0, False
    else:  # default
        return 0, True


_env_prec = os.environ.get("FUSED_GDN_PRECISION", None)
PRECISION_OVERRIDE: int | None = int(_env_prec) if _env_prec is not None else None


def _resolve_launch_config() -> tuple:
    """Returns (prec, dot_prec, state_fp32, num_warps).

    Uses ``PRECISION_OVERRIDE`` when set, else ``_kcfg()`` (per-GPU SRAM).
    num_warps clamped to 4 when dots run on fp32 operands (more registers).
    """
    cfg = _kcfg()
    prec = PRECISION_OVERRIDE if PRECISION_OVERRIDE is not None else 2
    dot_prec, state_fp32 = _precision_params(prec)
    if PRECISION_OVERRIDE is None:
        state_fp32 = cfg["STATE_FP32"]
    nw = cfg["num_warps"]
    if dot_prec >= 1:
        nw = min(nw, 4)
    return prec, dot_prec, state_fp32, nw


def prepare_rope_tables(
    rotary_emb, N: int, D: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex rotary_emb `(1, 1, N, D//2)` → expanded (N, D) cos/sin tables.

    Encodes the interleaved-pair rotation
        y[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
        y[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
    as  y[d] = x[d]*cos_exp[d] + x[d^1]*sin_exp[d]
    where sin_exp[2i] = -sin[i], sin_exp[2i+1] = +sin[i].

    Returns (cos_exp, sin_exp) both (N, D) float32, contiguous.
    """
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)  # (N, D//2) complex
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack([-sin_half, sin_half], dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


# =====================================================================
#  Fused single-pass Q+K inverse-RMS Triton kernel
# =====================================================================
# Single Triton launch that reads each `(b, n)` row of `qkv` once and emits
# both `q_inv_rms[b, n]` and `k_inv_rms[b, n]`. Replaces two separate PyTorch
# scans (cast→square→sum→rsqrt) over `qkv[:, :, 0]` and `qkv[:, :, 1]`.
#
# Layout assumed: `qkv` is (B, N, 3, H, D) contiguous, so the C = H*D channels
# for a given (b, n, qkv_idx) live in a contiguous memory span.


@triton.jit
def _fused_qk_inv_rms_kernel(
    qkv_ptr,  # *T_in     (B, N, 3, H, D), contiguous
    q_inv_rms_ptr,  # *float32  (B, N)
    k_inv_rms_ptr,  # *float32  (B, N)
    N: tl.constexpr,
    C: tl.constexpr,  # H * D
    eps,
    BLOCK_C: tl.constexpr,
):
    bn_id = tl.program_id(0)
    qkv_row_stride = 3 * C
    row_base = bn_id * qkv_row_stride
    q_base = row_base
    k_base = row_base + C

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    q_vals = tl.load(qkv_ptr + q_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(qkv_ptr + k_base + offs, mask=mask, other=0.0).to(tl.float32)

    q_sq = tl.sum(q_vals * q_vals, axis=0)
    k_sq = tl.sum(k_vals * k_vals, axis=0)

    inv_c = 1.0 / C
    q_inv = tl.rsqrt(q_sq * inv_c + eps)
    k_inv = tl.rsqrt(k_sq * inv_c + eps)

    tl.store(q_inv_rms_ptr + bn_id, q_inv)
    tl.store(k_inv_rms_ptr + bn_id, k_inv)


def fused_qk_inv_rms(
    qkv: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass Triton fused Q+K inverse-RMS.

    Replaces two separate PyTorch RMS scans with one launch that reads each
    ``(b, n)`` row of ``qkv`` exactly once.
    qkv: (B, N, 3, H, D) contiguous. Returns (q_inv_rms, k_inv_rms), each (B, N) float32.
    """
    assert qkv.is_contiguous(), "qkv must be contiguous (B, N, 3, H, D)"
    assert (
        qkv.dim() == 5 and qkv.shape[2] == 3
    ), f"expected (B, N, 3, H, D), got {tuple(qkv.shape)}"
    B, N, _, H, D = qkv.shape
    C = H * D
    q_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    k_inv_rms = torch.empty((B, N), dtype=torch.float32, device=qkv.device)
    BLOCK_C = triton.next_power_of_2(C)
    _fused_qk_inv_rms_kernel[(B * N,)](
        qkv,
        q_inv_rms,
        k_inv_rms,
        N=N,
        C=C,
        eps=eps,
        BLOCK_C=BLOCK_C,
    )
    return q_inv_rms, k_inv_rms


# =====================================================================
#  Bidirectional GDN entry point (delegates to chunkwise)
# =====================================================================


def fused_bigdn_func(
    qkv: torch.Tensor,  # (B, N, 3, H, D)
    q_inv_rms: torch.Tensor,  # (B, N) float32
    k_inv_rms: torch.Tensor,  # (B, N) float32
    q_norm_weight: torch.Tensor,  # (C,) float32
    k_norm_weight: torch.Tensor,  # (C,) float32
    rope_cos: torch.Tensor,  # (N, D) float32
    rope_sin: torch.Tensor,  # (N, D) float32
    beta: torch.Tensor,  # (B, H, F, S)
    decay: torch.Tensor,  # (B, H, F)
    F: int,
    S: int,
    k_scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional fused GDN. Returns ``(B, N, H, D)``.

    Thin entry point kept for call-site stability; delegates to
    :func:`fused_bigdn_bidi_chunkwise` from ``sana_wm_gdn_chunkwise``.
    """
    from sglang.jit_kernel.diffusion.triton.sana_wm_gdn_chunkwise import (
        fused_bigdn_bidi_chunkwise,
    )

    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
    )


# =====================================================================
#  SANA-WM fused QKV + RMSNorm (+ optional RoPE) preprocessing
# =====================================================================
# Triton kernels that fuse the SANA-WM GDN preprocessing pipeline (Q/K
# RMSNorm + ReLU + K scale + transpose to (B, H, D, N)) into one launch.

def sana_wm_fused_qk_inv_rms(
    qkv: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SANA-WM-style alias for ``fused_qk_inv_rms`` (keyword ``eps``)."""
    return fused_qk_inv_rms(qkv, eps=eps)


def prepare_sana_wm_rope_tables(
    rotary_emb: Optional[torch.Tensor],
    N: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SANA-WM-style alias for ``prepare_rope_tables`` with shape validation."""
    if rotary_emb is None:
        return (
            torch.ones(N, D, device=device, dtype=torch.float32),
            torch.zeros(N, D, device=device, dtype=torch.float32),
        )
    freqs = rotary_emb.squeeze(0).squeeze(0)
    if freqs.shape != (N, D // 2):
        raise ValueError(
            f"Expected rotary_emb shape compatible with (N, D/2)={(N, D // 2)}, "
            f"got {tuple(freqs.shape)}."
        )
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1)
    rope_sin = torch.stack((-sin_half, sin_half), dim=-1).reshape(N, D)
    return rope_cos.contiguous(), rope_sin.contiguous()


@triton.jit
def _sana_wm_qkv_gdn_preprocess_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bn = tl.program_id(0)
    b = bn // N
    n = bn - b * N

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    h = offs_c // D
    d = offs_c - h * D

    qkv_base = bn * 3 * C
    q = tl.load(qkv_ptr + qkv_base + offs_c, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(qkv_ptr + qkv_base + C + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )

    q_rstd = tl.rsqrt(tl.sum(q * q, axis=0) / C + EPS)
    k_rstd = tl.rsqrt(tl.sum(k * k, axis=0) / C + EPS)

    q_weight = tl.load(q_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    q = tl.maximum(q * q_rstd * q_weight, 0.0)
    k = tl.maximum(k * k_rstd * k_weight, 0.0) * K_SCALE
    v = tl.load(qkv_ptr + qkv_base + 2 * C + offs_c, mask=mask, other=0.0)

    out_base = ((b * H + h) * D + d) * N + n
    tl.store(q_out_ptr + out_base, q, mask=mask)
    tl.store(k_out_ptr + out_base, k, mask=mask)
    tl.store(v_out_ptr + out_base, v, mask=mask)


def sana_wm_qkv_gdn_preprocess(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SANA-WM GDN Q/K RMSNorm, ReLU, K scale, and QKV transpose.

    Returns ``(q, k, v)`` tensors with contiguous ``(B, H, D, N)`` layout.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape (B, N, 3, H, D), got {qkv.shape}.")
    if not qkv.is_cuda:
        raise ValueError("sana_wm_qkv_gdn_preprocess requires a CUDA tensor.")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    B, N, _, H, D = qkv.shape
    C = H * D
    if q_weight.numel() != C or k_weight.numel() != C:
        raise ValueError(
            "SANA-WM q/k norm weights must match the flattened local channel "
            f"dimension C={C}, got {q_weight.numel()} and {k_weight.numel()}."
        )

    q_out = torch.empty((B, H, D, N), device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(qkv.device):
        _sana_wm_qkv_gdn_preprocess_kernel[(B * N,)](
            qkv,
            q_out,
            k_out,
            v_out,
            q_weight.contiguous(),
            k_weight.contiguous(),
            N=N,
            H=H,
            D=D,
            C=C,
            EPS=eps,
            K_SCALE=k_scale,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out


@triton.jit
def _sana_wm_qkv_gdn_preprocess_rope_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_rot_out_ptr,
    k_rot_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    bn = tl.program_id(0)
    b = bn // N
    n = bn - b * N

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    h = offs_c // D
    d = offs_c - h * D
    d_pair = d + 1 - 2 * (d % 2)
    offs_c_pair = h * D + d_pair

    qkv_base = bn * 3 * C
    q_raw = tl.load(qkv_ptr + qkv_base + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )
    k_raw = tl.load(qkv_ptr + qkv_base + C + offs_c, mask=mask, other=0.0).to(
        tl.float32
    )
    q_pair_raw = tl.load(
        qkv_ptr + qkv_base + offs_c_pair, mask=mask, other=0.0
    ).to(tl.float32)
    k_pair_raw = tl.load(
        qkv_ptr + qkv_base + C + offs_c_pair, mask=mask, other=0.0
    ).to(tl.float32)

    q_rstd = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) / C + EPS)
    k_rstd = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) / C + EPS)

    q_weight = tl.load(q_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
    q_pair_weight = tl.load(q_weight_ptr + offs_c_pair, mask=mask, other=0.0).to(
        tl.float32
    )
    k_pair_weight = tl.load(k_weight_ptr + offs_c_pair, mask=mask, other=0.0).to(
        tl.float32
    )

    q = tl.maximum(q_raw * q_rstd * q_weight, 0.0)
    k = tl.maximum(k_raw * k_rstd * k_weight, 0.0) * K_SCALE
    q_pair = tl.maximum(q_pair_raw * q_rstd * q_pair_weight, 0.0)
    k_pair = tl.maximum(k_pair_raw * k_rstd * k_pair_weight, 0.0) * K_SCALE
    v = tl.load(qkv_ptr + qkv_base + 2 * C + offs_c, mask=mask, other=0.0)

    rope_cos = tl.load(rope_cos_ptr + n * D + d, mask=mask, other=1.0).to(
        tl.float32
    )
    rope_sin = tl.load(rope_sin_ptr + n * D + d, mask=mask, other=0.0).to(
        tl.float32
    )
    q_rot = q * rope_cos + q_pair * rope_sin
    k_rot = k * rope_cos + k_pair * rope_sin

    out_base = ((b * H + h) * D + d) * N + n
    tl.store(q_out_ptr + out_base, q, mask=mask)
    tl.store(k_out_ptr + out_base, k, mask=mask)
    tl.store(v_out_ptr + out_base, v, mask=mask)
    tl.store(q_rot_out_ptr + out_base, q_rot, mask=mask)
    tl.store(k_rot_out_ptr + out_base, k_rot, mask=mask)


def sana_wm_qkv_gdn_preprocess_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    *,
    k_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SANA-WM GDN preprocessing and RoPE application.

    Returns ``(q, k, v, q_rot, k_rot)`` in contiguous ``(B, H, D, N)`` layout.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape (B, N, 3, H, D), got {qkv.shape}.")
    if not qkv.is_cuda:
        raise ValueError("sana_wm_qkv_gdn_preprocess_rope requires a CUDA tensor.")
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    B, N, _, H, D = qkv.shape
    C = H * D
    if D % 2 != 0:
        raise ValueError(f"SANA-WM RoPE preprocessing requires even D, got D={D}.")
    if q_weight.numel() != C or k_weight.numel() != C:
        raise ValueError(
            "SANA-WM q/k norm weights must match the flattened local channel "
            f"dimension C={C}, got {q_weight.numel()} and {k_weight.numel()}."
        )

    rope_cos, rope_sin = prepare_sana_wm_rope_tables(rotary_emb, N, D, qkv.device)
    q_out = torch.empty((B, H, D, N), device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty_like(q_out)
    v_out = torch.empty_like(q_out)
    q_rot_out = torch.empty_like(q_out)
    k_rot_out = torch.empty_like(q_out)
    block_c = triton.next_power_of_2(C)
    num_warps = 8 if block_c >= 2048 else 4

    with torch.get_device_module().device(qkv.device):
        _sana_wm_qkv_gdn_preprocess_rope_kernel[(B * N,)](
            qkv,
            q_out,
            k_out,
            v_out,
            q_rot_out,
            k_rot_out,
            q_weight.contiguous(),
            k_weight.contiguous(),
            rope_cos,
            rope_sin,
            N=N,
            H=H,
            D=D,
            C=C,
            EPS=eps,
            K_SCALE=k_scale,
            BLOCK_C=block_c,
            num_warps=num_warps,
        )
    return q_out, k_out, v_out, q_rot_out, k_rot_out


def can_use_sana_wm_qkv_gdn_preprocess(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        return False
    if not qkv.is_cuda or not qkv.is_contiguous():
        return False
    if qkv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    _, _, _, H, D = qkv.shape
    return q_weight.numel() == H * D and k_weight.numel() == H * D


def can_use_sana_wm_fused_qk_inv_rms(qkv: torch.Tensor) -> bool:
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        return False
    if not qkv.is_cuda or not qkv.is_contiguous():
        return False
    return qkv.dtype in (torch.float16, torch.bfloat16, torch.float32)


def can_use_sana_wm_qkv_gdn_preprocess_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
) -> bool:
    if not can_use_sana_wm_qkv_gdn_preprocess(qkv, q_weight, k_weight):
        return False
    _, N, _, _, D = qkv.shape
    if D % 2 != 0:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)


# =====================================================================
#  TP-friendly bidirectional GDN entry points (caller-provided inv-RMS)
# =====================================================================

def sana_wm_fused_bigdn_bidi_with_inv_rms(
    qkv: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
    k_scale: float,
    eps: float,
    norm_eps: float,
    dot_precision: int = 0,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    init_state_kv: Optional[torch.Tensor] = None,
    init_state_z: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor:
    """Run fused bidirectional GDN with caller-provided Q/K inv-RMS.

    SANA-WM TP keeps Q/K heads sharded but normalizes over the full hidden
    dimension. The runtime computes the cross-rank inv-RMS and passes the local
    norm-weight shard through this entry point.
    """
    from sglang.jit_kernel.diffusion.triton.sana_wm_gdn_chunkwise import (
        fused_bigdn_bidi_chunkwise,
    )

    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    _, N, _, _, D = qkv.shape
    if rope_cos is None or rope_sin is None:
        if rope_cos is not None or rope_sin is not None:
            raise ValueError("rope_cos and rope_sin must be provided together.")
        rope_cos, rope_sin = prepare_sana_wm_rope_tables(
            rotary_emb, N, D, qkv.device
        )
    return fused_bigdn_bidi_chunkwise(
        qkv,
        q_inv_rms.contiguous(),
        k_inv_rms.contiguous(),
        q_weight.float().contiguous(),
        k_weight.float().contiguous(),
        rope_cos.contiguous(),
        rope_sin.contiguous(),
        beta.contiguous(),
        decay.contiguous(),
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_final_state=return_final_state,
    )


def sana_wm_fused_bigdn_bidi(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
    k_scale: float,
    eps: float,
    norm_eps: float,
    dot_precision: int = 0,
    init_state_kv: Optional[torch.Tensor] = None,
    init_state_z: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> torch.Tensor:
    """Run the upstream Sana-WM fused bidirectional GDN pipeline.

    Computes Q/K inv-RMS internally with :func:`fused_qk_inv_rms` then forwards
    to :func:`sana_wm_fused_bigdn_bidi_with_inv_rms`.
    """
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()

    q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)
    return sana_wm_fused_bigdn_bidi_with_inv_rms(
        qkv,
        q_inv_rms,
        k_inv_rms,
        q_weight,
        k_weight,
        rotary_emb,
        beta,
        decay,
        F=F,
        S=S,
        k_scale=k_scale,
        eps=eps,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        return_final_state=return_final_state,
    )


def can_use_sana_wm_fused_bigdn_bidi_with_inv_rms(
    qkv: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
) -> bool:
    if not can_use_sana_wm_fused_qk_inv_rms(qkv):
        return False
    B, N, _, H, D = qkv.shape
    if N != F * S or F <= 0 or S <= 0:
        return False
    if tuple(q_inv_rms.shape) != (B, N) or tuple(k_inv_rms.shape) != (B, N):
        return False
    if not q_inv_rms.is_cuda or not k_inv_rms.is_cuda:
        return False
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        return False
    if D % 2 != 0:
        return False
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    if tuple(beta.shape) != (B, H, F, S) or tuple(decay.shape) != (B, H, F):
        return False
    if not beta.is_cuda or not decay.is_cuda:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)


def can_use_sana_wm_fused_bigdn_bidi(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rotary_emb: Optional[torch.Tensor],
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    F: int,
    S: int,
) -> bool:
    if not can_use_sana_wm_fused_qk_inv_rms(qkv):
        return False
    B, N, _, H, D = qkv.shape
    if N != F * S or F <= 0 or S <= 0:
        return False
    if q_weight.numel() != H * D or k_weight.numel() != H * D:
        return False
    if D % 2 != 0:
        return False
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    if tuple(beta.shape) != (B, H, F, S) or tuple(decay.shape) != (B, H, F):
        return False
    if not beta.is_cuda or not decay.is_cuda:
        return False
    if rotary_emb is None:
        return True
    if not rotary_emb.is_cuda:
        return False
    return tuple(rotary_emb.squeeze(0).squeeze(0).shape) == (N, D // 2)


# =====================================================================
#  Camera scan thin shim (delegates to chunkwise cam_scan_bidi_chunkwise)
# =====================================================================

def sana_wm_cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: Optional[int] = None,
) -> torch.Tensor:
    """SGLang wrapper for Sana's full phase-A/B/C camera scan port."""
    from sglang.jit_kernel.diffusion.triton.sana_wm_gdn_chunkwise import (
        cam_scan_bidi_chunkwise,
    )

    return cam_scan_bidi_chunkwise(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        beta.contiguous(),
        decay.contiguous(),
        dot_precision=dot_precision,
    )


def can_use_sana_wm_cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> bool:
    if q.shape != k.shape or q.shape != v.shape:
        return False
    if q.dim() != 4 or not q.is_cuda:
        return False
    if q.dtype != torch.float32:
        return False
    B, H, D, N = q.shape
    if beta.ndim != 4 or decay.ndim != 3:
        return False
    T = beta.shape[2]
    if T <= 0 or N % T != 0:
        return False
    S = N // T
    return tuple(beta.shape) == (B, H, T, S) and tuple(decay.shape) == (B, H, T)

# =====================================================================
#  SANA-WM camera-branch preprocessing (UCPE 4x4 + RMSNorm + RoPE)
# =====================================================================

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
    DOWNSCALE_EPS: tl.constexpr,
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
    q_pre_half_sq = tl.sum(tl.where(mask_gj, q_half * q_half, 0.0))
    k_pre_half_sq = tl.sum(tl.where(mask_gj, k_half * k_half, 0.0))
    v_pre_half_sq = tl.sum(tl.where(mask_gj, v_half * v_half, 0.0))

    # out[g, i] = sum_j P[i, j] * in[g, j]
    q_half_out = tl.sum(q_half[:, None, :] * proj_q[None, :, :], axis=-1)
    k_half_out = tl.sum(k_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    v_half_out = tl.sum(v_half[:, None, :] * proj_kv[None, :, :], axis=-1)
    q_post_half_sq = tl.sum(tl.where(mask_gj, q_half_out * q_half_out, 0.0))
    k_post_half_sq = tl.sum(tl.where(mask_gj, k_half_out * k_half_out, 0.0))
    v_post_half_sq = tl.sum(tl.where(mask_gj, v_half_out * v_half_out, 0.0))

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
    q_pre_sq = q_pre_half_sq + q_pre_rope_sq
    k_pre_sq = k_pre_half_sq + k_pre_rope_sq
    v_pre_sq = v_pre_half_sq + v_pre_rope_sq
    q_post_sq = q_post_half_sq + q_post_rope_sq
    k_post_sq = k_post_half_sq + k_post_rope_sq
    v_post_sq = v_post_half_sq + v_post_rope_sq
    q_scale = tl.sqrt(q_pre_sq * inv_d + DOWNSCALE_EPS) / tl.sqrt(
        q_post_sq * inv_d + DOWNSCALE_EPS
    )
    k_scale = tl.sqrt(k_pre_sq * inv_d + DOWNSCALE_EPS) / tl.sqrt(
        k_post_sq * inv_d + DOWNSCALE_EPS
    )
    v_scale = tl.sqrt(v_pre_sq * inv_d + DOWNSCALE_EPS) / tl.sqrt(
        v_post_sq * inv_d + DOWNSCALE_EPS
    )
    q_scale = tl.minimum(q_scale, 1.0)
    k_scale = tl.minimum(k_scale, 1.0)
    v_scale = tl.minimum(v_scale, 1.0)
    k_post_scaled_sq = k_post_sq * k_scale * k_scale
    inflation = tl.maximum(k_post_scaled_sq, 1.0e-12) / tl.maximum(
        k_pre_sq,
        1.0e-12,
    )

    out_base = b_idx * (H * D * N) + h_idx * (D * N) + n_idx
    offs_d_half = offs_g[:, None] * 4 + offs_i[None, :]
    tl.store(
        q_out_ptr + out_base + offs_d_half * N,
        q_half_out * q_scale,
        mask=mask_gj,
    )
    tl.store(
        k_out_ptr + out_base + offs_d_half * N,
        k_half_out * k_scale,
        mask=mask_gj,
    )
    tl.store(
        v_out_ptr + out_base + offs_d_half * N,
        v_half_out * v_scale,
        mask=mask_gj,
    )

    offs_d_rope = D_HALF + offs_r
    tl.store(q_out_ptr + out_base + offs_d_rope * N, q_rope_out * q_scale, mask=mask_r)
    tl.store(k_out_ptr + out_base + offs_d_rope * N, k_rope_out * k_scale, mask=mask_r)
    tl.store(v_out_ptr + out_base + offs_d_rope * N, v_rope_out * v_scale, mask=mask_r)
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
    downscale_eps: float = 1e-6,
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
        downscale_eps=downscale_eps,
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
    downscale_eps: float = 1e-6,
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
            DOWNSCALE_EPS=downscale_eps,
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
