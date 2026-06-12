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


def _precompute_inv_rms(
    qkv: torch.Tensor, idx: int, C: int, eps: float = 1e-5
) -> torch.Tensor:
    """Compute 1/RMS for one component of QKV over the full C = H*D channel dim.

    qkv: (B, N, 3, H, D); idx: 0=Q, 1=K, 2=V; C: H*D. Returns (B, N) float32.
    """
    raw = qkv[:, :, idx].float()  # (B, N, H, D)
    sq_sum = (raw * raw).sum(dim=(-2, -1))  # (B, N)
    return torch.rsqrt(sq_sum / C + eps)


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

    Replaces two ``_precompute_inv_rms`` scans with one launch that reads each
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
    :func:`fused_bigdn_bidi_chunkwise` from ``fused_gdn_chunkwise``.
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
