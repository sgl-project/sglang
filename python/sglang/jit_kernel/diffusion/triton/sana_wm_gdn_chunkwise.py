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

# ruff: noqa: E501

"""
Fused GDN — Chunkwise-parallel forward (v2).

V2 changes vs v1:
  1. Phase A split into two kernels along GDN data streams (KV and Z — gating
     sub-paths, not CUDA streams; both launch on the same CUDA stream). Z is
     lighter (no V/Cos/Sin loads, no K_pair flip), enabling 2 blocks/SM
     resident on H100 for latency hiding.
  2. Phase A stores (I - P_kv) / (I - P_z) instead of P_kv/P_z so Phase B's
     MMA `(I-P_kv) @ M` folds the identity-add in (no separate elementwise pass).

BiGDN inference path: QK_NORM=1, USE_PRECOMPUTED_RMS=1, SAVE_STATE=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_CAM_IDENTITY_CACHE: dict = {}

# Per-architecture launch config (auto-selected via compute capability).
# Empirically tuned at production config (B=1..8, T=11, S=920, H=20, D=112).
# Two effects drive BLOCK_S:
#   1. Precision: fp32 operand fragments are 2× bf16. BLOCK_S=64 + fp32 → register
#      spills (40-100× slower); BLOCK_S=32 + fp32 → no spills. fp32 forces BLOCK_S=32.
#   2. Arch (bf16): A100 (192 KB SRAM) prefers BLOCK_S=32; H100/GB200 (228 KB)
#      tolerate BLOCK_S=64.
#
# 8 tuned knobs across 3 phases: Phase A (nw, BS); Phase B (nw, use_acc, ns);
# Phase C (nw, BS, ns). Values from empirical sweeps (T6 A100/H100 2026-04-19;
# Blackwell-DC 2026-04-20; Spark GB10 in 5da52db6 / 3ad104d0). Phase B persistent
# M[128,128] fp32 = 64 KB → nw controls register spread; Phase C loaded M = 64 KB
# → BS controls transient SMEM. New arch: pick closest bucket, then override in
# _CHUNKWISE_SHAPE_OVERRIDES once a targeted sweep lands.


@dataclass(frozen=True)
class _PhaseCfg:
    nw: int  # num_warps
    BS: int = 0  # BLOCK_S (Phase A/C only; 0 = N/A for Phase B)
    ns: int = 1  # num_stages
    use_acc: bool = False  # Phase B only: fold A_f via MMA accumulator


@dataclass(frozen=True)
class _ChunkwiseCfg:
    A: _PhaseCfg
    B: _PhaseCfg
    C: _PhaseCfg

    def as_tuple(self) -> tuple:
        """Flatten to the 8-tuple the legacy API returns."""
        return (
            self.A.nw,
            self.A.BS,
            self.B.nw,
            self.B.ns,
            self.B.use_acc,
            self.C.nw,
            self.C.BS,
            self.C.ns,
        )


# ──────────────────────────────────────────────────────────────────
# Primary tuning table: (arch_key, prec_key) → _ChunkwiseCfg.
# Arch keys:
#   "ampere"          sm_80     A100 (164 KB SRAM, no WGMMA)
#   "hopper"          sm_90     H100 (228 KB SRAM, WGMMA)
#   "blackwell_dc"    sm_100    B200 / GB200 (228 KB SRAM, WGMMA v2)
#   "blackwell_spark" sm_120+ with < 150 KB SRAM  5090 / GB10 (~102 KB SRAM)
# Prec keys:
#   "bf16"  dot_prec == 0  (bf16 TC, half-size operand fragments)
#   "fp32"  dot_prec >= 1  (TF32 TC or IEEE Markidis 3-pass; same launch shape)
# ──────────────────────────────────────────────────────────────────
_CHUNKWISE_TUNING: dict[tuple[str, str], _ChunkwiseCfg] = {
    # A100: smaller SRAM than Hopper, no WGMMA → bigger CTAs hide MMA latency.
    # Phase B fp32 needs nw=32 to spread persistent M across warps (no acc-fusion
    # available pre-Hopper, so ns=2 fills the MMA pipeline slot instead).
    ("ampere", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=4, BS=32, ns=1),  # nw=4 bf16 C: 27% faster than nw=8 per T6
    ),
    ("ampere", "fp32"): _ChunkwiseCfg(
        # 2026-04-30 retune: Phase A nw=16 BS=32 is 8-13× faster than the legacy
        # nw=8 across all F; Phase A was the A100 sink/rolling bottleneck.
        A=_PhaseCfg(nw=16, BS=32),
        B=_PhaseCfg(nw=32, use_acc=False, ns=2),  # ns=2 fills pipe (no acc-fusion)
        C=_PhaseCfg(
            nw=16, BS=32, ns=1
        ),  # 2026-04-30 retune: nw=16 BS=32 is 2.8x faster (was nw=8 BS=16)
    ),
    # Hopper (H100): WGMMA + 228 KB SRAM → big tiles win at bf16.
    # Phase B fp32 uses acc-fusion (MMA accumulator folds A_f in one op, +12%).
    ("hopper", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),  # small CTAs pack better on WGMMA
        C=_PhaseCfg(nw=8, BS=32, ns=1),
    ),
    ("hopper", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),  # fp32 operand 2× bigger → half BS
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=1
        ),  # 2026-04-29 retune: acc_fusion=False is 3x faster post precision-gate fix
        C=_PhaseCfg(
            nw=16, BS=32, ns=1
        ),  # 2026-04-30 retune: nw=16 BS=32 is 1.7x faster (was nw=8 BS=16)
    ),
    # Blackwell-DC (B200 / GB200): 228 KB SRAM + improved WGMMA codegen.
    # bf16 likes small CTAs (nw=4); fp32 stays at nw=8 (nw=4 + BS=64 fp32 = 92× regression).
    ("blackwell_dc", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=4, BS=64),
        B=_PhaseCfg(nw=4, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=64, ns=1),  # 228 KB SRAM leaves room for BS=64 bf16
    ),
    ("blackwell_dc", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(
            nw=8, BS=128
        ),  # 2026-04-30 retune: nw=8 BS=128 ~5% faster at production F=3-6 (sweep across F=3,5,6,11)
        B=_PhaseCfg(
            nw=32, use_acc=False, ns=3
        ),  # 2026-04-29 retune: 14x faster (was nw=8 acc=True 17ms; now nw=32 ns=3 acc=False 1.23ms)
        C=_PhaseCfg(
            nw=4, BS=64, ns=1
        ),  # 2026-04-30 retune: nw=4 BS=64 is 3-5x faster than old nw=8 BS=16 (sweep 2026-04-30)
    ),
    # Blackwell-Spark (5090 / GB10, ~102 KB SRAM): small-chip SRAM penalty, no
    # Blackwell-DC WGMMA-v2 register-spread benefit. Behaves like Hopper at fp32
    # (Phase B wants nw=32, not nw=8 like DC). BS one step smaller than DC; Phase A
    # bf16 wants nw=8 (nw=4 was 22× slower, 2026-04-20). Phase B nw=32 fp32:
    # 1.84×/2.65× (GB10/5090) over prior nw=8 (sweep 2026-04-24, F=11 S=920).
    ("blackwell_spark", "bf16"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=32),
        B=_PhaseCfg(
            nw=8, use_acc=False, ns=1
        ),  # nw=8 (not 4) at bf16: ~5% across F=3,6,11
        # C.nw=4 BS=32: ~3.5% faster than nw=8 (Phase C is bandwidth-bound, fewer
        # warps schedule better on small SRAM). BS=64 bf16 OOMs Spark SRAM.
        C=_PhaseCfg(nw=4, BS=32, ns=1),
    ),
    ("blackwell_spark", "fp32"): _ChunkwiseCfg(
        A=_PhaseCfg(nw=8, BS=16),  # fp32 operand 2× bigger → BS=16 (half of DC's 32)
        # nw=16 OOMs the 102 KB SRAM cap at TF32 (needs 131 KB); nw=8 fits and is
        # within noise. The D-tile path (auto-enabled on spark, see
        # `_pick_phase_b_d_splits`) is ~2.6× faster at TF32 / ~13% at IEEE — these
        # baseline params only apply when PHASE_B_D_SPLITS=1 is forced.
        B=_PhaseCfg(nw=8, use_acc=False, ns=1),
        C=_PhaseCfg(nw=8, BS=16, ns=1),  # binding constraint: M.fp32 64 KB + Q stage
    ),
}


# Shape-aware override table (empty by default). Keyed by
# (arch_key, prec_key, shape_hint); exact-match, values are full `_ChunkwiseCfg`
# (no partial overrides). Strictly additive — base table is the fallback.
# Populate only when a targeted sweep shows a shape regresses with the arch config.
_CHUNKWISE_SHAPE_OVERRIDES: dict[tuple[str, str, str], _ChunkwiseCfg] = {}


# Per-(cap, dot_prec) exact overrides (pins a specific GPU model if the arch
# bucket is wrong for it). Also empty by default.
_ARCH_OVERRIDES: dict = {}


def _arch_key(cap: tuple) -> str:
    """Map compute capability → named arch bucket in `_CHUNKWISE_TUNING`.

    Blackwell (cap[0] >= 10) splits into "blackwell_dc"/"blackwell_spark" by SRAM
    size (≥150 KB vs less). Unknown archs / no CUDA → conservative "ampere".
    """
    if cap[0] == 8:
        return "ampere"
    if cap[0] == 9:
        return "hopper"
    if cap[0] >= 10:
        has_big_sram = True
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            smem = getattr(props, "shared_memory_per_multiprocessor", 228 * 1024)
            has_big_sram = smem >= 150 * 1024
        return "blackwell_dc" if has_big_sram else "blackwell_spark"
    return "ampere"


def _prec_key(dot_prec: int) -> str:
    return "fp32" if dot_prec >= 1 else "bf16"


def _auto_config(dot_prec: int, cap: tuple, shape_hint: str | None = None) -> tuple:
    """Look up chunkwise launch params. Resolution order: shape override →
    per-(arch, prec) table → ("ampere", prec) fallback. (`_ARCH_OVERRIDES` is
    applied by `_get_arch_config`, not here.)

    Returns the legacy 8-tuple `(a_nw, a_BS, b_nw, b_ns, b_use_acc, c_nw, c_BS, c_ns)`.
    """
    arch = _arch_key(cap)
    prec = _prec_key(dot_prec)

    if shape_hint is not None:
        cfg = _CHUNKWISE_SHAPE_OVERRIDES.get((arch, prec, shape_hint))
        if cfg is not None:
            return cfg.as_tuple()

    cfg = _CHUNKWISE_TUNING.get((arch, prec)) or _CHUNKWISE_TUNING[("ampere", prec)]
    return cfg.as_tuple()


def _get_arch_config(
    dot_precision: int = 0,
    shape_hint: str | None = None,
    device: torch.device | int | None = None,
):
    """Returns (a_warps, a_BLOCK_S, b_warps, b_stages, b_use_acc_fusion,
                c_warps, c_BLOCK_S, c_stages).

    dot_precision: 0=bf16 TC, 1=TF32 TC, 2=IEEE fp32.
    device: capability source; pass ``qkv.device`` in multi-GPU single-process
            setups so the right tuning bucket is chosen (defaults to current device).
    """
    if not torch.cuda.is_available():
        cap = (9, 0)  # assume modern when querying from CPU
    else:
        if device is None:
            dev_idx = torch.cuda.current_device()
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        cap = torch.cuda.get_device_capability(dev_idx)
    key = (cap, dot_precision)
    if key in _ARCH_OVERRIDES:
        return _ARCH_OVERRIDES[key]
    return _auto_config(dot_precision, cap, shape_hint)


# ════════════════════════════════════════════════════════════════
#  Phase A — split into KV and Z kernels
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_a_kv_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    I_minus_P_kv_ptr,  # output: (I - K_rot^T diag(β) K_rot)
    A_ptr,  # output: K_rot^T diag(β) V
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    SKIP_RELU: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_kv_bhf = (
        I_minus_P_kv_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    )
    A_bhf = A_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    k_nw_pair = tl.load(
        k_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0
    ).to(tl.float32)

    # fp32 accumulators avoid bf16 round-off compounding across the loop.
    P_kv_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    A_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 1 * stride_3
            + offs_d[None, :] * stride_d
        )
        v_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 2 * stride_3
            + offs_d[None, :] * stride_d
        )
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        V_raw = tl.load(v_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(
            k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        if SKIP_RELU:
            K = K_normed * k_scale
        else:
            K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        K_pair_raw = tl.reshape(
            tl.flip(tl.reshape(K_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )
        K_pair_normed = K_pair_raw * k_inv_rms[:, None] * k_nw_pair[None, :]
        if SKIP_RELU:
            K_pair = K_pair_normed * k_scale
        else:
            K_pair = tl.where(K_pair_normed > 0, K_pair_normed, 0.0) * k_scale

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        K_rot = K * Cos + K_pair * Sin

        beta_Krot = beta_t[:, None] * K_rot
        beta_V = beta_t[:, None] * V_raw

        K_rot_T = tl.trans(K_rot)
        P_kv_acc += tl.dot(
            K_rot_T.to(dot_dtype),
            beta_Krot.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        A_acc += tl.dot(
            K_rot_T.to(dot_dtype),
            beta_V.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )

    # Padded positions are 0 by construction (K_rot is 0 outside D).
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (
        (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    )
    I_minus_P_kv = tl.where(diag_in_range, 1.0 - P_kv_acc, -P_kv_acc)
    if DOT_PRECISION >= 1:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv)
        tl.store(A_bhf + offs_dd, A_acc)
    else:
        tl.store(I_P_kv_bhf + offs_dd, I_minus_P_kv.to(tl.bfloat16))
        tl.store(A_bhf + offs_dd, A_acc.to(tl.bfloat16))


@triton.jit
def _phase_a_z_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    beta_ptr,
    k_inv_rms_ptr,
    k_norm_w_ptr,
    I_minus_P_z_ptr,  # output: (I - K^T diag(β) K)
    B_ptr,  # output: K^T β
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    K_SCALE,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Z stream: uses K (no RoPE). Cheaper than KV — no V load, no RoPE, no K_pair."""
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    beta_bhf = beta_ptr + bh * (F * S) + pid_f * S
    I_P_z_bhf = I_minus_P_z_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    B_bhf = B_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    nw_offset = pid_h * D
    k_nw = tl.load(k_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )

    P_z_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    B_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    k_scale = K_SCALE
    n_base = pid_f * S

    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        k_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 1 * stride_3
            + offs_d[None, :] * stride_d
        )
        K_raw = tl.load(k_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        beta_t = tl.load(beta_bhf + offs_s, mask=mask_s, other=0.0).to(tl.float32)

        k_inv_rms = tl.load(
            k_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        K_normed = K_raw * k_inv_rms[:, None] * k_nw[None, :]
        K = tl.where(K_normed > 0, K_normed, 0.0) * k_scale

        beta_K = beta_t[:, None] * K

        K_T = tl.trans(K)
        P_z_acc += tl.dot(
            K_T.to(dot_dtype),
            beta_K.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        B_acc += tl.sum(beta_K, axis=0)

    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    diag_in_range = (
        (offs_d[:, None] == offs_d[None, :]) & mask_d[:, None] & mask_d[None, :]
    )
    I_minus_P_z = tl.where(diag_in_range, 1.0 - P_z_acc, -P_z_acc)

    if DOT_PRECISION >= 1:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z)
    else:
        tl.store(I_P_z_bhf + offs_dd, I_minus_P_z.to(tl.bfloat16))
    # B stays fp32 (vector, ~0.5 KB, negligible HBM cost).
    tl.store(B_bhf + offs_d, B_acc)


def phase_a(
    qkv: torch.Tensor,
    beta: torch.Tensor,
    q_inv_rms: torch.Tensor,
    k_inv_rms: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    F: int,
    S: int,
    k_scale: float = 1.0,
    norm_eps: float = 1e-5,
    num_warps: int | None = None,
    num_stages: int = 1,
    BLOCK_S: int | None = None,
    dot_precision: int = 0,
    skip_relu: bool = False,
    skip_z: bool = False,
):
    """Compute (I-P_kv), A, (I-P_z), B for all (B, H, F) via 2 kernels (KV + Z).

    `skip_relu=True`: pure linear K-stream prep (no ReLU). Used by the camera
    branch where K is already ReLU'd then rotated by UCPE+RoPE — re-applying ReLU
    on rotated values would clobber legitimate negatives.

    `skip_z=True`: skip the Z kernel, return placeholder I_P_z/B_z. Used by
    NUM_ONLY (camera) callers that never consume the denominator scan.
    """
    if num_warps is None or BLOCK_S is None:
        a_w, a_bs, *_ = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = a_w
        if BLOCK_S is None:
            BLOCK_S = a_bs
    B, N, three, H, D = qkv.shape
    assert three == 3 and N == F * S
    BLOCK_D = triton.next_power_of_2(D)
    BH = B * H

    # FAIR-COMPARE PATCH: keep fp32 inter-phase bridge at P0/P1 to match pytorch/fused
    bridge_dtype = torch.float32 if dot_precision >= 1 else torch.bfloat16
    I_P_kv = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    A = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)

    beta_c = beta.contiguous()
    grid = (BH * F,)

    _phase_a_kv_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        rope_cos,
        rope_sin,
        I_P_kv,
        A,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        SKIP_RELU=skip_relu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if skip_z:
        # NUM_ONLY (camera) callers don't consume the Z scan; placeholders let
        # Phase B skip all Z loads/stores too.
        I_P_z = torch.empty(1, device=qkv.device, dtype=bridge_dtype)
        B_z = torch.empty(1, device=qkv.device, dtype=torch.float32)
        return I_P_kv, A, I_P_z, B_z

    I_P_z = torch.empty(BH, F, BLOCK_D, BLOCK_D, device=qkv.device, dtype=bridge_dtype)
    # B stays fp32 — small vector (~0.5 KB/frame), no benefit to downcast.
    B_z = torch.empty(BH, F, BLOCK_D, device=qkv.device, dtype=torch.float32)

    _phase_a_z_kernel[grid](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        beta_c,
        k_inv_rms,
        k_norm_w,
        I_P_z,
        B_z,
        H=H,
        F=F,
        S=S,
        D=D,
        K_SCALE=k_scale,
        NORM_EPS=norm_eps,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return I_P_kv, A, I_P_z, B_z


# ════════════════════════════════════════════════════════════════
#  Phase B — serial scan, uses pre-stored (I - P) so MMA folds in M
# ════════════════════════════════════════════════════════════════


@triton.jit
def _phase_b_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — read when LOAD_INIT_STATE=1
    init_state_z_ptr,  # (BH, BLOCK_D)
    final_state_kv_ptr,  # (BH, BLOCK_D, BLOCK_D) — written when SAVE_FINAL_STATE=1
    final_state_z_ptr,  # (BH, BLOCK_D)
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,  # forward scan seeded with init state (vs zeros)
    SAVE_FINAL_STATE: tl.constexpr,  # write M_{F-1} of forward scan to final_state_*
    DIRECTION: tl.constexpr,  # 0=both, 1=fwd-only, 2=rev-only
    COMBINED_HISTORY: tl.constexpr,  # rev read-add-stores into M_fwd_ptr → M_hist[f]
    # = M_fwd[f] + M_rev[f]; skips the F-1 zero-write (rev value there is zero).
    # DIRECTION=0 only. Saves one Phase C launch + one M-shaped buffer downstream.
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    bh = pid

    offs_d = tl.arange(0, BLOCK_D)
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]

    # ── Forward scan (skip when DIRECTION=2 i.e. rev-only) ──
    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd).to(
                tl.float32
            )
            if not SKIP_Z:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d).to(tl.float32)
        else:
            M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
            if not SKIP_Z:
                z = tl.zeros([BLOCK_D], dtype=tl.float32)
        for f in range(F):
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd
            )
            A_f = tl.load(
                A_ptr + bh * F * BLOCK_D * BLOCK_D + f * BLOCK_D * BLOCK_D + offs_dd
            )
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            # M = g · (I - P_kv) M + A_f
            if USE_ACC_FUSION:
                # Pre-scale (I-P) by g and fold A_f via the MMA accumulator →
                # A_f + g·(I-P)·M in one MMA, no separate M_temp tensor.
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            tl.store(
                M_fwd_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd,
                M,
            )
            if not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d)
                # z = g · (I - P_z) z + B_f
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d, z)

        # Save terminal forward state for state-cached inference (autoregressive).
        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd, M)
            if not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d, z)

    # ── Reverse scan (skip when DIRECTION=1 i.e. fwd-only) ──
    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
        # COMBINED_HISTORY: skip the F-1 zero-write so M_hist[F-1] keeps the fwd
        # value (rev value there is zero by construction).
        if not COMBINED_HISTORY:
            tl.store(
                M_rev_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + (F - 1) * BLOCK_D * BLOCK_D
                + offs_dd,
                M,
            )
            if not SKIP_Z:
                tl.store(z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d, z)
        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd
            )
            A_f = tl.load(
                A_ptr + bh * F * BLOCK_D * BLOCK_D + f_src * BLOCK_D * BLOCK_D + offs_dd
            )
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            if not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_src * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                # Read-add-store the rev contribution into the fwd buffer slot
                # (fwd just wrote M_fwd[f_dst]; stays in L1/L2).
                M_addr = (
                    M_fwd_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd
                )
                tl.store(M_addr, tl.load(M_addr) + M)
                if not SKIP_Z:
                    z_addr = z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(
                    M_rev_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd,
                    M,
                )
                if not SKIP_Z:
                    tl.store(z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d, z)


def phase_b_triton(
    I_P_kv,
    A,
    I_P_z,
    B,
    decay,
    F,
    num_warps=None,
    num_stages=None,
    use_acc_fusion=None,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
    direction=0,
    combined_history=False,
    skip_z=False,
):
    """Phase B serial-F scan over (B*H,).

    Forward scan can be seeded with `init_state_kv`/`init_state_z` (autoregressive
    chunk > 0) and write terminal `M_{F-1}`/`z_{F-1}` when `return_final_state=True`.

    `direction`: 0=both, 1=forward-only, 2=reverse-only (single-direction state
    cache). `combined_history` (direction=0 only): rev read-add-stores into the
    fwd buffer so it becomes M_hist[f]=M_fwd[f]+M_rev[f], letting Phase C run once
    (it's linear: `Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`); M_rev/z_rev are
    then placeholders. `skip_z`: skip the Z/denominator recurrence (camera
    num-only scans, Phase C with num_only=True).

    Returns (M_fwd, z_fwd, M_rev, z_rev) (+ (final_kv, final_z) when
    return_final_state). Skipped-direction outputs are 1-element placeholders the
    kernel never touches — callers must discard the slot they didn't ask for.
    Reverse scan always seeds from zero (upstream bidi convention: only forward
    state is cached).
    """
    BH = I_P_kv.shape[0]
    _, _, BLOCK_D, _ = A.shape  # A is always full [BH, F, BLOCK_D, BLOCK_D]
    device, fdtype = I_P_kv.device, torch.float32

    if num_warps is None or num_stages is None or use_acc_fusion is None:
        _, _, b_w, b_s, b_acc, *_ = _get_arch_config(dot_precision, device=device)
        if num_warps is None:
            num_warps = b_w
        if num_stages is None:
            num_stages = b_s
        if use_acc_fusion is None:
            use_acc_fusion = b_acc

    if combined_history and direction != 0:
        raise ValueError("combined_history=True requires direction=0 (bidi)")

    # Kernel is DIRECTION-gated (constexpr), so inactive buffers can be 1-element
    # placeholders — frees ~4× M_fwd-shaped allocs per single-direction call.
    decay_flat = decay.reshape(BH, F).contiguous().float()

    load_init = init_state_kv is not None
    dummy = torch.empty(1, device=device, dtype=fdtype)
    full_M = lambda: torch.empty(BH, F, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
    full_z = lambda: torch.empty(BH, F, BLOCK_D, device=device, dtype=fdtype)
    M_fwd = dummy if direction == 2 else full_M()
    z_fwd = dummy if (direction == 2 or skip_z) else full_z()
    # Combined-history reuses M_fwd/z_fwd as M_hist/z_hist; rev outputs are
    # placeholders even though DIRECTION!=1.
    M_rev = dummy if (direction == 1 or combined_history) else full_M()
    z_rev = dummy if (direction == 1 or combined_history or skip_z) else full_z()
    if load_init:
        init_kv = init_state_kv.contiguous().view(BH, BLOCK_D, BLOCK_D)
        init_z = dummy if skip_z else init_state_z.contiguous().view(BH, BLOCK_D)
    else:
        init_kv = dummy
        init_z = dummy

    if return_final_state:
        final_kv = torch.empty(BH, BLOCK_D, BLOCK_D, device=device, dtype=fdtype)
        final_z = (
            dummy if skip_z else torch.empty(BH, BLOCK_D, device=device, dtype=fdtype)
        )
    else:
        final_kv = dummy
        final_z = dummy

    d_splits, nw_override, ns_override, acc_override = _pick_phase_b_d_splits(
        BLOCK_D, dot_precision=dot_precision
    )
    if d_splits > 1:
        D_TILE = BLOCK_D // d_splits
        # D-tile-specific tuning if available, else baseline.
        nw_use = nw_override if nw_override is not None else num_warps
        ns_use = ns_override if ns_override is not None else num_stages
        acc_use = acc_override if acc_override is not None else use_acc_fusion
        _phase_b_dtile_kernel[(BH, d_splits)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            D_TILE=D_TILE,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=acc_use,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=nw_use,
            num_stages=ns_use,
        )
    else:
        _phase_b_kernel[(BH,)](
            I_P_kv,
            A,
            I_P_z,
            B,
            decay_flat,
            M_fwd,
            z_fwd,
            M_rev,
            z_rev,
            init_kv,
            init_z,
            final_kv,
            final_z,
            BH=BH,
            F=F,
            BLOCK_D=BLOCK_D,
            DOT_PRECISION=dot_precision,
            USE_ACC_FUSION=use_acc_fusion,
            LOAD_INIT_STATE=1 if load_init else 0,
            SAVE_FINAL_STATE=1 if return_final_state else 0,
            DIRECTION=direction,
            COMBINED_HISTORY=1 if combined_history else 0,
            SKIP_Z=1 if skip_z else 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if return_final_state:
        return M_fwd, z_fwd, M_rev, z_rev, final_kv, final_z
    return M_fwd, z_fwd, M_rev, z_rev


# Phase B D-tile — j-axis split for grid parallelism (#118).
# Same recurrence as _phase_b_kernel but each program owns a D_TILE-wide slice of
# M's output column dim. Grid: (BH, d_splits). M_new[*, j_tile] depends only on
# M_prev[*, j_tile] and full (I-P_kv) — independent across j-tiles. z is
# unsplittable; only `pid_d == 0` updates/writes z.
@triton.jit
def _phase_b_dtile_kernel(
    I_P_kv_ptr,
    A_ptr,
    I_P_z_ptr,
    B_ptr,
    decay_ptr,
    M_fwd_ptr,
    z_fwd_ptr,
    M_rev_ptr,
    z_rev_ptr,
    init_state_kv_ptr,
    init_state_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    BH: tl.constexpr,
    F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_ACC_FUSION: tl.constexpr,
    LOAD_INIT_STATE: tl.constexpr,
    SAVE_FINAL_STATE: tl.constexpr,
    DIRECTION: tl.constexpr,
    COMBINED_HISTORY: tl.constexpr,
    SKIP_Z: tl.constexpr,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = pid_bh

    offs_d_full = tl.arange(0, BLOCK_D)
    offs_d_tile = pid_d * D_TILE + tl.arange(0, D_TILE)
    offs_dd_full = offs_d_full[:, None] * BLOCK_D + offs_d_full[None, :]
    offs_dd_tile = offs_d_full[:, None] * BLOCK_D + offs_d_tile[None, :]

    is_lead = pid_d == 0

    if DIRECTION != 2:
        if LOAD_INIT_STATE:
            M = tl.load(init_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile).to(
                tl.float32
            )
        else:
            M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)
            if is_lead and LOAD_INIT_STATE:
                z = tl.load(init_state_z_ptr + bh * BLOCK_D + offs_d_full).to(
                    tl.float32
                )

        for f in range(F):
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_full
            )
            A_f = tl.load(
                A_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_tile
            )
            g_f = tl.load(decay_ptr + bh * F + f).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            tl.store(
                M_fwd_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f * BLOCK_D * BLOCK_D
                + offs_dd_tile,
                M,
            )

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f * BLOCK_D * BLOCK_D
                    + offs_dd_full
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f
                tl.store(z_fwd_ptr + bh * F * BLOCK_D + f * BLOCK_D + offs_d_full, z)

        if SAVE_FINAL_STATE:
            tl.store(final_state_kv_ptr + bh * BLOCK_D * BLOCK_D + offs_dd_tile, M)
            if is_lead and not SKIP_Z:
                tl.store(final_state_z_ptr + bh * BLOCK_D + offs_d_full, z)

    if DIRECTION != 1:
        M = tl.zeros([BLOCK_D, D_TILE], dtype=tl.float32)
        if not SKIP_Z:
            z = tl.zeros([BLOCK_D], dtype=tl.float32)

        if not COMBINED_HISTORY:
            tl.store(
                M_rev_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + (F - 1) * BLOCK_D * BLOCK_D
                + offs_dd_tile,
                M,
            )
            if is_lead and not SKIP_Z:
                tl.store(
                    z_rev_ptr + bh * F * BLOCK_D + (F - 1) * BLOCK_D + offs_d_full, z
                )

        for f_iter in range(F - 1):
            f_src = F - 1 - f_iter
            f_dst = f_src - 1
            I_P_kv_f = tl.load(
                I_P_kv_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd_full
            )
            A_f = tl.load(
                A_ptr
                + bh * F * BLOCK_D * BLOCK_D
                + f_src * BLOCK_D * BLOCK_D
                + offs_dd_tile
            )
            g_f = tl.load(decay_ptr + bh * F + f_src).to(tl.float32)

            if USE_ACC_FUSION:
                I_P_scaled = I_P_kv_f.to(tl.float32) * g_f
                M = tl.dot(
                    I_P_scaled.to(dot_dtype),
                    M.to(dot_dtype),
                    acc=A_f.to(tl.float32),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
            else:
                M_temp = tl.dot(
                    I_P_kv_f.to(dot_dtype),
                    M.to(dot_dtype),
                    out_dtype=tl.float32,
                    input_precision=dot_ip,
                )
                M = g_f * M_temp + A_f

            if is_lead and not SKIP_Z:
                I_P_z_f = tl.load(
                    I_P_z_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_src * BLOCK_D * BLOCK_D
                    + offs_dd_full
                )
                B_f = tl.load(B_ptr + bh * F * BLOCK_D + f_src * BLOCK_D + offs_d_full)
                z_temp = tl.sum(I_P_z_f * z[None, :], axis=1)
                z = g_f * z_temp + B_f

            if COMBINED_HISTORY:
                M_addr = (
                    M_fwd_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd_tile
                )
                tl.store(M_addr, tl.load(M_addr) + M)
                if is_lead and not SKIP_Z:
                    z_addr = (
                        z_fwd_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full
                    )
                    tl.store(z_addr, tl.load(z_addr) + z)
            else:
                tl.store(
                    M_rev_ptr
                    + bh * F * BLOCK_D * BLOCK_D
                    + f_dst * BLOCK_D * BLOCK_D
                    + offs_dd_tile,
                    M,
                )
                if is_lead and not SKIP_Z:
                    tl.store(
                        z_rev_ptr + bh * F * BLOCK_D + f_dst * BLOCK_D + offs_d_full, z
                    )


_PHASE_B_DTILE_ARCH_CACHE: dict = {}  # (dev, dot_prec) -> (d_splits, nw, ns, acc)


def _pick_phase_b_d_splits(BLOCK_D: int, dot_precision: int = 0):
    """Returns (d_splits, nw_override, ns_override, acc_override).

    `d_splits=1` → baseline `_phase_b_kernel` with `_CHUNKWISE_TUNING` config.
    `d_splits>1` → `_phase_b_dtile_kernel` with nw/ns/acc overrides.
    Per-arch optima from a 96-config sweep (2026-04-29, T=11 B=1 IEEE) and a
    multi-arch bf16/TF32 sweep (2026-05-06, F=11 S=920); see inline cfg comments.
    Env overrides: PHASE_B_D_SPLITS, PHASE_B_DTILE_NW/NS/ACC (1=True / 0=False).
    """
    import os

    env_d = os.environ.get("PHASE_B_D_SPLITS", None)
    if env_d is not None:
        d = int(env_d)
        if d < 1 or BLOCK_D % d != 0:
            return (1, None, None, None)
        nw = int(os.environ.get("PHASE_B_DTILE_NW", "0")) or None
        ns = int(os.environ.get("PHASE_B_DTILE_NS", "0")) or None
        acc_env = os.environ.get("PHASE_B_DTILE_ACC", None)
        acc = bool(int(acc_env)) if acc_env is not None else None
        return (d, nw, ns, acc)
    try:
        import torch

        if not torch.cuda.is_available():
            return (1, None, None, None)
        dev = torch.cuda.current_device()
        cache_key = (dev, dot_precision)
        if cache_key not in _PHASE_B_DTILE_ARCH_CACHE:
            cap = torch.cuda.get_device_capability(dev)
            major, minor = cap[0], cap[1]
            if dot_precision == 2:
                # IEEE fp32: D-tile dominates baseline on every arch.
                if major == 8 and minor == 0:
                    cfg = (4, 32, 1, True)  # A100
                elif major == 9:
                    cfg = (4, 32, 1, True)  # H100 (Hopper)
                elif major == 8 and minor == 9:
                    cfg = (8, 4, 1, False)  # Ada (assume Blackwell-like)
                elif major >= 10:
                    cfg = (8, 4, 1, False)  # GB200/B200, 5090, GB10
                else:
                    cfg = (1, None, None, None)  # unknown — baseline
            else:
                # bf16/TF32: cap-specific dispatch (per-cap D-tile-vs-baseline win):
                #   sm_80 A100 / sm_90 H100 / sm_100 GB200: D-tile WIN ~10-12% — (4,8,2,F).
                #   sm_120 5090: D-tile WIN 2.6×(P1)/1.13×(P2) — (8,8,1,F); TF32 baseline OOMs.
                #   sm_121 GB10: baseline WINS (4% over D-tile). Despite matching sm_120 SRAM,
                #     the baseline fits all configs up to nw=16 ns=2 here (Triton/codegen
                #     difference between consumer-Blackwell variants) and saturates the chip.
                if major == 8 and minor == 0:
                    cfg = (4, 8, 2, False)  # A100
                elif major == 9:
                    cfg = (4, 8, 2, False)  # H100
                elif major == 10:
                    cfg = (4, 8, 2, False)  # GB200 / B200
                elif major == 12 and minor == 0:
                    cfg = (8, 8, 1, False)  # 5090
                elif major == 12 and minor == 1:
                    cfg = (1, None, None, None)  # GB10 — baseline wins
                else:
                    cfg = (1, None, None, None)  # Ada, unknown
            _PHASE_B_DTILE_ARCH_CACHE[cache_key] = cfg
        return _PHASE_B_DTILE_ARCH_CACHE[cache_key]
    except Exception:
        return (1, None, None, None)


# Phase C — Pass 2 output (per (B, H, F)). Same as v1.


@triton.jit
def _phase_c_kernel(
    qkv_ptr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    stride_3: tl.constexpr,
    stride_h: tl.constexpr,
    stride_d: tl.constexpr,
    q_inv_rms_ptr,
    q_norm_w_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    M_ptr,
    z_ptr,
    num_ptr,
    den_ptr,
    H: tl.constexpr,
    F: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ACCUMULATE: tl.constexpr = False,
    SKIP_LAST_F: tl.constexpr = False,
    SKIP_RELU: tl.constexpr = False,
    NUM_ONLY: tl.constexpr = False,
):
    if DOT_PRECISION >= 1:
        dot_dtype = tl.float32
    else:
        dot_dtype = tl.bfloat16
    dot_ip: tl.constexpr = "ieee" if DOT_PRECISION == 2 else "tf32"

    pid = tl.program_id(0)
    pid_b = pid // (H * F)
    pid_hf = pid % (H * F)
    pid_h = pid_hf // F
    pid_f = pid_hf % F
    bh = pid_b * H + pid_h
    N: tl.constexpr = F * S

    # SKIP_LAST_F (reverse-accumulate callers): M_rev[F-1]/z_rev[F-1] are exactly
    # zero (rev scan inits to zero, write loop fills only f<F-1), so the f=F-1
    # program would re-write forward's output unchanged. Early-return saves one
    # frame's Q+RoPE HBM reads + dots per (B, H).
    if SKIP_LAST_F and pid_f == F - 1:
        return

    qkv_bh = qkv_ptr + pid_b * stride_b + pid_h * stride_h
    num_bh = num_ptr + pid_b * (N * H * D) + pid_h * D
    den_bh = den_ptr + bh * N
    M_bhf = M_ptr + bh * F * BLOCK_D * BLOCK_D + pid_f * BLOCK_D * BLOCK_D
    z_bhf = z_ptr + bh * F * BLOCK_D + pid_f * BLOCK_D

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_d_pair = offs_d ^ 1
    mask_d_pair = offs_d_pair < D
    offs_dd = offs_d[:, None] * BLOCK_D + offs_d[None, :]
    mask_dd = mask_d[:, None] & mask_d[None, :]

    nw_offset = pid_h * D
    q_nw = tl.load(q_norm_w_ptr + nw_offset + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    q_nw_pair = tl.load(
        q_norm_w_ptr + nw_offset + offs_d_pair, mask=mask_d_pair, other=0.0
    ).to(tl.float32)

    M_f = tl.load(M_bhf + offs_dd, mask=mask_dd, other=0.0)
    z_f = tl.load(z_bhf + offs_d, mask=mask_d, other=0.0)

    n_base = pid_f * S
    for s0 in range(0, S, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        mask_sd = mask_s[:, None] & mask_d[None, :]
        n_idx = n_base + offs_s

        q_ptrs = (
            qkv_bh
            + n_idx[:, None] * stride_n
            + 0 * stride_3
            + offs_d[None, :] * stride_d
        )
        Q_raw = tl.load(q_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_pair_raw = tl.reshape(
            tl.flip(tl.reshape(Q_raw, (BLOCK_S, BLOCK_D // 2, 2)), dim=2),
            (BLOCK_S, BLOCK_D),
        )

        q_inv_rms = tl.load(
            q_inv_rms_ptr + pid_b * N + n_idx, mask=mask_s, other=1.0
        ).to(tl.float32)
        Q_normed = Q_raw * q_inv_rms[:, None] * q_nw[None, :]
        Q_pair_normed = Q_pair_raw * q_inv_rms[:, None] * q_nw_pair[None, :]
        if SKIP_RELU:
            Q = Q_normed
            Q_pair = Q_pair_normed
        else:
            Q = tl.where(Q_normed > 0, Q_normed, 0.0)
            Q_pair = tl.where(Q_pair_normed > 0, Q_pair_normed, 0.0)

        rope_ptrs = n_idx[:, None] * D + offs_d[None, :]
        Cos = tl.load(rope_cos_ptr + rope_ptrs, mask=mask_sd, other=1.0).to(tl.float32)
        Sin = tl.load(rope_sin_ptr + rope_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
        Q_rot = Q * Cos + Q_pair * Sin

        num = tl.dot(
            Q_rot.to(dot_dtype),
            M_f.to(dot_dtype),
            out_dtype=tl.float32,
            input_precision=dot_ip,
        )
        if not NUM_ONLY:
            den = tl.sum(Q * z_f[None, :], axis=1)

        num_ptrs = num_bh + n_idx[:, None] * (H * D) + offs_d[None, :]
        if not NUM_ONLY:
            den_ptrs = den_bh + n_idx
        if ACCUMULATE:
            # Reverse-direction Phase C: add onto forward's already-written buffer
            # instead of allocating a separate one.
            prev_num = tl.load(num_ptrs, mask=mask_sd, other=0.0).to(tl.float32)
            num = num + prev_num
            if not NUM_ONLY:
                prev_den = tl.load(den_ptrs, mask=mask_s, other=0.0).to(tl.float32)
                den = den + prev_den
        if DOT_PRECISION >= 1:
            tl.store(num_ptrs, num, mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den, mask=mask_s)
        else:
            tl.store(num_ptrs, num.to(tl.bfloat16), mask=mask_sd)
            if not NUM_ONLY:
                tl.store(den_ptrs, den.to(tl.bfloat16), mask=mask_s)


def phase_c(
    qkv,
    q_inv_rms,
    q_norm_w,
    rope_cos,
    rope_sin,
    M,
    z,
    F,
    S,
    num_warps=None,
    num_stages=None,
    BLOCK_S=None,
    dot_precision=0,
    num_out=None,
    den_out=None,
    accumulate=False,
    skip_last_frame=False,
    skip_relu: bool = False,
    num_only: bool = False,
):
    """Phase C Pass-2 output.

    ``accumulate=True``: add into caller's ``num_out``/``den_out`` (fuses
    reverse-direction output into the forward buffer — saves ~45 MB at B=1 bf16).
    ``skip_last_frame=True``: early-return f=F-1 programs; valid only for the
    reverse-accumulate call, where M[F-1]/z[F-1] are guaranteed zero.
    ``skip_relu=True``: as in Phase A KV — Q is already ReLU'd then rotated by
    UCPE+RoPE; re-applying ReLU on rotated Q would clobber legitimate negatives.
    ``num_only=True``: skip denominator entirely (writes only ``num_out``;
    ``den_out`` may be None). Camera branch has no Z scan.
    """
    if num_warps is None or num_stages is None or BLOCK_S is None:
        *_, c_w, c_bs, c_s = _get_arch_config(dot_precision, device=qkv.device)
        if num_warps is None:
            num_warps = c_w
        if num_stages is None:
            num_stages = c_s
        if BLOCK_S is None:
            BLOCK_S = c_bs
    B, N, three, H, D = qkv.shape
    BLOCK_D = triton.next_power_of_2(D)
    if num_out is None:
        num_out = torch.empty(
            B,
            N,
            H,
            D,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )
    if den_out is None and not num_only:
        den_out = torch.empty(
            B,
            H,
            N,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )
    elif num_only and den_out is None:
        # 1-element placeholder; kernel guards den loads/stores under NUM_ONLY.
        den_out = torch.empty(
            1,
            device=qkv.device,
            dtype=(torch.float32 if dot_precision >= 1 else torch.bfloat16),
        )

    _phase_c_kernel[(B * H * F,)](
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        qkv.stride(3),
        qkv.stride(4),
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M,
        z,
        num_out,
        den_out,
        H=H,
        F=F,
        S=S,
        D=D,
        NORM_EPS=1e-5,
        DOT_PRECISION=dot_precision,
        BLOCK_D=BLOCK_D,
        BLOCK_S=BLOCK_S,
        ACCUMULATE=1 if accumulate else 0,
        SKIP_LAST_F=skip_last_frame,
        SKIP_RELU=skip_relu,
        NUM_ONLY=num_only,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return num_out, den_out


def fused_bigdn_bidi_chunkwise(
    qkv,
    q_inv_rms,
    k_inv_rms,
    q_norm_w,
    k_norm_w,
    rope_cos,
    rope_sin,
    beta,
    decay,
    F,
    S,
    k_scale=1.0,
    eps=1e-6,
    norm_eps=1e-5,
    dot_precision=0,
    init_state_kv=None,
    init_state_z=None,
    return_final_state=False,
):
    """Bidi chunkwise GDN forward, optionally state-cached for autoregressive
    sampling (chunk 0 = full bidi + state save; chunks > 0 seed forward from saved
    state). Reverse always seeds from zero (upstream convention).

    Pipeline: Phase A once → Phase B direction=0, combined_history=True (rev summed
    into fwd buffer in-kernel so M_hist[f]=M_fwd[f]+M_rev[f]) → Phase C ONCE on
    M_hist. Exact via Phase C linearity `Q @ (M_fwd + M_rev) = Q @ M_fwd + Q @ M_rev`.
    Replaces the prior 2× Phase B + 2× Phase C pattern.
    """
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        q_inv_rms,
        k_inv_rms,
        q_norm_w,
        k_norm_w,
        rope_cos,
        rope_sin,
        F=F,
        S=S,
        k_scale=k_scale,
        norm_eps=norm_eps,
        dot_precision=dot_precision,
    )

    if return_final_state:
        M_hist, z_hist, _, _, final_kv, final_z = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            return_final_state=True,
            combined_history=True,
        )
    else:
        M_hist, z_hist, _, _ = phase_b_triton(
            I_P_kv,
            A,
            I_P_z,
            B_z,
            decay,
            F=F,
            dot_precision=dot_precision,
            direction=0,
            init_state_kv=init_state_kv,
            init_state_z=init_state_z,
            combined_history=True,
        )
    num_out, den_out = phase_c(
        qkv,
        q_inv_rms,
        q_norm_w,
        rope_cos,
        rope_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        accumulate=False,
    )
    del M_hist, z_hist, I_P_kv, A, I_P_z, B_z

    total_den = den_out.float().permute(0, 2, 1).unsqueeze(-1)  # (B, N, H, 1)
    out = (num_out.float() / (total_den + eps)).to(qkv.dtype)
    del num_out, den_out, total_den
    if return_final_state:
        B = qkv.shape[0]
        H = qkv.shape[3]
        D = qkv.shape[4]
        BLOCK_D = final_kv.shape[1]
        state_kv = (
            final_kv.view(B, H, BLOCK_D, BLOCK_D)[:, :, :D, :D]
            .transpose(-1, -2)
            .contiguous()
        )
        state_z = final_z.view(B, H, BLOCK_D)[:, :, :D].unsqueeze(-1).contiguous()
        return out, state_kv, state_z
    return out


def _default_dot_prec() -> int:
    try:
        from sglang.jit_kernel.diffusion.triton.sana_wm_gdn import (
            _resolve_launch_config,
        )

        _, dot_prec, _, _ = _resolve_launch_config()
        return dot_prec
    except Exception:
        return 0


def _cam_identity_tables(
    *,
    B: int,
    N: int,
    H: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = device.index if device.type == "cuda" else None
    key = (device.type, device_index, B, N, H * D, D)
    cached = _CAM_IDENTITY_CACHE.get(key)
    if cached is not None:
        return cached

    ones_inv_rms = torch.ones(B, N, device=device, dtype=torch.float32)
    ones_nw = torch.ones(H * D, device=device, dtype=torch.float32)
    ones_cos = torch.ones(N, D, device=device, dtype=torch.float32)
    zeros_sin = torch.zeros(N, D, device=device, dtype=torch.float32)
    cached = (ones_inv_rms, ones_nw, ones_cos, zeros_sin)
    _CAM_IDENTITY_CACHE[key] = cached
    return cached


def cam_scan_bidi_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    dot_precision: int | None = None,
) -> torch.Tensor:
    """Bidirectional camera scan for SANA-WM's numerator-only branch.

    q, k, v: camera-prepared ``(B, H, D, N)`` fp32; beta: ``(B, H, F, S)`` fp32;
    decay: ``(B, H, F)`` fp32. Returns ``(B, H, D, N)`` fp32.
    """
    assert (
        q.shape == k.shape == v.shape
    ), f"q/k/v shape mismatch: {q.shape} {k.shape} {v.shape}"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert beta.is_contiguous() and decay.is_contiguous()
    assert (
        q.dtype == torch.float32
    ), f"cam_scan_bidi_chunkwise requires fp32 q/k/v, got {q.dtype}"

    B, H, D, N = q.shape
    F = beta.shape[2]
    assert N % F == 0
    S = N // F
    assert beta.shape == (B, H, F, S)
    assert decay.shape == (B, H, F)

    if dot_precision is None:
        dot_precision = _default_dot_prec()

    qkv = torch.empty(B, N, 3, H, D, device=q.device, dtype=q.dtype)
    qkv[:, :, 0].copy_(q.permute(0, 3, 1, 2))
    qkv[:, :, 1].copy_(k.permute(0, 3, 1, 2))
    qkv[:, :, 2].copy_(v.permute(0, 3, 1, 2))

    ones_inv_rms, ones_nw, ones_cos, zeros_sin = _cam_identity_tables(
        B=B, N=N, H=H, D=D, device=q.device
    )
    I_P_kv, A, I_P_z, B_z = phase_a(
        qkv,
        beta,
        ones_inv_rms,
        ones_inv_rms,
        ones_nw,
        ones_nw,
        ones_cos,
        zeros_sin,
        F=F,
        S=S,
        k_scale=1.0,
        norm_eps=1e-5,
        dot_precision=dot_precision,
        skip_relu=True,
        skip_z=True,
    )
    M_hist, z_hist, _, _ = phase_b_triton(
        I_P_kv,
        A,
        I_P_z,
        B_z,
        decay,
        F=F,
        dot_precision=dot_precision,
        direction=0,
        combined_history=True,
        skip_z=True,
    )
    num_out, _ = phase_c(
        qkv,
        ones_inv_rms,
        ones_nw,
        ones_cos,
        zeros_sin,
        M_hist,
        z_hist,
        F=F,
        S=S,
        dot_precision=dot_precision,
        skip_relu=True,
        num_only=True,
    )
    return num_out.permute(0, 2, 3, 1).contiguous().to(torch.float32)
