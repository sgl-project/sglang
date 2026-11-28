# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU Triton-Ascend fused DiT kernels for MiniMax-H3.

All kernels are guarded by runtime detection of ``triton`` with the Ascend
backend and fall back cleanly when unavailable.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn_triton import (
    _HAS_TRITON,
    _require_num_vectorcores,
    can_use_npu_triton,
)

if _HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_indexed_scale_shift_kernel(
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        out_ptr,
        rows,
        stride_x_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        HIDDEN_SIZE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        lane = tl.arange(0, HIDDEN_SIZE)
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        w = tl.load(weight_ptr + lane).to(tl.float32)

        for row_id in range(pid, rows, num_progs):
            idx = tl.load(indices_ptr + row_id * stride_indices)
            x = tl.load(x_ptr + row_id * stride_x_row + lane).to(tl.float32)
            mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
            normed = x * tl.rsqrt(mean_sq + EPS) * w
            sc = tl.load(scale_ptr + idx * stride_scale_row + lane).to(tl.float32)
            sh = tl.load(shift_ptr + idx * stride_shift_row + lane).to(tl.float32)
            tl.store(
                out_ptr + row_id * HIDDEN_SIZE + lane,
                (normed * (1.0 + sc) + sh).to(tl.bfloat16),
            )

    @triton.jit
    def _indexed_gate_rms_norm_scale_shift_kernel(
        res_ptr,
        branch_ptr,
        gate_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        indices_ptr,
        res_out_ptr,
        mod_out_ptr,
        rows,
        stride_res_row,
        stride_res_out_row,
        stride_branch_row,
        stride_gate_row,
        stride_shift_row,
        stride_scale_row,
        stride_indices,
        HIDDEN_SIZE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        lane = tl.arange(0, HIDDEN_SIZE)
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        w = tl.load(weight_ptr + lane).to(tl.float32)

        for row_id in range(pid, rows, num_progs):
            idx = tl.load(indices_ptr + row_id * stride_indices)
            r = tl.load(res_ptr + row_id * stride_res_row + lane).to(tl.float32)
            b = tl.load(branch_ptr + row_id * stride_branch_row + lane).to(tl.float32)
            g = tl.load(gate_ptr + idx * stride_gate_row + lane).to(tl.float32)
            updated = r + g * b
            tl.store(
                res_out_ptr + row_id * stride_res_out_row + lane,
                updated.to(tl.bfloat16),
            )

            mean_sq = tl.sum(updated * updated, axis=0) / HIDDEN_SIZE
            normed = updated * tl.rsqrt(mean_sq + EPS) * w
            sc = tl.load(scale_ptr + idx * stride_scale_row + lane).to(tl.float32)
            sh = tl.load(shift_ptr + idx * stride_shift_row + lane).to(tl.float32)
            mod = normed * (1.0 + sc) + sh
            tl.store(mod_out_ptr + row_id * HIDDEN_SIZE + lane, mod.to(tl.bfloat16))

    @triton.jit
    def _split_qkv_norm_rope_pack_2d_kernel(
        qkv_ptr,
        qw_ptr,
        kw_ptr,
        cos_sin_ptr,
        positions_ptr,
        packed_ptr,
        num_tokens,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        ROPE_HALF: tl.constexpr,
        TAIL_DIM: tl.constexpr,
        WS: tl.constexpr,
        GROUPS: tl.constexpr,
        GH: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        wg = WS * GROUPS
        total_tasks = num_tokens * wg

        gh_local = tl.arange(0, GH)
        half_lane = tl.arange(0, ROPE_HALF)
        second_lane = ROPE_HALF + half_lane
        tail_lane = tl.arange(0, TAIL_DIM)
        full_lane = tl.arange(0, HEAD_DIM)

        qw1 = tl.load(qw_ptr + half_lane).to(tl.float32)
        qw2 = tl.load(qw_ptr + second_lane).to(tl.float32)
        qwt = tl.load(qw_ptr + ROPE_DIM + tail_lane).to(tl.float32)
        kw1 = tl.load(kw_ptr + half_lane).to(tl.float32)
        kw2 = tl.load(kw_ptr + second_lane).to(tl.float32)
        kwt = tl.load(kw_ptr + ROPE_DIM + tail_lane).to(tl.float32)

        q_stride = NUM_HEADS * HEAD_DIM
        row_width = 3 * q_stride
        pack_gh_stride = 3 * HEAD_DIM
        pack_t_stride = GH * pack_gh_stride
        pack_ws_stride = num_tokens * pack_t_stride
        pack_g_stride = WS * pack_ws_stride

        for task in range(pid, total_tasks, num_progs):
            t = task // wg
            wg_idx = task % wg
            w_idx = wg_idx // GROUPS
            g_idx = wg_idx % GROUPS
            heads = wg_idx * GH + gh_local

            pos = tl.load(positions_ptr + t)
            cs_base = pos * ROPE_DIM
            cos = tl.load(cos_sin_ptr + cs_base + half_lane).to(tl.float32)
            sin = tl.load(cos_sin_ptr + cs_base + ROPE_HALF + half_lane).to(tl.float32)

            q_base = t * row_width + heads * HEAD_DIM
            k_base = q_base + q_stride
            v_base = k_base + q_stride
            out_base = (
                g_idx * pack_g_stride
                + w_idx * pack_ws_stride
                + t * pack_t_stride
                + gh_local * pack_gh_stride
            )

            q1 = tl.load(qkv_ptr + q_base[:, None] + half_lane[None, :]).to(tl.float32)
            q2 = tl.load(qkv_ptr + q_base[:, None] + second_lane[None, :]).to(
                tl.float32
            )
            qt = tl.load(qkv_ptr + q_base[:, None] + ROPE_DIM + tail_lane[None, :]).to(
                tl.float32
            )
            q_rstd = tl.rsqrt(
                (
                    tl.sum(q1 * q1, axis=1)
                    + tl.sum(q2 * q2, axis=1)
                    + tl.sum(qt * qt, axis=1)
                )
                / HEAD_DIM
                + EPS
            )
            qn1 = (q1 * q_rstd[:, None] * qw1[None, :]).to(tl.bfloat16).to(tl.float32)
            qn2 = (q2 * q_rstd[:, None] * qw2[None, :]).to(tl.bfloat16).to(tl.float32)
            tl.store(
                packed_ptr + out_base[:, None] + half_lane[None, :],
                (qn1 * cos[None, :] - qn2 * sin[None, :]).to(tl.bfloat16),
            )
            tl.store(
                packed_ptr + out_base[:, None] + second_lane[None, :],
                (qn2 * cos[None, :] + qn1 * sin[None, :]).to(tl.bfloat16),
            )
            tl.store(
                packed_ptr + out_base[:, None] + ROPE_DIM + tail_lane[None, :],
                (qt * q_rstd[:, None] * qwt[None, :]).to(tl.bfloat16),
            )

            k1 = tl.load(qkv_ptr + k_base[:, None] + half_lane[None, :]).to(tl.float32)
            k2 = tl.load(qkv_ptr + k_base[:, None] + second_lane[None, :]).to(
                tl.float32
            )
            kt = tl.load(qkv_ptr + k_base[:, None] + ROPE_DIM + tail_lane[None, :]).to(
                tl.float32
            )
            k_rstd = tl.rsqrt(
                (
                    tl.sum(k1 * k1, axis=1)
                    + tl.sum(k2 * k2, axis=1)
                    + tl.sum(kt * kt, axis=1)
                )
                / HEAD_DIM
                + EPS
            )
            kn1 = (k1 * k_rstd[:, None] * kw1[None, :]).to(tl.bfloat16).to(tl.float32)
            kn2 = (k2 * k_rstd[:, None] * kw2[None, :]).to(tl.bfloat16).to(tl.float32)
            tl.store(
                packed_ptr + out_base[:, None] + HEAD_DIM + half_lane[None, :],
                (kn1 * cos[None, :] - kn2 * sin[None, :]).to(tl.bfloat16),
            )
            tl.store(
                packed_ptr + out_base[:, None] + HEAD_DIM + second_lane[None, :],
                (kn2 * cos[None, :] + kn1 * sin[None, :]).to(tl.bfloat16),
            )
            tl.store(
                packed_ptr
                + out_base[:, None]
                + HEAD_DIM
                + ROPE_DIM
                + tail_lane[None, :],
                (kt * k_rstd[:, None] * kwt[None, :]).to(tl.bfloat16),
            )

            v_val = tl.load(qkv_ptr + v_base[:, None] + full_lane[None, :])
            tl.store(
                packed_ptr + out_base[:, None] + 2 * HEAD_DIM + full_lane[None, :],
                v_val,
            )


# Whole-row FP32 fusion keeps ~5-6 live vectors in Ascend's 192 KB Vector Core UB;
# 7168 stays safely below UB overflow while covering MiniMax-H3's 5376 hidden size.
_MAX_NPU_TRITON_HIDDEN_SIZE = 7168


def can_use_npu_triton_rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Return True when inputs satisfy ``npu_triton_rms_norm_indexed_scale_shift``."""
    return (
        x.device.type == "npu"
        and x.dtype == shift.dtype == scale.dtype == torch.bfloat16
        and x.stride(-1) == shift.stride(-1) == scale.stride(-1) == 1
        and x.shape[-1] <= _MAX_NPU_TRITON_HIDDEN_SIZE
        and weight is not None
        and weight.is_contiguous()
        and weight.device == x.device
        and can_use_npu_triton(x.device)
    )


def npu_triton_rms_norm_indexed_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + indexed ``(1 + scale) * norm + shift`` on Ascend NPU."""
    if x.dtype != torch.bfloat16:
        raise ValueError("npu_triton_rms_norm_indexed_scale_shift requires bfloat16 x")
    if not (
        x.stride(-1) == 1
        and weight.is_contiguous()
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
    ):
        raise ValueError(
            "npu_triton_rms_norm_indexed_scale_shift requires contiguous inner dim"
        )
    rows, hidden = x.shape
    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    if rows == 0:
        return out
    num_cores = _require_num_vectorcores(x.device)
    _rms_norm_indexed_scale_shift_kernel[(min(num_cores, rows),)](
        x,
        weight,
        shift,
        scale,
        indices,
        out,
        rows,
        x.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        HIDDEN_SIZE=hidden,
        EPS=eps,
    )
    return out


def can_use_npu_triton_indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    """Return True when inputs satisfy ``npu_triton_indexed_gate_rms_norm_scale_shift``."""
    return (
        residual.device.type == "npu"
        and residual.dtype
        == branch.dtype
        == gate.dtype
        == shift.dtype
        == scale.dtype
        == torch.bfloat16
        and residual.stride(-1)
        == branch.stride(-1)
        == gate.stride(-1)
        == shift.stride(-1)
        == scale.stride(-1)
        == 1
        and residual.shape[-1] <= _MAX_NPU_TRITON_HIDDEN_SIZE
        and weight is not None
        and weight.is_contiguous()
        and weight.device == residual.device
        and can_use_npu_triton(residual.device)
    )


def npu_triton_indexed_gate_rms_norm_scale_shift(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    *,
    inplace_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``res = res + gate[idx] * branch`` + RMSNorm + indexed affine."""
    if residual.dtype != torch.bfloat16:
        raise ValueError(
            "npu_triton_indexed_gate_rms_norm_scale_shift requires bfloat16 residual"
        )
    if not (
        residual.stride(-1) == 1
        and branch.stride(-1) == 1
        and gate.stride(-1) == 1
        and weight.is_contiguous()
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
    ):
        raise ValueError(
            "npu_triton_indexed_gate_rms_norm_scale_shift requires contiguous inner dim"
        )
    rows, hidden = residual.shape
    res_out = (
        residual
        if inplace_residual
        else torch.empty((rows, hidden), device=residual.device, dtype=residual.dtype)
    )
    mod_out = torch.empty((rows, hidden), device=residual.device, dtype=residual.dtype)
    if rows == 0:
        return res_out, mod_out
    num_cores = _require_num_vectorcores(residual.device)
    _indexed_gate_rms_norm_scale_shift_kernel[(min(num_cores, rows),)](
        residual,
        branch,
        gate,
        weight,
        shift,
        scale,
        indices,
        res_out,
        mod_out,
        rows,
        residual.stride(0),
        res_out.stride(0),
        branch.stride(0),
        gate.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        HIDDEN_SIZE=hidden,
        EPS=eps,
    )
    return res_out, mod_out


def can_use_npu_triton_split_qkv_norm_rope_pack(
    qkv: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
) -> bool:
    """Return True when inputs satisfy ``npu_triton_split_qkv_norm_rope_pack``."""
    return (
        qkv.device.type == "npu"
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and head_dim == 128
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[-1] == 96
        and cos_sin_cache.is_contiguous()
        and cos_sin_cache.device == qkv.device
        and positions.ndim == 1
        and positions.shape[0] == qkv.shape[0]
        and positions.is_contiguous()
        and positions.device == qkv.device
        and q_weight is not None
        and k_weight is not None
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and can_use_npu_triton(qkv.device)
    )


def npu_triton_split_qkv_norm_rope_pack(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    world_size: int,
    groups: int,
    eps: float,
) -> torch.Tensor:
    """Fused QKV split + Q/K RMSNorm + partial NeoX RoPE -> Ulysses packed buf."""
    if qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        raise ValueError(
            "npu_triton_split_qkv_norm_rope_pack requires contiguous bfloat16 qkv"
        )
    if not (
        q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.is_contiguous()
        and positions.ndim == 1
        and positions.is_contiguous()
        and positions.shape[0] == qkv.shape[0]
    ):
        raise ValueError(
            "npu_triton_split_qkv_norm_rope_pack requires contiguous weights, 2D cache, and 1D positions"
        )
    num_tokens = qkv.shape[0]
    gh = num_heads // (world_size * groups)
    packed = torch.empty(
        (groups, world_size, num_tokens, gh, 3 * head_dim),
        device=qkv.device,
        dtype=qkv.dtype,
    )
    if num_tokens == 0:
        return packed
    num_cores = _require_num_vectorcores(qkv.device)
    total_tasks = num_tokens * world_size * groups
    _split_qkv_norm_rope_pack_2d_kernel[(min(num_cores, total_tasks),)](
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        packed,
        num_tokens,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        ROPE_HALF=rope_dim // 2,
        TAIL_DIM=head_dim - rope_dim,
        WS=world_size,
        GROUPS=groups,
        GH=gh,
        EPS=eps,
    )
    return packed


__all__ = [
    "can_use_npu_triton",
    "can_use_npu_triton_indexed_gate_rms_norm_scale_shift",
    "can_use_npu_triton_rms_norm_indexed_scale_shift",
    "can_use_npu_triton_split_qkv_norm_rope_pack",
    "npu_triton_indexed_gate_rms_norm_scale_shift",
    "npu_triton_rms_norm_indexed_scale_shift",
    "npu_triton_split_qkv_norm_rope_pack",
]
