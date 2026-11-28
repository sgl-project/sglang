# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU Triton-Ascend fused pre/post-processing kernels for LaserAttention.

All kernels are guarded by runtime detection of ``triton`` with the Ascend
backend and fall back cleanly when unavailable.
"""

from __future__ import annotations

from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
    from triton.runtime import driver as _triton_driver

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - non-Triton environments
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _triton_driver = None
    _HAS_TRITON = False


@lru_cache(maxsize=8)
def _get_num_vectorcores(device_index: int) -> int:
    if not _HAS_TRITON or _triton_driver is None:
        return 0
    try:
        props = _triton_driver.active.utils.get_device_properties(device_index)
        return max(0, int(props.get("num_vectorcore", 0)))
    except Exception:
        return 0


def _require_num_vectorcores(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.npu.current_device()
    num_cores = _get_num_vectorcores(idx)
    if num_cores <= 0:
        raise RuntimeError(
            f"Triton-Ascend vector cores are unavailable on device {device}."
        )
    return num_cores


def can_use_npu_triton(device: torch.device | None = None) -> bool:
    """Return True when Triton-Ascend can launch vector kernels on ``device``."""
    if not _HAS_TRITON or not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False
    if device is not None and device.type != "npu":
        return False
    idx = (
        device.index
        if device is not None and device.index is not None
        else torch.npu.current_device()
    )
    return _get_num_vectorcores(idx) > 0


if _HAS_TRITON:

    @triton.jit
    def _la_preprocess_scaled_qkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        s_real,
        s_pad,
        stride_qs,
        stride_qh,
        stride_ks,
        stride_kh,
        stride_vs,
        stride_vh,
        INV_SCALE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        total_s_tiles = tl.cdiv(s_pad, BLOCK_S)
        total_tiles = NUM_HEADS * total_s_tiles
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        d_offs = tl.arange(0, HEAD_DIM)
        s_local = tl.arange(0, BLOCK_S)

        for tile_id in range(pid, total_tiles, num_progs):
            h = tile_id // total_s_tiles
            s_tile = tile_id % total_s_tiles
            s_offs = s_tile * BLOCK_S + s_local
            in_pad = s_offs < s_pad
            in_real = s_offs < s_real
            out_base = (h * s_pad + s_offs)[:, None] * HEAD_DIM + d_offs[None, :]
            out_mask = in_pad[:, None]
            real_mask = in_real[:, None]

            q_in_offs = s_offs[:, None] * stride_qs + h * stride_qh + d_offs[None, :]
            k_in_offs = s_offs[:, None] * stride_ks + h * stride_kh + d_offs[None, :]
            v_in_offs = s_offs[:, None] * stride_vs + h * stride_vh + d_offs[None, :]

            q_val = (
                tl.load(q_ptr + q_in_offs, mask=real_mask, other=0.0).to(tl.float32)
                * INV_SCALE
            )
            tl.store(q_out_ptr + out_base, q_val.to(tl.float16), mask=out_mask)

            k_val = (
                tl.load(k_ptr + k_in_offs, mask=real_mask, other=0.0).to(tl.float32)
                * INV_SCALE
            )
            tl.store(k_out_ptr + out_base, k_val.to(tl.float16), mask=out_mask)

            v_val = (
                tl.load(v_ptr + v_in_offs, mask=real_mask, other=0.0).to(tl.float32)
                * INV_SCALE
            )
            tl.store(v_out_ptr + out_base, v_val.to(tl.float16), mask=out_mask)

    @triton.jit
    def _la_postprocess_scaled_kernel(
        la_ptr,
        out_ptr,
        s_real,
        s_pad,
        s_total,
        SCALE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        total_s_tiles = tl.cdiv(s_total, BLOCK_S)
        total_tiles = NUM_HEADS * total_s_tiles
        pid = tl.program_id(0)
        num_progs = tl.num_programs(0)
        d_offs = tl.arange(0, HEAD_DIM)
        s_local = tl.arange(0, BLOCK_S)

        for tile_id in range(pid, total_tiles, num_progs):
            h = tile_id // total_s_tiles
            s_tile = tile_id % total_s_tiles
            s_offs = s_tile * BLOCK_S + s_local
            in_total = s_offs < s_total
            in_real = s_offs < s_real
            in_offs = (h * s_pad + s_offs)[:, None] * HEAD_DIM + d_offs[None, :]
            out_offs = (s_offs * NUM_HEADS + h)[:, None] * HEAD_DIM + d_offs[None, :]
            val = (
                tl.load(la_ptr + in_offs, mask=in_real[:, None], other=0.0).to(
                    tl.float32
                )
                * SCALE
            )
            tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=in_total[:, None])


def npu_triton_la_preprocess_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    s_real: int,
    s_pad: int,
    inv_scale: float,
    block_s: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused BSND BF16 -> scaled BNSD FP16 with zero-padding to ``s_pad``."""
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise ValueError("npu_triton_la_preprocess_qkv requires bfloat16 Q/K/V")
    if not (q.stride(-1) == k.stride(-1) == v.stride(-1) == 1):
        raise ValueError("npu_triton_la_preprocess_qkv requires contiguous head dim")
    _, h, d = q.shape
    q_out = torch.empty((1, h, s_pad, d), device=q.device, dtype=torch.float16)
    k_out = torch.empty((1, h, s_pad, d), device=q.device, dtype=torch.float16)
    v_out = torch.empty((1, h, s_pad, d), device=q.device, dtype=torch.float16)
    if s_pad == 0 or h == 0:
        return q_out, k_out, v_out
    num_cores = _require_num_vectorcores(q.device)
    total_tiles = h * ((s_pad + block_s - 1) // block_s)
    _la_preprocess_scaled_qkv_kernel[(min(num_cores, total_tiles),)](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        s_real,
        s_pad,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        INV_SCALE=inv_scale,
        NUM_HEADS=h,
        HEAD_DIM=d,
        BLOCK_S=block_s,
    )
    return q_out, k_out, v_out


def npu_triton_la_postprocess(
    la_out: torch.Tensor,
    *,
    s_real: int,
    s_total: int,
    scale: float,
    block_s: int = 64,
) -> torch.Tensor:
    """Fused BNSD FP16/FP32 -> scaled BSND BF16 with zeroed ``[s_real:s_total]``."""
    if la_out.dtype not in (torch.float16, torch.float32) or not la_out.is_contiguous():
        raise ValueError(
            "npu_triton_la_postprocess requires contiguous float16/float32 la_out"
        )
    _, h, s_pad, d = la_out.shape
    out = torch.empty((s_total, h, d), device=la_out.device, dtype=torch.bfloat16)
    if s_total == 0 or h == 0:
        return out
    num_cores = _require_num_vectorcores(la_out.device)
    total_tiles = h * ((s_total + block_s - 1) // block_s)
    _la_postprocess_scaled_kernel[(min(num_cores, total_tiles),)](
        la_out,
        out,
        s_real,
        s_pad,
        s_total,
        SCALE=scale,
        NUM_HEADS=h,
        HEAD_DIM=d,
        BLOCK_S=block_s,
    )
    return out
