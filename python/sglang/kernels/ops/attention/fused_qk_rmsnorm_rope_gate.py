import itertools
import logging
import math
import time
from functools import cache

"""Fused Q/K GemmaRMSNorm + NeoX RoPE + gate deinterleave (Triton).

Single kernel launch fusing per-head GemmaRMSNorm, partial NeoX RoPE over 1-D or
mrope positions, and gate deinterleave for Qwen3.5's interleaved Q+Gate layout.

2D grid (T, num_q_heads + num_kv_heads) — each program handles one
(token, head) pair. Q programs also copy the gate slice.
"""

from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.arch import current_arch


# MRoPE axis stride is the request length, not a compile-time geometry.
@triton.jit(do_not_specialize=["stride_pos_axis"])
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    mrope_axis_map_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_axis,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    MROPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    WEIGHT_SHIFT: tl.constexpr = 1.0,
    EXPLICIT_ROPE_FMA: tl.constexpr = False,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
    else:
        if HAS_GATE:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        else:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    # Full load -> RMSNorm variance
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    x_norm = (x * inv_rms * (w + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)

    # Pass-through tail [rotary_dim, head_dim)
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # Reload rotary portion from L1 -> re-norm -> RoPE
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    xr1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    xr2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    wr1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    wr2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    xr1 = (xr1 * inv_rms * (wr1 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + WEIGHT_SHIFT)).to(out_dtype).to(tl.float32)

    if MROPE:
        axis = tl.load(mrope_axis_map_ptr + rot_offs, mask=rot_mask, other=0)
        pos = tl.load(
            positions_ptr + axis * stride_pos_axis + token, mask=rot_mask, other=0
        )
    else:
        pos = tl.load(positions_ptr + token)
    cache_off = pos.to(tl.int64) * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    if EXPLICIT_ROPE_FMA:
        first = tl.fma(xr1, cos, -libdevice.mul_rn(xr2, sin))
        second = tl.fma(xr2, cos, libdevice.mul_rn(xr1, sin))
    else:
        first = xr1 * cos - xr2 * sin
        second = xr2 * cos + xr1 * sin
    tl.store(out_base + rot_offs, first, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, second, mask=rot_mask)

    # Gate copy (Q heads only)
    if HAS_GATE and not is_k:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)

    # PDL: signal dependent kernels (attention/allreduce) can start early.
    # Only available on NVIDIA Hopper+ (sm_90+); guarded for AMD/other backends.
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    has_gate: bool = True,
    mrope_axis_map: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave.

    Args:
        q_gate: [T, num_q_heads * (1 + has_gate) * head_dim] — interleaved Q+Gate if has_gate
        k: [T, num_kv_heads * head_dim]
        q_weight, k_weight: [head_dim] — raw GemmaRMSNorm weights (kernel adds +1.0)
        cos_sin_cache: [max_seq_len, rotary_dim] — [cos..., sin...]
        positions: [T] token positions, or [3, T] mrope rows (temporal, height, width)
        mrope_axis_map: [rotary_dim // 2] — the axis owning each rotary lane, from
            MRotaryEmbedding
    """
    assert positions.dim() in (1, 2), f"want [T] or [3, T], got {positions.shape}"
    mrope = positions.dim() == 2
    assert mrope == (mrope_axis_map is not None), "mrope_axis_map needs [3, T]"
    if mrope:
        assert positions.shape[0] == 3 and positions.stride(1) == 1, (
            f"want [3, T] contiguous over T, got {positions.shape} "
            f"stride {positions.stride()}"
        )
        lanes = rotary_dim // 2
        assert mrope_axis_map.shape == (lanes,), f"want one axis per lane ({lanes})"
    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_out = torch.empty(T, q_size, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(T, kv_size, dtype=k.dtype, device=k.device)
    gate_out = (
        torch.empty(T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device)
        if has_gate
        else q_out
    )

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)

    grid = (T, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        mrope_axis_map,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        EPS=eps,
        FP16=q_gate.dtype == torch.float16,
        HAS_PASS=rotary_dim < head_dim,
        HAS_GATE=has_gate,
        MROPE=mrope,
        ENABLE_PDL=current_arch().supports_pdl,
    )

    return q_out, k_out, gate_out if has_gate else None


def mrope_axis_map_values(
    rotary_dim: int,
    sections,
    *,
    interleaved: bool = False,
    interleaved_glm: bool = False,
):
    """Mirror MRotaryEmbedding's lane ownership using configuration only."""
    if (
        rotary_dim <= 0
        or rotary_dim % 2
        or sections is None
        or len(sections) != 3
        or any(section < 0 for section in sections)
        or sum(sections) != rotary_dim // 2
    ):
        return None
    if interleaved and interleaved_glm:
        counts = [0, 0, 0]
        axes = []
        for lane in range(rotary_dim // 2):
            axis = lane % 3
            while counts[axis] >= sections[axis]:
                axis = (axis + 1) % 3
            axes.append(axis)
            counts[axis] += 1
        return tuple(axes)
    if interleaved:
        return tuple(
            (
                1
                if lane % 3 == 1 and lane < 3 * sections[1]
                else 2
                if lane % 3 == 2 and lane < 3 * sections[2]
                else 0
            )
            for lane in range(rotary_dim // 2)
        )
    return tuple(axis for axis, count in enumerate(sections) for _ in range(count))


def can_use_fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    has_gate: bool = True,
    mrope_axis_map: Optional[torch.Tensor] = None,
    *,
    is_neox_style: bool = True,
) -> bool:
    """Check tensor metadata only; unsupported model paths keep separate ops.

    Position values must index the supplied cache; axis-map values must be 0..2.
    These are the same caller invariants as the separate RoPE implementation.
    No device synchronization or capability query is needed for this check.
    """
    floating_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        not is_neox_style
        or q_gate.device.type != "cuda"
        or torch.version.hip is not None
        or q_gate.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q_gate.dtype
        or q_weight.dtype not in floating_dtypes
        or k_weight.dtype != q_weight.dtype
        or cos_sin_cache.dtype not in floating_dtypes
        or positions.dtype not in (torch.int32, torch.int64)
        or not math.isfinite(eps)
        or eps <= 0
        or num_q_heads <= 0
        or num_kv_heads <= 0
        or head_dim <= 0
        or head_dim > 1024
        or not 0 < rotary_dim <= head_dim
        or rotary_dim % 2
        or q_gate.ndim != 2
        or k.ndim != 2
        or q_gate.shape[1] != num_q_heads * (1 + has_gate) * head_dim
        or k.shape != (q_gate.shape[0], num_kv_heads * head_dim)
        or q_gate.stride(-1) != 1
        or k.stride(-1) != 1
        or q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.stride(-1) != 1
        or not positions.is_contiguous()
    ):
        return False
    tensors = (k, q_weight, k_weight, cos_sin_cache, positions)
    if any(tensor.device != q_gate.device for tensor in tensors):
        return False
    tokens = q_gate.shape[0]
    if positions.ndim == 1:
        return positions.shape == (tokens,) and mrope_axis_map is None
    return (
        positions.shape == (3, tokens)
        and mrope_axis_map is not None
        and mrope_axis_map.device == q_gate.device
        and mrope_axis_map.dtype in (torch.int32, torch.int64)
        and mrope_axis_map.shape == (rotary_dim // 2,)
        and mrope_axis_map.is_contiguous()
    )


@cache
def warmup_fused_qk_rmsnorm_rope_gate(
    *,
    device,
    dtype,
    weight_dtype,
    cache_dtype,
    num_heads,
    num_kv_heads,
    head_size,
    rotary_dim,
    eps,
    has_gate,
    mrope_axes=None,
):
    """Compile the actual layer's QKV view geometry using private buffers.

    The fused path reads the supplied cache directly, including FP32 MRoPE
    caches. Its axes select the actual model path. MRoPE's 1-D native fallback
    is not fused. The kernel does not specialize on token count or axis stride.
    """

    if dtype not in (torch.float16, torch.bfloat16):
        return False
    started = time.monotonic()
    with torch.cuda.device(device), torch.inference_mode():
        q_weight = torch.zeros(head_size, dtype=weight_dtype, device=device)
        k_weight = torch.zeros_like(q_weight)
        axis_map = (
            torch.tensor(mrope_axes, dtype=torch.int64, device=device)
            if mrope_axes is not None
            else None
        )
        q_width = num_heads * head_size * (1 + has_gate)
        kv_width = num_kv_heads * head_size
        mrope = axis_map is not None
        rope_cache = torch.zeros((1, rotary_dim), dtype=cache_dtype, device=device)
        for tokens, index_dtype in itertools.product(
            (1, 64), (torch.int32, torch.int64)
        ):
            projection = torch.zeros(
                (tokens, q_width + 2 * kv_width), dtype=dtype, device=device
            )
            q_gate, k, _ = projection.split((q_width, kv_width, kv_width), dim=-1)
            positions = torch.zeros(
                (3, tokens) if mrope else (tokens,), dtype=index_dtype, device=device
            )
            arguments = (
                q_gate,
                k,
                q_weight,
                k_weight,
                rope_cache,
                positions,
                eps,
                num_heads,
                num_kv_heads,
                head_size,
                rotary_dim,
                has_gate,
                axis_map,
            )
            if not can_use_fused_qk_gemma_rmsnorm_rope_gate(*arguments):
                return False
            fused_qk_gemma_rmsnorm_rope_gate(*arguments)
        torch.cuda.synchronize(device)
    logger.info(
        "CP QK/RoPE warmup complete: heads=%d/%d head_size=%d rotary_dim=%d "
        "mrope=%s scratch_only=True elapsed=%.3fs",
        num_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        axis_map is not None,
        time.monotonic() - started,
    )
    return True
