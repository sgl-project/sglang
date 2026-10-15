"""Fused Q/K GemmaRMSNorm + NeoX RoPE + gate deinterleave (Triton).

Single kernel launch fusing per-head GemmaRMSNorm, partial NeoX RoPE over 1-D or
mrope positions, and gate deinterleave for Qwen3.5's interleaved Q+Gate layout.

2D grid (T, num_q_heads + num_kv_heads) — each program handles one
(token, head) pair. Q programs also copy the gate slice.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def _pdl_supported() -> bool:
    """Check if Programmatic Dependent Launch is supported (NVIDIA SM >= 90)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


_ENABLE_PDL = _pdl_supported()


@triton.jit
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
    v_ptr=None,
    v_cache_ptr=None,
    cache_loc_ptr=None,
    k_scale_ptr=None,
    v_scale_ptr=None,
    stride_v_t: tl.constexpr = 0,
    stride_v_cache_t: tl.constexpr = 0,
    FP8_CACHE: tl.constexpr = False,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16
    if FP8_CACHE:
        slot = tl.load(cache_loc_ptr + token).to(tl.int64)
        # Match the cache writer's round-to-nearest reciprocal.
        inv_k = tl.div_rn(1.0, tl.load(k_scale_ptr).to(tl.float32))

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
        if FP8_CACHE:
            out_base = k_out_ptr + slot * stride_ko_t + local_head * HEAD_DIM
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
        tail = x_norm
        if FP8_CACHE and is_k:
            tail *= inv_k
        tl.store(out_base + head_offs, tail, mask=pass_mask)

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
    if FP8_CACHE:
        first = first.to(out_dtype).to(tl.float32)
        second = second.to(out_dtype).to(tl.float32)
        if is_k:
            first *= inv_k
            second *= inv_k
    tl.store(out_base + rot_offs, first, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, second, mask=rot_mask)

    if FP8_CACHE and is_k:
        value = tl.load(
            v_ptr + token * stride_v_t + local_head * HEAD_DIM + head_offs,
            head_mask,
            0.0,
        ).to(tl.float32)
        inv_v = tl.div_rn(1.0, tl.load(v_scale_ptr).to(tl.float32))
        value_out = v_cache_ptr + slot * stride_v_cache_t + local_head * HEAD_DIM
        tl.store(value_out + head_offs, value * inv_v, head_mask)

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
    *,
    v: Optional[torch.Tensor] = None,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    cache_loc: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

    fp8_cache = kv_cache is not None
    q_out = torch.empty(
        T,
        q_size,
        dtype=torch.float8_e4m3fn if fp8_cache else q_gate.dtype,
        device=q_gate.device,
    )
    k_out = (
        kv_cache[0]
        if fp8_cache
        else torch.empty(T, kv_size, dtype=k.dtype, device=k.device)
    )
    gate_out = (
        torch.empty(T, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device)
        if has_gate
        else q_out
    )

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)

    grid = (T, num_q_heads + num_kv_heads)
    kwargs = {}
    if fp8_cache:
        assert v is not None and cache_loc is not None
        assert k_scale is not None and v_scale is not None
        assert kv_cache[0].dtype == kv_cache[1].dtype == torch.float8_e4m3fn
        kwargs = dict(
            v_ptr=v,
            v_cache_ptr=kv_cache[1],
            cache_loc_ptr=cache_loc,
            k_scale_ptr=k_scale,
            v_scale_ptr=v_scale,
            stride_v_t=v.stride(0),
            stride_v_cache_t=kv_cache[1].stride(0),
            FP8_CACHE=True,
        )
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
        ENABLE_PDL=_ENABLE_PDL,
        **kwargs,
    )

    return q_out, None if fp8_cache else k_out, gate_out if has_gate else None
