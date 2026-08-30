"""Fused Q/K GemmaRMSNorm + NeoX RoPE + gate deinterleave (Triton).

Single kernel launch fusing per-head GemmaRMSNorm, partial NeoX RoPE,
and gate deinterleave for Qwen3.5's interleaved Q+Gate layout.

Positions are either [T] (plain RoPE) or [3, T] mRoPE positions, in which
case each rotary pair takes its angle from the temporal, height or width
row as selected by ``mrope_section``.

2D grid (T, num_q_heads + num_kv_heads) — each program handles one
(token, head) pair. Q programs also copy the gate slice.
"""

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl


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
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    stride_pos_m,
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
    MROPE_SECTION_T: tl.constexpr,
    MROPE_SECTION_H: tl.constexpr,
    MROPE_SECTION_W: tl.constexpr,
    MROPE_INTERLEAVED: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
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
    x_norm = (x * inv_rms * (w + 1.0)).to(out_dtype).to(tl.float32)

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
    xr1 = (xr1 * inv_rms * (wr1 + 1.0)).to(out_dtype).to(tl.float32)
    xr2 = (xr2 * inv_rms * (wr2 + 1.0)).to(out_dtype).to(tl.float32)

    if MROPE:
        # Masks are ANDed with rot_mask so padded lanes (ROT_HALF_BLOCK >
        # HALF_ROTARY) never address past the end of a cos_sin_cache row.
        pos_t = tl.load(positions_ptr + 0 * stride_pos_m + token).to(tl.int64)
        pos_h = tl.load(positions_ptr + 1 * stride_pos_m + token).to(tl.int64)
        pos_w = tl.load(positions_ptr + 2 * stride_pos_m + token).to(tl.int64)
        if MROPE_INTERLEAVED:
            h_mask = ((rot_offs % 3) == 1) & (rot_offs < 3 * MROPE_SECTION_H)
            w_mask = ((rot_offs % 3) == 2) & (rot_offs < 3 * MROPE_SECTION_W)
        else:
            h_end = MROPE_SECTION_T + MROPE_SECTION_H
            h_mask = (rot_offs >= MROPE_SECTION_T) & (rot_offs < h_end)
            w_mask = rot_offs >= h_end
        h_mask = h_mask & rot_mask
        w_mask = w_mask & rot_mask
        t_mask = rot_mask & ~(h_mask | w_mask)
        t_off = pos_t * stride_cos_t + rot_offs
        h_off = pos_h * stride_cos_t + rot_offs
        w_off = pos_w * stride_cos_t + rot_offs
        cos = (
            tl.load(cos_sin_cache_ptr + t_off, mask=t_mask, other=0.0)
            + tl.load(cos_sin_cache_ptr + h_off, mask=h_mask, other=0.0)
            + tl.load(cos_sin_cache_ptr + w_off, mask=w_mask, other=0.0)
        ).to(tl.float32)
        sin = (
            tl.load(cos_sin_cache_ptr + HALF_ROTARY + t_off, mask=t_mask, other=0.0)
            + tl.load(cos_sin_cache_ptr + HALF_ROTARY + h_off, mask=h_mask, other=0.0)
            + tl.load(cos_sin_cache_ptr + HALF_ROTARY + w_off, mask=w_mask, other=0.0)
        ).to(tl.float32)
    else:
        pos = tl.load(positions_ptr + token).to(tl.int64)
        cache_off = pos * stride_cos_t
        cos = tl.load(
            cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs,
            mask=rot_mask,
            other=0.0,
        ).to(tl.float32)
    tl.store(out_base + rot_offs, (xr1 * cos - xr2 * sin), mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, (xr2 * cos + xr1 * sin), mask=rot_mask)

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


def _check_positions_contract(
    positions: torch.Tensor,
    num_tokens: int,
    half_rotary: int,
    mrope_section: Optional[Sequence[int]],
    mrope_interleaved: bool,
    mrope_interleaved_glm: bool,
) -> bool:
    """Validate the positions contract; return True iff the mRoPE path applies.

    Raises rather than asserts: these guard against silently computing the
    wrong RoPE, and asserts vanish under ``python -O``.
    """
    if mrope_interleaved_glm:
        raise ValueError(
            "fused_qk_gemma_rmsnorm_rope_gate does not implement GLM-interleaved "
            "mRoPE (mrope_interleaved_glm=True); use the unfused RoPE path."
        )
    if positions.ndim not in (1, 2):
        raise ValueError(
            f"positions must be [T] or [3, T], got shape {tuple(positions.shape)}"
        )
    if positions.shape[-1] != num_tokens:
        raise ValueError(
            f"positions has {positions.shape[-1]} tokens but q_gate has {num_tokens}"
        )
    if positions.ndim == 1:
        if mrope_section is not None or mrope_interleaved:
            raise ValueError(
                "mRoPE arguments require [3, T] positions, got 1D [T] positions"
            )
        return False
    if positions.shape[0] != 3:
        raise ValueError(
            "2D positions must be [3, T] (temporal, height, width), got "
            f"{tuple(positions.shape)}"
        )
    if mrope_section is None:
        raise ValueError(
            "[3, T] positions require mrope_section; without it this kernel would "
            "read only the temporal row and silently drop height/width RoPE."
        )
    if len(mrope_section) != 3:
        raise ValueError(
            f"mrope_section must have 3 entries, got {list(mrope_section)}"
        )
    if sum(mrope_section) != half_rotary:
        raise ValueError(
            f"sum(mrope_section)={sum(mrope_section)} != rotary_dim // 2 = {half_rotary}"
        )
    if positions.stride(1) != 1:
        raise ValueError(
            "[3, T] positions must be contiguous along the token dim "
            f"(stride(1) == 1), got strides {positions.stride()}"
        )
    return True


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
    mrope_section: Optional[Sequence[int]] = None,
    mrope_interleaved: bool = False,
    mrope_interleaved_glm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fused QK GemmaRMSNorm + NeoX RoPE + gate deinterleave.

    Args:
        q_gate: [T, num_q_heads * (1 + has_gate) * head_dim] — interleaved Q+Gate if has_gate
        k: [T, num_kv_heads * head_dim]
        q_weight, k_weight: [head_dim] — raw GemmaRMSNorm weights (kernel adds +1.0)
        cos_sin_cache: [max_seq_len, rotary_dim] — [cos..., sin...]
        positions: [T] token positions, or [3, T] mRoPE (temporal, height, width)
            positions, which require ``mrope_section``.
        mrope_section: 3 entries summing to rotary_dim // 2, selecting which
            rotary pairs follow the temporal, height and width axes.
        mrope_interleaved: round-robin axis assignment (Qwen3-VL / Qwen3.5)
            instead of contiguous per-axis sections.
        mrope_interleaved_glm: unsupported here; passing True raises.
    """
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
    use_mrope = _check_positions_contract(
        positions=positions,
        num_tokens=T,
        half_rotary=half_rotary,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
        mrope_interleaved_glm=mrope_interleaved_glm,
    )
    stride_pos_m = positions.stride(0) if use_mrope else 0
    sec_t, sec_h, sec_w = tuple(mrope_section) if use_mrope else (0, 0, 0)
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
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        stride_pos_m,
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
        MROPE=use_mrope,
        MROPE_SECTION_T=sec_t,
        MROPE_SECTION_H=sec_h,
        MROPE_SECTION_W=sec_w,
        MROPE_INTERLEAVED=bool(mrope_interleaved),
        ENABLE_PDL=_ENABLE_PDL,
    )

    return q_out, k_out, gate_out if has_gate else None
