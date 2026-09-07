import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _scale_residual_norm_scale_shift_kernel(
    residual_out_ptr,
    out_ptr,  # outputs, x.dtype
    residual_ptr,
    x_ptr,  # inputs, x.dtype
    gate_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    frame_seqlen,
    gate_frame_stride,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_AFFINE: tl.constexpr,
    HAS_GATE: tl.constexpr,
    SCALE_VEC: tl.constexpr,
    SHIFT_VEC: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    off = row * D + cols

    res = tl.load(residual_ptr + off, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_GATE:
        frame = row // frame_seqlen
        g = tl.load(gate_ptr + frame * gate_frame_stride + cols, mask=mask, other=0.0)
        residual_output = res + xv * g
    else:
        residual_output = res + xv

    tl.store(
        residual_out_ptr + off,
        residual_output.to(residual_out_ptr.dtype.element_ty),
        mask=mask,
    )

    mean = tl.sum(residual_output, axis=0) / D
    centered = tl.where(mask, residual_output - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / D
    normed = centered * (1.0 / tl.sqrt(var + eps))
    if HAS_AFFINE:
        normed = normed * tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
            tl.float32
        ) + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    if SCALE_VEC:
        sc = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    else:
        sc = tl.load(scale_ptr)
    if SHIFT_VEC:
        sh = tl.load(shift_ptr + cols, mask=mask, other=0.0)
    else:
        sh = tl.load(shift_ptr)
    tl.store(
        out_ptr + off,
        (normed * (1.0 + sc) + sh).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def can_use_fused_scale_residual_norm_scale_shift_triton(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> bool:
    if x.device.type != "xpu" or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.dim() != 3 or x.shape[0] != 1 or not x.is_contiguous():
        return False
    if (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or not residual.is_contiguous()
    ):
        return False
    hidden = x.shape[-1]
    if isinstance(gate, torch.Tensor):
        if (
            gate.dim() not in (3, 4)
            or gate.shape[0] != 1
            or gate.shape[-1] != hidden
            or not gate.is_contiguous()
        ):
            return False
        if gate.dim() == 3:
            if gate.shape[1] != 1:
                return False
        elif gate.shape[2] != 1 or x.shape[1] % gate.shape[1] != 0:
            return False
    elif gate != 1:
        return False
    for modulation in (scale, shift):
        if not isinstance(modulation, torch.Tensor) or not modulation.is_contiguous():
            return False
        if modulation.numel() not in (1, hidden):
            return False
    if (weight is None) != (bias is None):
        return False
    if weight is not None and (weight.numel() != hidden or bias.numel() != hidden):
        return False
    return True


def fused_scale_residual_norm_scale_shift_triton(
    *,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len, hidden = x.shape[1], x.shape[2]
    residual_output = torch.empty_like(x)
    out = torch.empty_like(x)
    has_gate = isinstance(gate, torch.Tensor)
    if has_gate and gate.dim() == 4:
        frame_seqlen, gate_frame_stride = seq_len // gate.shape[1], hidden
    else:
        frame_seqlen, gate_frame_stride = seq_len, 0
    _scale_residual_norm_scale_shift_kernel[(seq_len,)](
        residual_output,
        out,
        residual,
        x,
        gate.reshape(-1) if has_gate else x,
        weight if weight is not None else x,
        bias if bias is not None else x,
        scale.reshape(-1),
        shift.reshape(-1),
        frame_seqlen,
        gate_frame_stride,
        eps,
        D=hidden,
        BLOCK_D=triton.next_power_of_2(hidden),
        HAS_AFFINE=weight is not None,
        HAS_GATE=has_gate,
        SCALE_VEC=scale.numel() == hidden,
        SHIFT_VEC=shift.numel() == hidden,
        num_warps=8,
    )
    return out, residual_output
