import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_gelu_tanh_inplace_kernel(
    x_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x3 = x * x * x
    y = x * tl.sigmoid(1.5957691216057308 * (x + 0.044715 * x3))
    tl.store(x_ptr + offsets, y, mask=mask)


def ltx2_gelu_tanh_inplace(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("x must be a CUDA fp16/bf16 tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    n_elements = x.numel()
    if n_elements == 0:
        return x
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _ltx2_gelu_tanh_inplace_kernel[grid](
        x,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return x


@triton.jit
def _ltx2_bias_residual_gate_kernel(
    update_ptr,
    residual_ptr,
    gate_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    gate_stride_b: tl.constexpr,
    gate_stride_t: tl.constexpr,
    gate_stride_d: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < hidden
    batch = row // rows
    token = row - batch * rows
    base = row * hidden + cols
    gate_base = batch * gate_stride_b + token * gate_stride_t + cols * gate_stride_d

    update = tl.load(update_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_base, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = residual + (update + bias) * gate
    tl.store(out_ptr + base, out, mask=mask)


def ltx2_bias_residual_gate(
    update: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if update.shape != residual.shape or update.ndim != 3:
        raise ValueError("update and residual must have shape [B, T, D]")
    if (
        not update.is_cuda
        or not residual.is_cuda
        or not gate.is_cuda
        or not bias.is_cuda
    ):
        raise ValueError("all inputs must be CUDA tensors")
    if (
        update.dtype != residual.dtype
        or update.dtype != gate.dtype
        or update.dtype != bias.dtype
    ):
        raise ValueError("all inputs must have the same dtype")
    if update.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("only fp16/bf16 are supported")
    if not update.is_contiguous() or not residual.is_contiguous():
        raise ValueError("update and residual must be contiguous")
    if not bias.is_contiguous():
        raise ValueError("bias must be contiguous")

    batch, tokens, hidden = update.shape
    if bias.ndim != 1 or bias.shape[0] != hidden:
        raise ValueError("bias must match hidden size")
    if gate.ndim == 2:
        if gate.shape != (batch, hidden):
            raise ValueError("2D gate must have shape [B, D]")
        gate_view = gate.contiguous()
        gate_stride_b = gate_view.stride(0)
        gate_stride_t = 0
        gate_stride_d = gate_view.stride(1)
    elif gate.ndim == 3:
        if (
            gate.shape[0] != batch
            or gate.shape[-1] != hidden
            or gate.shape[1] not in (1, tokens)
        ):
            raise ValueError("3D gate must have shape [B, 1|T, D]")
        gate_view = gate.contiguous() if not gate.is_contiguous() else gate
        gate_stride_b = gate_view.stride(0)
        gate_stride_t = 0 if gate_view.shape[1] == 1 else gate_view.stride(1)
        gate_stride_d = gate_view.stride(2)
    else:
        raise ValueError("gate must be 2D or 3D")

    out = torch.empty_like(residual)
    _ltx2_bias_residual_gate_kernel[(batch * tokens,)](
        update,
        residual,
        gate_view,
        bias,
        out,
        tokens,
        hidden,
        gate_stride_b,
        gate_stride_t,
        gate_stride_d,
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=8 if hidden >= 4096 else 4,
    )
    return out
