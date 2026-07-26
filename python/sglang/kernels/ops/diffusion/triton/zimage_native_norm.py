import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _rmsnorm_scale_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    scale_ptr,
    x_row_stride,
    scale_row_stride,
    seq_len,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < dim

    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    square = (x * x).to(tl.bfloat16)
    mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
    rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    batch = row // seq_len
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(
        scale_ptr + batch * scale_row_stride + offsets, mask=mask, other=0.0
    )
    y = (((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16) * scale).to(tl.bfloat16)
    tl.store(y_ptr + row * dim + offsets, y, mask=mask)


@triton.jit
def _rmsnorm_tanh_residual_kernel(
    y_ptr,
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    x_row_stride,
    gate_row_stride,
    residual_row_stride,
    seq_len,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_dim)
    mask = offsets < dim

    x = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
    square = (x * x).to(tl.bfloat16)
    mean_square = (tl.sum(square, axis=0) / dim).to(tl.bfloat16)
    rstd = tl.rsqrt((mean_square + eps).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

    batch = row // seq_len
    gate = tl.load(gate_ptr + batch * gate_row_stride + offsets, mask=mask, other=0.0)
    residual = tl.load(
        residual_ptr + row * residual_row_stride + offsets, mask=mask, other=0.0
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    norm = ((x * rstd).to(tl.bfloat16) * weight).to(tl.bfloat16)
    gated = (_tanh(gate.to(tl.float32)).to(tl.bfloat16) * norm).to(tl.bfloat16)
    y = (residual + gated).to(tl.bfloat16)
    tl.store(y_ptr + row * dim + offsets, y, mask=mask)


def _flat_row_stride(x: torch.Tensor) -> int | None:
    if x.dim() < 2 or x.stride(-1) != 1:
        return None
    row_stride = x.stride(-2)
    expected_stride = row_stride * x.shape[-2]
    for dim in range(x.dim() - 3, -1, -1):
        if x.stride(dim) != expected_stride:
            return None
        expected_stride *= x.shape[dim]
    return row_stride


def _can_use(x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and other.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and other.dtype == torch.bfloat16
        and weight.is_contiguous()
        and x.shape[-1] <= 8192
        and _flat_row_stride(x) is not None
        and _flat_row_stride(other) is not None
    )


def zimage_rmsnorm_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if not _can_use(x, weight, scale):
        return None
    shape = x.shape
    dim = shape[-1]
    x_rows = x.numel() // dim
    scale_rows = scale.numel() // dim
    if x_rows % scale_rows != 0:
        return None
    seq_len = x_rows // scale_rows
    x_row_stride = _flat_row_stride(x)
    scale_row_stride = _flat_row_stride(scale)
    if x_row_stride is None or scale_row_stride is None:
        return None
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_scale_kernel[(x_rows,)](
            y.reshape(-1, dim),
            x,
            weight,
            scale,
            x_row_stride,
            scale_row_stride,
            seq_len,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    return y


def zimage_rmsnorm_tanh_residual(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if not (_can_use(x, weight, gate) and residual.is_cuda):
        return None
    if residual.dtype != x.dtype or _flat_row_stride(residual) is None:
        return None
    shape = x.shape
    dim = shape[-1]
    x_rows = x.numel() // dim
    gate_rows = gate.numel() // dim
    if x_rows % gate_rows != 0:
        return None
    seq_len = x_rows // gate_rows
    x_row_stride = _flat_row_stride(x)
    gate_row_stride = _flat_row_stride(gate)
    residual_row_stride = _flat_row_stride(residual)
    if x_row_stride is None or gate_row_stride is None or residual_row_stride is None:
        return None
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.get_device_module().device(x.device):
        _rmsnorm_tanh_residual_kernel[(x_rows,)](
            y.reshape(-1, dim),
            x,
            gate,
            residual,
            weight,
            x_row_stride,
            gate_row_stride,
            residual_row_stride,
            seq_len,
            dim,
            eps,
            block_dim=triton.next_power_of_2(dim),
            num_warps=8,
        )
    return y
