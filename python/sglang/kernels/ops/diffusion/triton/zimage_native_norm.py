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


@triton.jit
def _qk_rmsnorm_native_kernel(
    y_ptr,
    x_ptr,
    weight_ptr,
    token_stride,
    nheads,
    n_rows,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    rows_per_prog: tl.constexpr,
):
    """Per-head RMSNorm replicating the eager ZImageRMSNorm kernel chain bit-for-bit.

    The eager path (`ZImageRMSNorm.forward` on a `[rows, head_dim]` bf16 tensor)
    lowers to aten pow/mean/rsqrt/mul kernels. For bf16 rows of 128, aten's
    reduce_kernel vectorizes with 16-byte loads (8 bf16 elements per lane):
    lane t serially accumulates contiguous elements 8t .. 8t+7 in fp32
    (squares rounded to bf16 first, matching aten::pow), then a shfl-down
    butterfly tree combines the 32 lane partials (halves 16/8/4/2/1). Every
    intermediate below is rounded to bf16 at exactly the same points as the
    aten kernels, so the output satisfies `torch.equal` against the eager path
    (verified over 9M+ random rows across scales 0.005-200).
    """
    prog = tl.program_id(0)
    row_offs = tl.arange(0, rows_per_prog)
    rows = prog * rows_per_prog + row_offs
    row_mask = rows < n_rows
    tokens = rows // nheads
    heads = rows % nheads
    base = x_ptr + tokens * token_stride + heads * head_dim

    offs = tl.arange(0, head_dim)
    x = tl.load(base[:, None] + offs[None, :], mask=row_mask[:, None], other=0.0)
    sq = (x * x).to(tl.bfloat16)

    # Lane t's serial accumulation of contiguous elements 8t..8t+7: peel the
    # 8-element groups apart with ordered tl.split chains (loads stay one
    # coalesced pass) and add them in exact serial order in fp32.
    g = tl.reshape(sq, (rows_per_prog, head_dim // 8, 4, 2), can_reorder=False)
    p0, p1 = tl.split(g)  # elements (0,2,4,6) / (1,3,5,7)
    p00, p01 = tl.split(
        tl.reshape(p0, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )  # (0,4) / (2,6)
    p10, p11 = tl.split(
        tl.reshape(p1, (rows_per_prog, head_dim // 8, 2, 2), can_reorder=False)
    )  # (1,5) / (3,7)
    s0, s4 = tl.split(
        tl.reshape(p00, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s2, s6 = tl.split(
        tl.reshape(p01, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s1, s5 = tl.split(
        tl.reshape(p10, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    s3, s7 = tl.split(
        tl.reshape(p11, (rows_per_prog, head_dim // 8, 1, 2), can_reorder=False)
    )
    acc = tl.reshape(s0, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s1, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s2, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s3, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s4, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s5, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s6, (rows_per_prog, head_dim // 8)).to(tl.float32)
    acc = acc + tl.reshape(s7, (rows_per_prog, head_dim // 8)).to(tl.float32)
    # shfl-down butterfly over the 16 lane partials (lanes 16..31 hold the
    # additive identity in aten's 32-lane warp, so their stage is an exact
    # no-op): acc[i] += acc[i + half] for half in (8, 4, 2, 1).
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 8), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 4), can_reorder=False), axis=1)
    acc = tl.sum(tl.reshape(acc, (rows_per_prog, 2, 2), can_reorder=False), axis=1)
    ssum = tl.sum(acc, axis=1)
    ms = (ssum / head_dim).to(tl.bfloat16)
    # aten adds the python-float eps in fp32 opmath, then rsqrt rounds to bf16.
    rstd = tl.rsqrt((ms.to(tl.float32) + eps).to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16
    )

    weight = tl.load(weight_ptr + offs)
    y = ((x.to(tl.float32) * rstd.to(tl.float32)[:, None]).to(tl.bfloat16) * weight).to(
        tl.bfloat16
    )
    tl.store(
        y_ptr + rows[:, None] * head_dim + offs[None, :], y, mask=row_mask[:, None]
    )


def _qk_head_token_stride(x: torch.Tensor, head_dim: int) -> int | None:
    """Token stride for a `[B, S, H, head_dim]` view whose head block is packed.

    Accepts the strided views produced by slicing a fused qkv projection: the
    last dim must be contiguous, heads packed (`stride(-2) == head_dim`), and
    tokens uniformly strided across the flattened `(B, S)` dims.
    """
    if x.dim() != 4 or x.shape[-1] != head_dim:
        return None
    if x.stride(-1) != 1 or x.stride(-2) != head_dim:
        return None
    token_stride = x.stride(1)
    if x.shape[0] > 1 and x.stride(0) != x.shape[1] * token_stride:
        return None
    return token_stride


def can_use_qk_rmsnorm_native(
    x: torch.Tensor, weight: torch.Tensor, head_dim: int
) -> bool:
    return (
        x.is_cuda
        and weight.is_cuda
        and x.dtype == torch.bfloat16
        and head_dim == 128
        and weight.numel() == head_dim
        and weight.is_contiguous()
        and _qk_head_token_stride(x, head_dim) is not None
    )


def zimage_qk_rmsnorm_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fused, bit-exact ZImageRMSNorm over q/k heads; returns a contiguous tensor.

    Reads strided `[B, S, H, head_dim]` views (e.g. fused-qkv slices) directly,
    absorbing the `.contiguous()` materialization the eager path needs.
    Returns None when the input is not supported (caller falls back).
    """
    head_dim = x.shape[-1]
    if not can_use_qk_rmsnorm_native(x, weight, head_dim):
        return None
    token_stride = _qk_head_token_stride(x, head_dim)
    if token_stride is None:
        return None
    if weight.dtype != x.dtype:
        weight = weight.to(dtype=x.dtype)
    nheads = x.shape[2]
    n_rows = x.shape[0] * x.shape[1] * nheads
    rows_per_prog = 8
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n_rows, rows_per_prog),)
    with torch.get_device_module().device(x.device):
        _qk_rmsnorm_native_kernel[grid](
            y,
            x,
            weight,
            token_stride,
            nheads,
            n_rows,
            head_dim,
            eps,
            rows_per_prog=rows_per_prog,
            num_warps=8,
        )
    return y
