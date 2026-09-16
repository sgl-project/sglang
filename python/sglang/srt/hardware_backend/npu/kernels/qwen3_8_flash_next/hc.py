"""HC vector fusions retaining the Torch intermediate rounding boundaries."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# The NPU runtime limits the total number of programs in a launch.
_MAX_GRID_PROGRAMS = 65535
# Larger batches did not improve over the eager mix/combine in graph microtests.
_MAX_FUSED_ROWS = 32


@triton.jit
def _norm(
    X,
    W,
    Y,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * DIM + col, col < DIM, other=0).to(tl.float32)
    w = tl.load(W + (row * DIM % WIDTH) + col, col < DIM, other=0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / DIM + EPS)
    y = (x * inv) * (1.0 + w.to(tl.float32))
    tl.store(Y + row * DIM + col, y, col < DIM)


@triton.jit
def _silu(X, Y, N: tl.constexpr, HC: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dtype = Y.dtype.element_ty
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    x = (x / HC).to(dtype).to(tl.float32)
    tl.store(Y + i, x * tl.sigmoid(x), i < N)


@triton.jit
def _mix(
    X,
    G,
    Y,
    HC: tl.constexpr,
    DIM: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    head = tl.arange(0, HEADS)
    offset = row * HC * DIM + head[:, None] * DIM + col[None, :]
    mask = (head[:, None] < HC) & (col[None, :] < DIM)
    dtype = Y.dtype.element_ty
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    g = tl.load(G + offset, mask, other=0).to(tl.float32)
    gate = tl.sigmoid(g).to(dtype).to(tl.float32)
    product = (x * gate).to(dtype).to(tl.float32)
    y = tl.sum(product, 0) / HC
    tl.store(Y + row * DIM + col, y, col < DIM)


@triton.jit
def _combine(
    B,
    R,
    G,
    Y,
    HC: tl.constexpr,
    DIM: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.arange(0, HEADS)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dtype = Y.dtype.element_ty
    gate = tl.load(G + row * HC + head, head < HC, other=0)
    gate = (gate.to(tl.float32) / HC).to(dtype).to(tl.float32)
    gate = tl.sigmoid(gate).to(dtype).to(tl.float32)
    gate = (2 * gate).to(dtype).to(tl.float32)
    block = tl.load(B + row * DIM + col, col < DIM, other=0)
    offset = (row * HC + head[:, None]) * DIM + col[None, :]
    mask = (head[:, None] < HC) & (col[None, :] < DIM)
    residual = tl.load(R + offset, mask, other=0)
    injection = block[None, :].to(tl.float32) * gate[:, None]
    injection = injection.to(dtype).to(tl.float32)
    y = residual.to(tl.float32) + injection
    tl.store(Y + offset, y, mask)


def _supported(x):
    return (
        x.device.type == "npu"
        and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and x.ndim >= 2
    )


def can_run_norm(x, weight, group_size):
    if not _supported(x):
        return False
    dim = group_size if group_size is not None else x.shape[-1]
    return (
        isinstance(dim, int)
        and 0 < dim <= 8192
        and x.shape[-1] % dim == 0
        and x.numel() // dim <= _MAX_GRID_PROGRAMS
        and weight.shape == (x.shape[-1],)
        and weight.device == x.device
        and weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and weight.is_contiguous()
    )


def grouped_norm(x, weight, group_size, eps):
    dim = group_size if group_size is not None else x.shape[-1]
    y = torch.empty_like(x)
    if x.numel():
        _norm[(x.numel() // dim,)](
            x,
            weight,
            y,
            dim,
            weight.numel(),
            eps,
            triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )
    return y


def can_run_mix(x, down, up, hc, hs):
    return (
        _supported(x)
        and x.ndim == 2
        and x.shape[0] <= _MAX_FUSED_ROWS
        and hc in (1, 2, 3, 4, 5, 8)
        and 0 < hs <= 8192
        and x.shape[1] == hc * hs
        and down.ndim == up.ndim == 2
        and down.shape[1] == hc * hs
        and 0 < down.shape[0]
        # This also keeps the flat SiLU element offsets within int32 range.
        and triton.cdiv(x.shape[0] * down.shape[0], 256) <= _MAX_GRID_PROGRAMS
        and up.shape == (hc * hs, down.shape[0])
        and down.device == up.device == x.device
        and down.dtype == up.dtype == x.dtype
        and down.is_contiguous()
        and up.is_contiguous()
    )


def mix(x, down, up, hc, hs):
    hidden = F.linear(x, down)
    activated = torch.empty_like(hidden)
    if hidden.numel():
        _silu[(triton.cdiv(hidden.numel(), 256),)](
            hidden,
            activated,
            hidden.numel(),
            hc,
            256,
            enable_fp_fusion=False,
        )
    gates = F.linear(activated, up)
    y = x.new_empty((x.shape[0], hs))
    if x.shape[0]:
        _mix[(x.shape[0], triton.cdiv(hs, 256))](
            x,
            gates,
            y,
            hc,
            hs,
            triton.next_power_of_2(hc),
            256,
            enable_fp_fusion=False,
        )
    return y


def can_run_combine(block, residual, normed, weight, hc, hs):
    return (
        _supported(residual)
        and residual.ndim == 2
        and residual.shape[0] <= _MAX_FUSED_ROWS
        and hc in (1, 2, 3, 4, 5, 8)
        and 0 < hs <= 8192
        and residual.shape[1] == hc * hs
        and block.shape == (residual.shape[0], hs)
        and normed.shape == residual.shape
        and weight.shape == (hc, hc * hs)
        and all(
            t.device == residual.device
            and t.dtype == residual.dtype
            and t.is_contiguous()
            for t in (block, normed, weight)
        )
    )


def combine(block, residual, normed, weight, hc, hs):
    gates = F.linear(normed, weight)
    y = torch.empty_like(residual)
    if residual.shape[0]:
        _combine[(residual.shape[0], triton.cdiv(hs, 512))](
            block,
            residual,
            gates,
            y,
            hc,
            hs,
            triton.next_power_of_2(hc),
            512,
            enable_fp_fusion=False,
        )
    return y
