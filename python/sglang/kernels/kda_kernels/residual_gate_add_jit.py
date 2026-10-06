from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kda_kernels import _cuda_source
from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BIT_EXACT_DTYPES = (torch.float16, torch.bfloat16)
_TRANSPOSE_TILE = 32
_MAX_GRID_DIM = 65535


@cache_once
def _jit_residual_gate_add_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported residual_gate_add dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_residual_gate_add",
        *args,
        cuda_files=[_cuda_source("diffusion/residual_gate_add.cuh")],
        cuda_wrappers=[
            (
                "residual_gate_add",
                f"residual_gate_add::ResidualGateAddKernel<{args}>::run",
            ),
        ],
    )


def _fake_impl(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.empty_strided(
        residual.shape,
        residual.stride(),
        dtype=residual.dtype,
        device=residual.device,
    )


def _residual_gate_add_cuda_impl(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_strided(
        residual.shape,
        residual.stride(),
        dtype=residual.dtype,
        device=residual.device,
    )
    module = _jit_residual_gate_add_module(residual.dtype)
    gate_mode = _gate_mode(residual, gate)
    module.residual_gate_add(
        out.view(-1),
        residual.view(-1),
        update.view(-1),
        gate.view(-1),
        residual.shape[-1],
        gate_mode,
    )
    return out


@triton.jit
def _cuda_round16_f32(x, IS_BF16: tl.constexpr):
    if IS_BF16:
        bits = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.int16, is_pure=True, pack=1
        )
        return bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        bits = tl.inline_asm_elementwise(
            "cvt.rn.f16.f32 $0, $1;", "=h,r", [x], dtype=tl.int16, is_pure=True, pack=1
        )
        return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _rga_transposed(
    out,
    res,
    upd,
    gate,
    tokens,
    hid: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_16: tl.constexpr,
    TILE: tl.constexpr,
):
    # residual/out have logical [batch, tokens, hid] with physical strides (T*H, 1, T);
    # update is contiguous. out[b,t,h] = res + upd * gate[h].
    # h-major tile indexing ([TILE_h, TILE_t]) makes the residual/output access
    # coalesced along the physical stride-1 token dim; update (read once) takes
    # the strided pattern instead. Measured faster than the t-major layout.
    pid_t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)
    t = pid_t * TILE + tl.arange(0, TILE).to(tl.int64)
    h = pid_h * TILE + tl.arange(0, TILE).to(tl.int64)
    mt = t < tokens
    mh = h < hid
    gv = tl.load(gate + h, mask=mh, other=0.0)
    base = pid_b * tokens * hid
    m2 = mh[:, None] & mt[None, :]
    roffs = h[:, None] * tokens + t[None, :]
    rv = tl.load(res + base + roffs, mask=m2, other=0.0)
    uv = tl.load(upd + base + t[None, :] * hid + h[:, None], mask=m2, other=0.0)
    if IS_16:
        p32 = uv.to(tl.float32) * gv[:, None].to(tl.float32)
        pf = _cuda_round16_f32(p32, IS_BF16)
        o32 = rv.to(tl.float32) + pf
        if IS_BF16:
            bits = tl.inline_asm_elementwise(
                "cvt.rn.bf16.f32 $0, $1;",
                "=h,r",
                [o32],
                dtype=tl.int16,
                is_pure=True,
                pack=1,
            )
            tl.store(out + base + roffs, bits.to(tl.bfloat16, bitcast=True), mask=m2)
        else:
            bits = tl.inline_asm_elementwise(
                "cvt.rn.f16.f32 $0, $1;",
                "=h,r",
                [o32],
                dtype=tl.int16,
                is_pure=True,
                pack=1,
            )
            tl.store(out + base + roffs, bits.to(tl.float16, bitcast=True), mask=m2)
    else:
        tl.store(out + base + roffs, rv + uv * gv[:, None], mask=m2)


def _residual_gate_add_transposed(residual, update, gate):
    out = torch.empty_strided(
        residual.shape, residual.stride(), dtype=residual.dtype, device=residual.device
    )
    batch, tokens, hidden = residual.shape
    if residual.numel() <= 65536:
        tile, warps = 16, 4
    elif residual.numel() >= 1 << 20:
        tile, warps = 64, 8
    else:
        tile, warps = 32, 8
    grid = (triton.cdiv(tokens, tile), triton.cdiv(hidden, tile), batch)
    _rga_transposed[grid](
        out,
        residual,
        update,
        gate,
        tokens,
        hidden,
        residual.dtype == torch.bfloat16,
        residual.dtype in (torch.float16, torch.bfloat16),
        TILE=tile,
        num_warps=warps,
    )
    return out


@triton.jit
def _rocm_round16_f32(x, IS_BF16: tl.constexpr):
    """Round to 16-bit precision on AMD without NVIDIA PTX."""
    if IS_BF16:
        return round_bf16_to_fp32(x)
    bits = tl.inline_asm_elementwise(
        "v_cvt_f16_f32 $0, $1",
        "=v,v",
        [x],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _rocm_store16(out_ptr, offs, value_f32, mask, IS_BF16: tl.constexpr):
    if IS_BF16:
        value = value_f32.to(tl.bfloat16)
    else:
        value = value_f32.to(tl.float16)
    tl.store(out_ptr + offs, value, mask=mask)


@triton.jit
def _rga_rocm_flat(
    out,
    res,
    upd,
    gate,
    numel,
    hid,
    MODE: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_16: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLK + tl.arange(0, BLK).to(tl.int64)
    mask = offs < numel
    rv = tl.load(res + offs, mask=mask)
    uv = tl.load(upd + offs, mask=mask)
    if MODE == 1:
        gv = tl.load(gate + (offs % hid), mask=mask)
    elif MODE == 2:
        gv = tl.load(gate + (offs // hid), mask=mask)
    else:
        gv = tl.load(gate + offs, mask=mask)
    if IS_16:
        value = rv.to(tl.float32) + _rocm_round16_f32(
            uv.to(tl.float32) * gv.to(tl.float32), IS_BF16
        )
        _rocm_store16(out, offs, value, mask, IS_BF16)
    else:
        tl.store(out + offs, rv + uv * gv, mask=mask)


@triton.jit
def _rga_rocm_transposed(
    out,
    res,
    upd,
    gate,
    tokens,
    hid: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_16: tl.constexpr,
    TILE: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)
    t = pid_t * TILE + tl.arange(0, TILE).to(tl.int64)
    h = pid_h * TILE + tl.arange(0, TILE).to(tl.int64)
    mt = t < tokens
    mh = h < hid
    gv = tl.load(gate + h, mask=mh, other=0.0)
    base = pid_b * tokens * hid
    mask = mh[:, None] & mt[None, :]
    roffs = h[:, None] * tokens + t[None, :]
    rv = tl.load(res + base + roffs, mask=mask, other=0.0)
    uv = tl.load(upd + base + t[None, :] * hid + h[:, None], mask=mask, other=0.0)
    if IS_16:
        value = rv.to(tl.float32) + _rocm_round16_f32(
            uv.to(tl.float32) * gv[:, None].to(tl.float32), IS_BF16
        )
        _rocm_store16(out, base + roffs, value, mask, IS_BF16)
    else:
        tl.store(out + base + roffs, rv + uv * gv[:, None], mask=mask)


def _residual_gate_add_rocm_triton(residual, update, gate):
    out = torch.empty_strided(
        residual.shape, residual.stride(), dtype=residual.dtype, device=residual.device
    )
    is_bf16 = residual.dtype == torch.bfloat16
    is_16 = residual.dtype in (torch.float16, torch.bfloat16)
    if _is_transposed_dense_residual(residual, update, gate):
        _, tokens, hidden = residual.shape
        # Tile/warp sizes reused from the CUDA transposed kernel; not
        # independently tuned or profiled on AMD GPUs.
        tile = (
            16
            if residual.numel() <= 65536
            else 64
            if residual.numel() >= 1 << 20
            else 32
        )
        warps = 4 if tile == 16 else 8
        _rga_rocm_transposed[
            (triton.cdiv(tokens, tile), triton.cdiv(hidden, tile), residual.shape[0])
        ](
            out,
            residual,
            update,
            gate,
            tokens,
            hidden,
            is_bf16,
            is_16,
            TILE=tile,
            num_warps=warps,
        )
        return out

    numel = residual.numel()
    hidden = residual.shape[-1]
    if gate.shape == residual.shape:
        mode = 0
    elif _is_row_broadcast_gate(residual, gate):
        mode = 1
    else:
        mode = 2
    block = 256 if numel <= 8192 else 1024
    _rga_rocm_flat[(triton.cdiv(numel, block),)](
        out,
        residual,
        update,
        gate,
        numel,
        hidden,
        mode,
        is_bf16,
        is_16,
        BLK=block,
        num_warps=4,
    )
    return out


@register_custom_op(
    op_name="diffusion_residual_gate_add",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _residual_gate_add_custom_op(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    with torch.cuda.device(residual.device):
        # Take the Triton path only for the transposed-dense layout it was
        # benchmarked on. For the contiguous layouts every other diffusion model
        # uses, the existing JIT CUDA kernel is faster at every measured shape.
        if _is_transposed_dense_residual(residual, update, gate):
            return _residual_gate_add_transposed(residual, update, gate)
        return _residual_gate_add_cuda_impl(residual, update, gate)


@register_custom_op(
    op_name="diffusion_residual_gate_add_rocm",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _residual_gate_add_rocm_custom_op(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    with torch.cuda.device(residual.device):
        return _residual_gate_add_rocm_triton(residual, update, gate)


def _gate_mode(residual: torch.Tensor, gate: torch.Tensor) -> int:
    """0 = full, 1 = broadcast row (hidden_size), 2 = per-token (rows)."""
    if gate.shape == residual.shape:
        return 0
    if _is_row_broadcast_gate(residual, gate):
        return 1
    return 2


def _is_row_broadcast_gate(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    if gate.dim() != residual.dim() or gate.shape[-1] != residual.shape[-1]:
        return False
    return all(size == 1 for size in gate.shape[:-1])


def _is_per_token_gate(residual: torch.Tensor, gate: torch.Tensor) -> bool:
    """Gate holds one scalar per token (row), broadcast along the hidden dim."""
    return (
        gate.dim() == residual.dim()
        and gate.shape[-1] == 1
        and gate.shape[:-1] == residual.shape[:-1]
    )


def _is_transposed_dense_residual(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    if residual.dim() != 3 or gate.shape != (1, 1, residual.shape[-1]):
        return False
    batch, tokens, hidden_size = residual.shape
    return (
        batch <= _MAX_GRID_DIM
        and (tokens + _TRANSPOSE_TILE - 1) // _TRANSPOSE_TILE <= _MAX_GRID_DIM
        and (hidden_size + _TRANSPOSE_TILE - 1) // _TRANSPOSE_TILE <= _MAX_GRID_DIM
        and residual.stride() == (tokens * hidden_size, 1, tokens)
        and update.is_contiguous()
        and gate.is_contiguous()
    )


def _can_use_residual_gate_add_common(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    return (
        residual.dtype in _SUPPORTED_DTYPES
        and residual.dtype == update.dtype
        and residual.dtype == gate.dtype
        and residual.is_cuda
        and update.is_cuda
        and gate.is_cuda
        and residual.device == update.device == gate.device
        and residual.dim() >= 2
        and residual.numel() > 0
        and update.shape == residual.shape
        and (
            gate.shape == residual.shape
            or _is_row_broadcast_gate(residual, gate)
            or _is_per_token_gate(residual, gate)
        )
        and (
            (residual.is_contiguous() and update.is_contiguous())
            or _is_transposed_dense_residual(residual, update, gate)
        )
        and gate.is_contiguous()
    )


def can_use_residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Return whether the NVIDIA-only JIT/Triton dispatch can handle the input."""
    return torch.version.hip is None and _can_use_residual_gate_add_common(
        residual, update, gate
    )


def can_use_residual_gate_add_rocm(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    """Return whether the dedicated ROCm Triton path can handle the input."""
    return torch.version.hip is not None and _can_use_residual_gate_add_common(
        residual, update, gate
    )


def residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if not can_use_residual_gate_add_cuda(residual, update, gate):
        raise RuntimeError("unsupported input for residual_gate_add CUDA")
    return _residual_gate_add_custom_op(residual, update, gate)


def residual_gate_add_rocm(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if not can_use_residual_gate_add_rocm(residual, update, gate):
        raise RuntimeError("unsupported input for residual_gate_add ROCm")
    return _residual_gate_add_rocm_custom_op(residual, update, gate)


def residual_gate_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Use the backend's bit-exact fast path for supported layouts, otherwise eager."""
    if residual.dtype in _BIT_EXACT_DTYPES:
        if can_use_residual_gate_add_cuda(residual, update, gate):
            return _residual_gate_add_custom_op(residual, update, gate)
        if can_use_residual_gate_add_rocm(residual, update, gate):
            return _residual_gate_add_rocm_custom_op(residual, update, gate)
    return residual + update * gate


__all__ = [
    "can_use_residual_gate_add_cuda",
    "can_use_residual_gate_add_rocm",
    "residual_gate_add",
    "residual_gate_add_cuda",
    "residual_gate_add_rocm",
]
