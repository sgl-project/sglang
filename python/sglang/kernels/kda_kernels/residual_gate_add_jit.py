from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.kda_kernels import _cuda_source
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BIT_EXACT_DTYPES = (torch.float16, torch.bfloat16)
_TRANSPOSE_TILE = 32
_MAX_GRID_DIM = 65535
_FAILED_RUNTIME_KEYS: set[tuple[int | None, torch.dtype]] = set()
# HIP tensors also report is_cuda, but this kernel only matches eager CUDA
# rounding exactly.
_IS_HIP = torch.version.hip is not None

logger = logging.getLogger(__name__)


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
            (
                "residual_gate_add_transposed",
                f"residual_gate_add::ResidualGateAddKernel<{args}>::run_transposed",
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
    if _is_transposed_dense_residual(residual, update, gate):
        module.residual_gate_add_transposed(out, residual, update, gate)
        return out
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
def _round16_f32(x, IS_BF16: tl.constexpr):
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
def _store16(out_ptr, offs, value_f32, mask, IS_BF16: tl.constexpr):
    if IS_BF16:
        bits = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;",
            "=h,r",
            [value_f32],
            dtype=tl.int16,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offs, bits.to(tl.bfloat16, bitcast=True), mask=mask)
    else:
        bits = tl.inline_asm_elementwise(
            "cvt.rn.f16.f32 $0, $1;",
            "=h,r",
            [value_f32],
            dtype=tl.int16,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offs, bits.to(tl.float16, bitcast=True), mask=mask)


@triton.jit
def _rga_flat(
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
        p32 = uv.to(tl.float32) * gv.to(tl.float32)
        pf = _round16_f32(p32, IS_BF16)
        o32 = rv.to(tl.float32) + pf
        _store16(out, offs, o32, mask, IS_BF16)
    else:
        p32 = uv * gv
        o32 = rv + p32
        tl.store(out + offs, o32, mask=mask)


@triton.jit
def _rga_row(
    out,
    res,
    upd,
    gate,
    hid: tl.constexpr,
    IS_BF16: tl.constexpr,
    EVG: tl.constexpr,
    CG: tl.constexpr,
    BLK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLK)
    mask = offs < hid
    if EVG:
        gv = tl.load(gate + offs, mask=mask, eviction_policy="evict_last")
    else:
        gv = tl.load(gate + offs, mask=mask)
    base = row * hid
    if CG:
        rv = tl.load(res + base + offs, mask=mask, cache_modifier=".cg")
        uv = tl.load(upd + base + offs, mask=mask, cache_modifier=".cg")
    else:
        rv = tl.load(res + base + offs, mask=mask, eviction_policy="evict_first")
        uv = tl.load(upd + base + offs, mask=mask, eviction_policy="evict_first")
    p32 = uv.to(tl.float32) * gv.to(tl.float32)
    pf = _round16_f32(p32, IS_BF16)
    o32 = rv.to(tl.float32) + pf
    _store16(out, base + offs, o32, mask, IS_BF16)


@triton.jit
def _rga_flat_m1(
    out,
    res,
    upd,
    gate,
    numel,
    hid: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLK + tl.arange(0, BLK).to(tl.int64)
    mask = offs < numel
    rv = tl.load(res + offs, mask=mask, eviction_policy="evict_first")
    uv = tl.load(upd + offs, mask=mask, eviction_policy="evict_first")
    gv = tl.load(gate + (offs % hid), mask=mask, eviction_policy="evict_last")
    p32 = uv.to(tl.float32) * gv.to(tl.float32)
    pf = _round16_f32(p32, IS_BF16)
    o32 = rv.to(tl.float32) + pf
    _store16(out, offs, o32, mask, IS_BF16)


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
        pf = _round16_f32(p32, IS_BF16)
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


def _residual_gate_add_triton(residual, update, gate):
    # Both callers validate the inputs first; repeating the predicate here cost
    # ~2.7us of host time on every call.
    out = torch.empty_strided(
        residual.shape, residual.stride(), dtype=residual.dtype, device=residual.device
    )
    is_bf16 = residual.dtype == torch.bfloat16
    dtype16 = residual.dtype in (torch.float16, torch.bfloat16)

    if _is_transposed_dense_residual(residual, update, gate):
        tokens = residual.shape[1]
        hid = residual.shape[2]
        batch = residual.shape[0]
        if tokens * hid * batch <= 65536:
            tile, warps = 16, 4
        elif tokens * hid * batch >= 1 << 20:
            tile, warps = 64, 8
        else:
            tile, warps = 32, 8
        grid = (triton.cdiv(tokens, tile), triton.cdiv(hid, tile), batch)
        _rga_transposed[grid](
            out,
            residual,
            update,
            gate,
            tokens,
            hid,
            is_bf16,
            dtype16,
            TILE=tile,
            num_warps=warps,
        )
        return out

    if not dtype16:
        # fp32: exact fp32 math, plain Triton elementwise
        numel = residual.numel()
        hid = residual.shape[-1]
        if gate.shape == residual.shape:
            mode = 0
        elif _is_row_broadcast_gate(residual, gate):
            mode = 1
        else:
            mode = 2
        if numel <= 8192:
            _rga_flat[(triton.cdiv(numel, 256),)](
                out,
                residual,
                update,
                gate,
                numel,
                hid,
                mode,
                False,
                False,
                BLK=256,
                num_warps=4,
            )
        else:
            _rga_flat[(triton.cdiv(numel, 1024),)](
                out,
                residual,
                update,
                gate,
                numel,
                hid,
                mode,
                False,
                False,
                BLK=1024,
                num_warps=4,
            )
        return out

    hid = residual.shape[-1]
    numel = residual.numel()
    if _is_row_broadcast_gate(residual, gate):
        rows = numel // hid
        if hid >= 4096 or (rows <= 512 and numel >= 1 << 20):
            _rga_flat_m1[(triton.cdiv(numel, 1024),)](
                out, residual, update, gate, numel, hid, is_bf16, BLK=1024, num_warps=4
            )
        elif hid <= 16384:
            blk = triton.next_power_of_2(hid)
            _rga_row[(rows,)](
                out,
                residual,
                update,
                gate,
                hid,
                is_bf16,
                hid < 4608,
                hid >= 4096,
                BLK=blk,
                num_warps=4 if hid >= 4608 else 8,
            )
        else:
            _rga_flat[(triton.cdiv(numel, 1024),)](
                out,
                residual,
                update,
                gate,
                numel,
                hid,
                1,
                is_bf16,
                True,
                BLK=1024,
                num_warps=4,
            )
        return out
    mode = 0 if gate.shape == residual.shape else 2
    if numel <= 8192:
        _rga_flat[(triton.cdiv(numel, 256),)](
            out,
            residual,
            update,
            gate,
            numel,
            hid,
            mode,
            is_bf16,
            True,
            BLK=256,
            num_warps=4,
        )
    else:
        _rga_flat[(triton.cdiv(numel, 1024),)](
            out,
            residual,
            update,
            gate,
            numel,
            hid,
            mode,
            is_bf16,
            True,
            BLK=1024,
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
            return _residual_gate_add_triton(residual, update, gate)
        return _residual_gate_add_cuda_impl(residual, update, gate)


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


def can_use_residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> bool:
    return (
        residual.dtype in _SUPPORTED_DTYPES
        and residual.dtype == update.dtype
        and residual.dtype == gate.dtype
        and residual.is_cuda
        and not _IS_HIP
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


def residual_gate_add_cuda(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if not can_use_residual_gate_add_cuda(residual, update, gate):
        raise RuntimeError("unsupported input for residual_gate_add CUDA")
    return _residual_gate_add_custom_op(residual, update, gate)


def residual_gate_add(
    residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Use the bit-exact CUDA fast path when supported, otherwise eager.

    Runtime build failures are cached per device and dtype so every diffusion
    model shares one fallback policy instead of maintaining model-local flags.
    """
    runtime_key = (residual.device.index, residual.dtype)
    if (
        residual.dtype in _BIT_EXACT_DTYPES
        and runtime_key not in _FAILED_RUNTIME_KEYS
        and can_use_residual_gate_add_cuda(residual, update, gate)
    ):
        try:
            return _residual_gate_add_custom_op(residual, update, gate)
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_KEYS.add(runtime_key)
            logger.warning(
                "Disabling diffusion residual-gate CUDA fast path on %s/%s: %s",
                residual.device,
                residual.dtype,
                exc,
            )
    return residual + update * gate


__all__ = [
    "can_use_residual_gate_add_cuda",
    "residual_gate_add",
    "residual_gate_add_cuda",
]
