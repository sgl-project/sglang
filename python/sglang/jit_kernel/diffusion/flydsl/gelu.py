"""FlyDSL GeLU kernel: vectorized pointwise tanh-approximation GeLU for AMD ROCm.

Bandwidth-optimized replacement for Inductor-generated GeLU using
vec-8 buffer_load/store for higher memory throughput on gfx950.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_ops
from flydsl._mlir.dialects import math as math_ops
from flydsl._mlir.dialects import (
    scf,
)
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import Int32, T

WARP_SIZE = 64

_GELU_VEC = 8
_GELU_NUM_WAVES = 4
_GELU_BLOCK = _GELU_NUM_WAVES * WARP_SIZE  # 256 threads
_GELU_ELEMS_PER_BLOCK = _GELU_BLOCK * _GELU_VEC  # 2048 elements


def _v(x):
    return buffer_ops._unwrap_value(x)


def _build_gelu_tanh_module():
    VEC = _GELU_VEC
    BLOCK = _GELU_BLOCK
    EPB = _GELU_ELEMS_PER_BLOCK

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flydsl_gelu_tanh_kernel(
        out_ptr: fx.Tensor,
        in_ptr: fx.Tensor,
        numel: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        i32 = T.i32
        f32 = T.f32
        bf16 = T.bf16
        vec_f32_t = ir.VectorType.get([VEC], f32)
        vec_bf16_t = ir.VectorType.get([VEC], bf16)

        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        in_rsrc = buffer_ops.create_buffer_resource(in_ptr, max_size=True)

        bid_i32 = ArithValue(bid)
        tid_i32 = ArithValue(tid)

        off = bid_i32 * arith.constant(EPB, type=i32) + tid_i32 * arith.constant(
            VEC, type=i32
        )

        c_c1 = _v(arith.constant(-2.0 * 0.035677, type=f32))
        c_c2 = _v(arith.constant(-2.0 * 0.797885, type=f32))
        c_one = _v(arith.constant(1.0, type=f32))

        c1_splat = _vector.BroadcastOp(vec_f32_t, c_c1).result
        c2_splat = _vector.BroadcastOp(vec_f32_t, c_c2).result
        one_splat = _vector.BroadcastOp(vec_f32_t, c_one).result

        in_bounds = arith.cmpi(CmpIPredicate.slt, off, ArithValue(numel))
        _if = scf.IfOp(in_bounds)
        with ir.InsertionPoint(_if.then_block):
            x_vec = buffer_ops.buffer_load(in_rsrc, off, vec_width=VEC, dtype=bf16)
            x_f32 = arith_ops.ExtFOp(vec_f32_t, _v(x_vec)).result

            x_sq = arith_ops.MulFOp(x_f32, x_f32).result
            c1x2 = arith_ops.MulFOp(x_sq, c1_splat).result
            inner = arith_ops.AddFOp(c1x2, c2_splat).result
            u = arith_ops.MulFOp(x_f32, inner).result

            exp_u = math_ops.ExpOp(u).result
            denom = arith_ops.AddFOp(one_splat, exp_u).result
            result = arith_ops.DivFOp(x_f32, denom).result

            out_bf16 = arith_ops.TruncFOp(vec_bf16_t, result).result
            buffer_ops.buffer_store(out_bf16, out_rsrc, off)
            scf.YieldOp([])

    @flyc.jit
    def launch_gelu(
        out: fx.Tensor,
        inp: fx.Tensor,
        numel: fx.Int32,
        num_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        grid_x = arith.index_cast(T.index, num_blocks)
        launcher = flydsl_gelu_tanh_kernel(out, inp, numel)
        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), smem=0, stream=stream)

    return launch_gelu


# ---------------------------------------------------------------------------
# Compilation cache & custom_op
# ---------------------------------------------------------------------------
_GELU_COMPILE_CACHE = {}


def _get_or_compile_gelu(args):
    key = "gelu_tanh"
    if key not in _GELU_COMPILE_CACHE:
        launcher = _build_gelu_tanh_module()
        cf = flyc.compile(launcher, *args)
        _GELU_COMPILE_CACHE[key] = cf
    return _GELU_COMPILE_CACHE[key]


@torch.library.custom_op("sglang::flydsl_gelu_tanh", mutates_args=())
def flydsl_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f"Expected bf16 input, got {x.dtype}"
    x_flat = x.contiguous().view(-1)
    numel = x_flat.numel()
    y_flat = torch.empty_like(x_flat)

    EPB = _GELU_ELEMS_PER_BLOCK
    num_blocks = (numel + EPB - 1) // EPB
    stream = torch.cuda.current_stream()

    dummy_args = (y_flat, x_flat, numel, num_blocks, stream)
    cf = _get_or_compile_gelu(dummy_args)
    cf(y_flat, x_flat, numel, num_blocks, stream)

    return y_flat.view(x.shape)


@flydsl_gelu_tanh.register_fake
def _fake_gelu_tanh(x):
    return torch.empty_like(x)
