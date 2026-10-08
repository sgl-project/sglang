"""BF16 QKV projection, Q/K RMSNorm, MRoPE, and HND cache write."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
    SplitKDenseGemmKernel,
    SplitKTactic,
    _align_up,
    _smem_bytes,
    _to_cute_swap,
    validate_tactic,
)


@dsl_user_op
def _norm_factor(x0, x1, x2, x3, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value() for x in (x0, x1, x2, x3)],
            "{ .reg .f32 s,t; mul.rn.f32 s,$1,$1; fma.rn.f32 s,$2,$2,s; fma.rn.f32 s,$3,$3,s; fma.rn.f32 s,$4,$4,s; "
            "shfl.sync.bfly.b32 t,s,16,31,-1; add.rn.f32 s,s,t; shfl.sync.bfly.b32 t,s,8,31,-1; add.rn.f32 s,s,t; "
            "shfl.sync.bfly.b32 t,s,4,31,-1; add.rn.f32 s,s,t; shfl.sync.bfly.b32 t,s,2,31,-1; add.rn.f32 s,s,t; "
            "shfl.sync.bfly.b32 t,s,1,31,-1; add.rn.f32 s,s,t; mul.rn.f32 s,s,0f3C000000; add.rn.f32 s,s,0f358637BD; rsqrt.approx.ftz.f32 $0,s; }",
            "=f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _norm_value(x, scale, w, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value() for v in (x, scale, w)],
            "{ .reg .f32 t; mul.rn.f32 t,$1,$2; mul.rn.f32 $0,t,$3; }",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _rotate(x, y, c, s, upper, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value() for v in (x, y, c, s, upper)],
            "{ .reg .b16 a,b,c,d,t,o; .reg .pred p; cvt.rn.bf16.f32 a,$1; cvt.rn.bf16.f32 b,$2; cvt.rn.bf16.f32 c,$3; cvt.rn.bf16.f32 d,$4; "
            "setp.ne.s32 p,$5,0; @!p mul.rn.bf16 t,b,d; @!p neg.bf16 t,t; @!p fma.rn.bf16 o,a,c,t; "
            "@p mul.rn.bf16 t,a,c; @p fma.rn.bf16 o,b,d,t; cvt.f32.bf16 $0,o; }",
            "=f,f,f,f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class _QKVNormMRopeEpilogue:
    def __init__(self, num_tokens):
        self.iterations = num_tokens // 4

    @cute.experimental.jit
    def allocate_scratch(self, dtype: cutlass.Constexpr):
        return cute_ext.allocate(
            dtype,
            cute.AddressSpace.smem,
            cute.make_layout((128, 8), stride=(1, 128)),
            alignment=128,
        )

    @cute.experimental.jit
    def store(self, rD, thr_t2r, gD_tile, epi_tid, head, n_idx, sE, ep):
        c_dtype = rD.element_type
        # Redistribute the rounded GEMM result, preserving BF16 before norm.
        cute_ext.partition_and_copy(thr_t2r, rD, sE)
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        lane = epi_tid % 32
        warp = epi_tid // 32
        qw, kw, rope, positions, axes, kc, vc, slots = ep
        for iteration in cutlass.range_constexpr(self.iterations):
            token = warp + iteration * 4
            d = lane * 4
            x0 = sE[d, token].to(cutlass.Float32)
            x1 = sE[d + 1, token].to(cutlass.Float32)
            x2 = sE[d + 2, token].to(cutlass.Float32)
            x3 = sE[d + 3, token].to(cutlass.Float32)
            scale = cutlass.Float32(1.0)
            if head < 40:
                scale = _norm_factor(x0, x1, x2, x3)
            for component in cutlass.range_constexpr(4):
                dim = d + component
                value = sE[dim, token].to(cutlass.Float32)
                if head < 40:
                    weight = cutlass.Float32(0.0)
                    if head < 32:
                        weight = qw[dim].to(cutlass.Float32)
                    else:
                        weight = kw[dim].to(cutlass.Float32)
                    normalized = (
                        _norm_value(value, scale, weight)
                        .to(c_dtype)
                        .to(cutlass.Float32)
                    )
                    partner = cute.arch.shuffle_sync_bfly(normalized, 16)
                    pair_dim = dim % 64
                    pos = positions[axes[pair_dim], token]
                    cosine = rope[pos, pair_dim].to(cutlass.Float32)
                    sine = rope[pos, pair_dim + 64].to(cutlass.Float32)
                    value = _rotate(
                        normalized, partner, cosine, sine, cutlass.Int32(lane >= 16)
                    )
                rounded = value.to(c_dtype)
                gD_tile[dim, token] = rounded
                slot = slots[token]
                if head >= 32 and slot >= 0:
                    if head < 40:
                        kc[slot // 32, head - 32, slot % 32, dim] = rounded
                    else:
                        vc[slot // 32, head - 40, slot % 32, dim] = rounded


@cute.experimental.jit
def _run(kernel: cutlass.Constexpr, a, b, c, ep, stream: cuda.CUstream):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    kernel(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        c,
        c,
        stream,
        ep,
    )


_COMPILED = {}
_TACTIC = SplitKTactic(128, 8, 2, 6)


def qkv_norm_mrope(
    x,
    weight,
    q_weight,
    k_weight,
    rope,
    positions,
    axes,
    key_cache,
    value_cache,
    slots,
    *,
    out=None,
):
    """Write exact M4/M8 BF16 QKV and HND page32 caches, or return None if unsupported.

    Slots must be valid physical cache locations or negative padding sentinels.
    Positions and axis values must index the supplied rotary table. Buffers must
    not overlap inputs or each other. Compilation occurs only outside capture.
    """
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or tuple(x.shape) not in ((4, 2560), (8, 2560))
    ):
        return None
    m = x.shape[0]
    tensors = (
        x,
        weight,
        q_weight,
        k_weight,
        rope,
        positions,
        axes,
        key_cache,
        value_cache,
        slots,
    )
    if (
        tuple(weight.shape) != (6144, 2560)
        or not x.is_cuda
        or torch.cuda.current_device() != x.device.index
        or torch.cuda.get_device_capability(x.device) != (10, 3)
        or any(t.device != x.device or t.requires_grad for t in tensors)
        or any(not t.is_contiguous() for t in tensors if t is not positions)
        or any(t.data_ptr() % 8 for t in tensors)
        or x.data_ptr() % 32
        or weight.data_ptr() % 32
        or any(
            t.dtype != torch.bfloat16
            for t in (x, weight, q_weight, k_weight, rope, key_cache, value_cache)
        )
        or tuple(q_weight.shape) != (128,)
        or tuple(k_weight.shape) != (128,)
        or rope.ndim != 2
        or rope.shape[1] != 128
        or tuple(positions.shape) != (3, m)
        or positions.dtype != torch.int64
        or positions.stride(1) != 1
        or positions.stride(0) < m
        or tuple(axes.shape) != (64,)
        or axes.dtype != torch.int64
        or tuple(slots.shape) != (m,)
        or slots.dtype != torch.int64
        or key_cache.ndim != 4
        or tuple(key_cache.shape[1:]) != (8, 32, 128)
        or value_cache.shape != key_cache.shape
    ):
        return None
    if out is not None and (
        tuple(out.shape) != (m, 6144)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
        or out.requires_grad
        or out.data_ptr() % 32
    ):
        return None
    outputs = (key_cache, value_cache) + (() if out is None else (out,))
    inputs = (x, weight, q_weight, k_weight, rope, positions, axes, slots)
    for index, target in enumerate(outputs):
        begin = target.data_ptr()
        end = begin + target.numel() * target.element_size()
        for other in inputs + outputs[:index]:
            other_begin = other.data_ptr()
            other_span = 1 + sum(
                (size - 1) * stride for size, stride in zip(other.shape, other.stride())
            )
            if (
                begin < other_begin + other_span * other.element_size()
                and other_begin < end
            ):
                return None
    ep = (q_weight, k_weight, rope, positions, axes, key_cache, value_cache, slots)
    key = (
        x.device.index,
        tuple((t.dtype, tuple(t.shape), t.stride()) for t in tensors),
    )
    compiled = _COMPILED.get(key)
    if compiled is None and torch.cuda.is_current_stream_capturing():
        return None
    if out is None:
        out = torch.empty((m, 6144), dtype=x.dtype, device=x.device)
    args = _to_cute_swap(x, weight.T, out, None)
    extra = tuple(from_dlpack(t, assumed_align=8) for t in ep)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    if compiled is None:
        validate_tactic(_TACTIC, m, 6144, 2560)
        assert _align_up(_smem_bytes(_TACTIC, 6), 128) + 2048 <= 227 * 1024
        kernel = SplitKDenseGemmKernel(
            tactic=_TACTIC,
            use_pdl=True,
            has_bias=False,
            epilogue_hook=_QKVNormMRopeEpilogue(m),
        )
        compiled = cute_ext.compile(_run, kernel, *args[:3], extra, stream)
        _COMPILED[key] = compiled
    compiled(*args[:3], extra, stream)
    return out
