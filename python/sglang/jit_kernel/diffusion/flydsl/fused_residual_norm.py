"""FlyDSL fused normalization kernels for AMD ROCm (gfx950).

Provides two fused kernels:
  - flydsl_fused_residual_norm_scale_shift:
        residual_add + gate_mul + RMSNorm/LayerNorm + scale·shift
  - flydsl_norm_scale_shift:
        RMSNorm/LayerNorm + scale·shift

Both kernels use register-cache optimization: Phase 2 (scale·shift)
reuses f32 intermediate values from Phase 1 (norm) registers instead
of re-reading from HBM, saving ~20% bandwidth.
"""

from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_ops
from flydsl._mlir.dialects import gpu as _gpu
from flydsl._mlir.dialects import math as math_ops
from flydsl._mlir.dialects import memref as _memref
from flydsl._mlir.dialects import (
    scf,
)
from flydsl._mlir.dialects import vector as _vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import Int32, T

WARP_SIZE = 64
_VEC = 8
_NUM_WAVES = 10
FLYDSL_NORM_MIN_ALIGNED_DIM = WARP_SIZE * _NUM_WAVES * _VEC  # 5120


def _v(x):
    return buffer_ops._unwrap_value(x)


def _build_fused_norm_module(D: int, is_rms: bool, has_gate: bool, has_weight: bool):
    VEC = _VEC
    NUM_WAVES = _NUM_WAVES
    BLOCK = NUM_WAVES * WARP_SIZE
    assert (
        D % FLYDSL_NORM_MIN_ALIGNED_DIM == 0
    ), f"FlyDSL fused_residual_norm requires D % {FLYDSL_NORM_MIN_ALIGNED_DIM} == 0, got D={D}"
    NUM_ITERS = D // (BLOCK * VEC)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flydsl_fused_residual_norm_ss_kernel(
        y_ptr: fx.Tensor,
        res_out_ptr: fx.Tensor,
        res_ptr: fx.Tensor,
        x_ptr: fx.Tensor,
        gate_ptr: fx.Tensor,
        weight_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        scale_ptr: fx.Tensor,
        shift_ptr: fx.Tensor,
        total_rows: Int32,
        gate_stride: Int32,
        scale_stride: Int32,
        shift_stride: Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        i32 = T.i32
        f32 = T.f32
        bf16 = T.bf16
        vec_f32_t = ir.VectorType.get([VEC], f32)
        vec_bf16_t = ir.VectorType.get([VEC], bf16)

        y_rsrc = buffer_ops.create_buffer_resource(y_ptr, max_size=True)
        ro_rsrc = buffer_ops.create_buffer_resource(res_out_ptr, max_size=True)
        r_rsrc = buffer_ops.create_buffer_resource(res_ptr, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(x_ptr, max_size=True)
        g_rsrc = buffer_ops.create_buffer_resource(gate_ptr, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weight_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(bias_ptr, max_size=True)
        sc_rsrc = buffer_ops.create_buffer_resource(scale_ptr, max_size=True)
        sh_rsrc = buffer_ops.create_buffer_resource(shift_ptr, max_size=True)

        row_i32 = ArithValue(row)
        tid_i32 = ArithValue(tid)
        D_i32 = arith.constant(D, type=i32)
        row_off = row_i32 * D_i32
        gate_row_off = row_i32 * ArithValue(gate_stride)
        scale_row_off = row_i32 * ArithValue(scale_stride)
        shift_row_off = row_i32 * ArithValue(shift_stride)

        c_zero_f32 = arith.constant(0.0, type=f32)
        c_one_f32 = arith.constant(1.0, type=f32)
        eps_val = arith.constant(1e-6, type=f32)
        D_float = arith.constant(float(D), type=f32)

        # LDS
        LDS_SLOTS = NUM_WAVES * 2 + 2
        ws_attr = ir.Attribute.parse("#gpu.address_space<workgroup>")
        lds_i8_type = ir.MemRefType.get(
            [ir.ShapedType.get_dynamic_size()], T.i8, memory_space=ws_attr
        )
        lds_f32_type = ir.MemRefType.get([LDS_SLOTS], f32, memory_space=ws_attr)
        lds_i8 = _gpu.DynamicSharedMemoryOp(lds_i8_type).result
        byte_zero = arith_ops.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
        ).result
        lds = _memref.ViewOp(lds_f32_type, lds_i8, byte_zero, []).result

        lane_id = tid_i32 % arith.constant(WARP_SIZE, type=i32)
        wave_id = tid_i32 // arith.constant(WARP_SIZE, type=i32)

        # Phase 1: residual + gate*x, accumulate stats, save f32 in registers
        _saved_ro_f32 = []
        partial_sum = _v(c_zero_f32)
        partial_sum_sq = _v(c_zero_f32)

        for it in range_constexpr(NUM_ITERS):
            col = tid_i32 * arith.constant(VEC, type=i32) + arith.constant(
                it * BLOCK * VEC, type=i32
            )
            off = row_off + col

            r_vec = buffer_ops.buffer_load(r_rsrc, off, vec_width=VEC, dtype=bf16)
            x_vec = buffer_ops.buffer_load(x_rsrc, off, vec_width=VEC, dtype=bf16)
            r_f32 = arith_ops.ExtFOp(vec_f32_t, _v(r_vec)).result
            x_f32 = arith_ops.ExtFOp(vec_f32_t, _v(x_vec)).result

            if const_expr(has_gate):
                g_off = gate_row_off + col
                g_vec = buffer_ops.buffer_load(g_rsrc, g_off, vec_width=VEC, dtype=bf16)
                g_f32 = arith_ops.ExtFOp(vec_f32_t, _v(g_vec)).result
                gx = arith_ops.MulFOp(g_f32, x_f32).result
                ro_f32 = arith_ops.AddFOp(r_f32, gx).result
            else:
                ro_f32 = arith_ops.AddFOp(r_f32, x_f32).result

            ro_bf16 = arith_ops.TruncFOp(vec_bf16_t, ro_f32).result
            buffer_ops.buffer_store(ro_bf16, ro_rsrc, off)
            _saved_ro_f32.append(ro_f32)

            if const_expr(not is_rms):
                v_sum = _vector.ReductionOp(
                    f32, _vector.CombiningKind.ADD, ro_f32
                ).result
                partial_sum = arith_ops.AddFOp(partial_sum, v_sum).result
            ro_sq = arith_ops.MulFOp(ro_f32, ro_f32).result
            v_sum_sq = _vector.ReductionOp(f32, _vector.CombiningKind.ADD, ro_sq).result
            partial_sum_sq = arith_ops.AddFOp(partial_sum_sq, v_sum_sq).result

        # Intra-wave shuffle
        width_c = _v(arith.constant(WARP_SIZE, type=i32))
        w_sum = partial_sum
        w_sq = partial_sum_sq
        for sh in [32, 16, 8, 4, 2, 1]:
            off_sh = _v(arith.constant(sh, type=i32))
            if const_expr(not is_rms):
                peer_sum = _gpu.ShuffleOp(
                    w_sum, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
                ).shuffleResult
                w_sum = arith_ops.AddFOp(w_sum, peer_sum).result
            peer_sq = _gpu.ShuffleOp(
                w_sq, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
            ).shuffleResult
            w_sq = arith_ops.AddFOp(w_sq, peer_sq).result

        # Cross-wave LDS
        lane_0 = arith.cmpi(CmpIPredicate.eq, lane_id, arith.constant(0, type=i32))
        wave_idx = arith_ops.IndexCastOp(ir.IndexType.get(), _v(wave_id)).result

        _if_lane0 = scf.IfOp(lane_0)
        with ir.InsertionPoint(_if_lane0.then_block):
            if const_expr(not is_rms):
                _memref.StoreOp(w_sum, lds, [wave_idx])
            sq_slot = arith_ops.AddIOp(
                wave_idx,
                arith_ops.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES),
                ).result,
            ).result
            _memref.StoreOp(w_sq, lds, [sq_slot])
            scf.YieldOp([])
        _gpu.BarrierOp()

        wave_0 = arith.cmpi(CmpIPredicate.eq, wave_id, arith.constant(0, type=i32))
        active = arith.andi(
            wave_0,
            arith.cmpi(CmpIPredicate.ult, lane_id, arith.constant(NUM_WAVES, type=i32)),
        )
        lane_idx = arith_ops.IndexCastOp(ir.IndexType.get(), _v(lane_id)).result
        lane_idx_sq = arith_ops.AddIOp(
            lane_idx,
            arith_ops.ConstantOp(
                ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES)
            ).result,
        ).result

        if const_expr(is_rms):
            _if_active = scf.IfOp(active, [f32], has_else=True)
            with ir.InsertionPoint(_if_active.then_block):
                sq_val = _memref.LoadOp(lds, [lane_idx_sq]).result
                scf.YieldOp([sq_val])
            with ir.InsertionPoint(_if_active.else_block):
                scf.YieldOp([_v(c_zero_f32)])
            loaded_sq = _if_active.results[0]
            loaded_sum = _v(c_zero_f32)
        else:
            _if_active = scf.IfOp(active, [f32, f32], has_else=True)
            with ir.InsertionPoint(_if_active.then_block):
                s_val = _memref.LoadOp(lds, [lane_idx]).result
                sq_val = _memref.LoadOp(lds, [lane_idx_sq]).result
                scf.YieldOp([s_val, sq_val])
            with ir.InsertionPoint(_if_active.else_block):
                scf.YieldOp([_v(c_zero_f32), _v(c_zero_f32)])
            loaded_sum = _if_active.results[0]
            loaded_sq = _if_active.results[1]

        final_sum = loaded_sum
        final_sq = loaded_sq
        for sh in [32, 16, 8, 4, 2, 1]:
            off_sh = _v(arith.constant(sh, type=i32))
            if const_expr(not is_rms):
                ps = _gpu.ShuffleOp(
                    final_sum, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
                ).shuffleResult
                final_sum = arith_ops.AddFOp(final_sum, ps).result
            pq = _gpu.ShuffleOp(
                final_sq, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
            ).shuffleResult
            final_sq = arith_ops.AddFOp(final_sq, pq).result

        both_0 = arith.andi(wave_0, lane_0)
        final_sum_slot = arith_ops.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES * 2)
        ).result
        final_sq_slot = arith_ops.ConstantOp(
            ir.IndexType.get(),
            ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES * 2 + 1),
        ).result
        _if_both = scf.IfOp(both_0)
        with ir.InsertionPoint(_if_both.then_block):
            if const_expr(not is_rms):
                _memref.StoreOp(final_sum, lds, [final_sum_slot])
            _memref.StoreOp(final_sq, lds, [final_sq_slot])
            scf.YieldOp([])
        _gpu.BarrierOp()

        if const_expr(not is_rms):
            total_sum = _memref.LoadOp(lds, [final_sum_slot]).result
        else:
            total_sum = _v(c_zero_f32)
        total_sq = _memref.LoadOp(lds, [final_sq_slot]).result

        # Norm
        d_f = _v(D_float)
        eps_v = _v(eps_val)
        if const_expr(is_rms):
            var = arith_ops.DivFOp(total_sq, d_f).result
            var_eps = arith_ops.AddFOp(var, eps_v).result
            rstd = math_ops.RsqrtOp(var_eps).result
            mean = _v(c_zero_f32)
        else:
            mean = arith_ops.DivFOp(total_sum, d_f).result
            mean_sq = arith_ops.MulFOp(mean, mean).result
            var = arith_ops.SubFOp(
                arith_ops.DivFOp(total_sq, d_f).result, mean_sq
            ).result
            var_eps = arith_ops.AddFOp(var, eps_v).result
            rstd = math_ops.RsqrtOp(var_eps).result

        # Phase 2: normalize using register-cached f32 values (no HBM re-read)
        mean_splat = _vector.BroadcastOp(vec_f32_t, mean).result
        rstd_splat = _vector.BroadcastOp(vec_f32_t, rstd).result
        one_splat = _vector.BroadcastOp(vec_f32_t, _v(c_one_f32)).result

        for it in range_constexpr(NUM_ITERS):
            col = tid_i32 * arith.constant(VEC, type=i32) + arith.constant(
                it * BLOCK * VEC, type=i32
            )
            off = row_off + col

            ro_f32 = _saved_ro_f32[it]

            if const_expr(is_rms):
                x_hat = arith_ops.MulFOp(ro_f32, rstd_splat).result
            else:
                centered = arith_ops.SubFOp(ro_f32, mean_splat).result
                x_hat = arith_ops.MulFOp(centered, rstd_splat).result

            if const_expr(has_weight):
                w_vec = buffer_ops.buffer_load(w_rsrc, col, vec_width=VEC, dtype=bf16)
                w_f32 = arith_ops.ExtFOp(vec_f32_t, _v(w_vec)).result
                x_hat = arith_ops.MulFOp(x_hat, w_f32).result
                b_vec = buffer_ops.buffer_load(b_rsrc, col, vec_width=VEC, dtype=bf16)
                b_f32 = arith_ops.ExtFOp(vec_f32_t, _v(b_vec)).result
                x_hat = arith_ops.AddFOp(x_hat, b_f32).result

            sc_off = scale_row_off + col
            sc_vec = buffer_ops.buffer_load(sc_rsrc, sc_off, vec_width=VEC, dtype=bf16)
            sc_f32 = arith_ops.ExtFOp(vec_f32_t, _v(sc_vec)).result
            sc_p1 = arith_ops.AddFOp(one_splat, sc_f32).result
            x_hat = arith_ops.MulFOp(x_hat, sc_p1).result

            sh_off = shift_row_off + col
            sh_vec = buffer_ops.buffer_load(sh_rsrc, sh_off, vec_width=VEC, dtype=bf16)
            sh_f32 = arith_ops.ExtFOp(vec_f32_t, _v(sh_vec)).result
            y_f32 = arith_ops.AddFOp(x_hat, sh_f32).result

            y_bf16 = arith_ops.TruncFOp(vec_bf16_t, y_f32).result
            buffer_ops.buffer_store(y_bf16, y_rsrc, off)

    @flyc.jit
    def launch_fused_norm(
        y: fx.Tensor,
        res_out: fx.Tensor,
        res: fx.Tensor,
        x: fx.Tensor,
        gate: fx.Tensor,
        weight: fx.Tensor,
        bias: fx.Tensor,
        scale: fx.Tensor,
        shift: fx.Tensor,
        total_rows: fx.Int32,
        gate_stride: fx.Int32,
        scale_stride: fx.Int32,
        shift_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        grid_x = arith.index_cast(T.index, total_rows)
        launcher = flydsl_fused_residual_norm_ss_kernel(
            y,
            res_out,
            res,
            x,
            gate,
            weight,
            bias,
            scale,
            shift,
            total_rows,
            gate_stride,
            scale_stride,
            shift_stride,
        )
        LDS_BYTES = (NUM_WAVES * 2 + 2) * 4
        launcher.launch(
            grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), smem=LDS_BYTES, stream=stream
        )

    return launch_fused_norm


_COMPILE_CACHE = {}


def _get_or_compile(D, is_rms, has_gate, has_weight, args):
    key = (D, is_rms, has_gate, has_weight)
    if key not in _COMPILE_CACHE:
        launcher = _build_fused_norm_module(D, is_rms, has_gate, has_weight)
        cf = flyc.compile(launcher, *args)
        _COMPILE_CACHE[key] = cf
    return _COMPILE_CACHE[key]


def _to_bf16(t):
    """Convert to bf16 only if not already bf16."""
    return t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)


def _prep_slices(t, B, L, C):
    """Prepare per-batch tensor slices and kernel row_stride.

    Returns (slices, row_stride) where:
      slices[b] = tensor to pass to kernel for batch b
      row_stride = 0 (broadcast: all rows share one row) or C (per-row data)
    """
    t = _to_bf16(t)

    if t.numel() < C:
        row = t.flatten()[0].expand(C).contiguous().unsqueeze(0)
        return [row] * B, 0

    if t.dim() == 1:
        return [t.unsqueeze(0).contiguous()] * B, 0

    if t.dim() == 2:
        if t.shape[0] == 1:
            return [t.contiguous()] * B, 0
        return [t.contiguous()] * B, C

    if t.dim() == 3:
        if t.shape[0] == 1 and t.shape[1] == 1:
            return [t.reshape(1, C).contiguous()] * B, 0
        if t.shape[1] == 1:
            t_c = t.contiguous()
            return [t_c[b] for b in range(B)], 0
        t_exp = t.expand(B, L, C).contiguous()
        return [t_exp[b] for b in range(B)], C

    if t.dim() == 4:
        nf = t.shape[1]
        fs = L // nf
        t_exp = t.expand(B, nf, fs, C).reshape(B, L, C).contiguous()
        return [t_exp[b] for b in range(B)], C

    t_exp = t.reshape(B, L, C).contiguous()
    return [t_exp[b] for b in range(B)], C


def _ensure_bf16_contig(t):
    """Return bf16-contiguous view, avoiding copies when possible."""
    if t.dtype == torch.bfloat16 and t.is_contiguous():
        return t
    return _to_bf16(t).contiguous()


@torch.library.custom_op(
    "sglang::flydsl_fused_residual_norm_scale_shift", mutates_args=()
)
def flydsl_fused_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, C = x.shape
    rows = B * L
    bf16 = torch.bfloat16

    x_2d = _ensure_bf16_contig(x).reshape(rows, C)
    res_2d = _ensure_bf16_contig(residual).reshape(rows, C)
    y = torch.empty_like(x_2d)
    res_out = torch.empty_like(x_2d)

    has_gate = gate is not None
    if has_gate:
        g_slices, g_stride = _prep_slices(gate, B, L, C)
    else:
        g_slices, g_stride = [x_2d[:1]] * B, 0

    has_weight = weight is not None
    weight_c = (
        _ensure_bf16_contig(weight)
        if has_weight
        else torch.empty(C, device=x.device, dtype=bf16)
    )
    bias_c = (
        _ensure_bf16_contig(bias)
        if bias is not None
        else torch.zeros(C, device=x.device, dtype=bf16)
    )

    sc_slices, sc_stride = _prep_slices(scale, B, L, C)
    sh_slices, sh_stride = _prep_slices(shift, B, L, C)

    is_rms = norm_type == "rms"
    stream = torch.cuda.current_stream()

    dummy_args = (
        y[:L],
        res_out[:L],
        res_2d[:L],
        x_2d[:L],
        g_slices[0],
        weight_c,
        bias_c,
        sc_slices[0],
        sh_slices[0],
        L,
        g_stride,
        sc_stride,
        sh_stride,
        stream,
    )
    cf = _get_or_compile(C, is_rms, has_gate, has_weight, dummy_args)

    for b in range(B):
        s, e = b * L, (b + 1) * L
        cf(
            y[s:e],
            res_out[s:e],
            res_2d[s:e],
            x_2d[s:e],
            g_slices[b],
            weight_c,
            bias_c,
            sc_slices[b],
            sh_slices[b],
            L,
            g_stride,
            sc_stride,
            sh_stride,
            stream,
        )

    return y.view(B, L, C), res_out.view(B, L, C)


@flydsl_fused_residual_norm_scale_shift.register_fake
def _fake_flydsl_fused_residual_norm(
    residual,
    x,
    gate,
    weight,
    bias,
    scale,
    shift,
    norm_type,
    eps=1e-6,
):
    B, L, C = x.shape
    bf16 = torch.bfloat16
    y = torch.empty(B, L, C, device=x.device, dtype=bf16)
    res_out = torch.empty(B, L, C, device=x.device, dtype=bf16)
    return y, res_out


###############################################################################
# _NormScaleShift kernel: norm(x) * (1+scale) + shift  (no residual path)
###############################################################################


def _build_norm_scale_shift_module(D: int, is_rms: bool, has_weight: bool):
    VEC = _VEC
    NUM_WAVES = _NUM_WAVES
    BLOCK = NUM_WAVES * WARP_SIZE
    assert (
        D % FLYDSL_NORM_MIN_ALIGNED_DIM == 0
    ), f"FlyDSL norm_scale_shift requires D % {FLYDSL_NORM_MIN_ALIGNED_DIM} == 0, got D={D}"
    NUM_ITERS = D // (BLOCK * VEC)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flydsl_norm_scale_shift_kernel(
        y_ptr: fx.Tensor,
        x_ptr: fx.Tensor,
        weight_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        scale_ptr: fx.Tensor,
        shift_ptr: fx.Tensor,
        total_rows: Int32,
        scale_stride: Int32,
        shift_stride: Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        i32 = T.i32
        f32 = T.f32
        bf16 = T.bf16
        vec_f32_t = ir.VectorType.get([VEC], f32)
        vec_bf16_t = ir.VectorType.get([VEC], bf16)

        y_rsrc = buffer_ops.create_buffer_resource(y_ptr, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(x_ptr, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(weight_ptr, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(bias_ptr, max_size=True)
        sc_rsrc = buffer_ops.create_buffer_resource(scale_ptr, max_size=True)
        sh_rsrc = buffer_ops.create_buffer_resource(shift_ptr, max_size=True)

        row_i32 = ArithValue(row)
        tid_i32 = ArithValue(tid)
        D_i32 = arith.constant(D, type=i32)
        row_off = row_i32 * D_i32
        scale_row_off = row_i32 * ArithValue(scale_stride)
        shift_row_off = row_i32 * ArithValue(shift_stride)

        c_zero_f32 = arith.constant(0.0, type=f32)
        c_one_f32 = arith.constant(1.0, type=f32)
        eps_val = arith.constant(1e-6, type=f32)
        D_float = arith.constant(float(D), type=f32)

        LDS_SLOTS = NUM_WAVES * 2 + 2
        ws_attr = ir.Attribute.parse("#gpu.address_space<workgroup>")
        lds_i8_type = ir.MemRefType.get(
            [ir.ShapedType.get_dynamic_size()], T.i8, memory_space=ws_attr
        )
        lds_f32_type = ir.MemRefType.get([LDS_SLOTS], f32, memory_space=ws_attr)
        lds_i8 = _gpu.DynamicSharedMemoryOp(lds_i8_type).result
        byte_zero = arith_ops.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 0)
        ).result
        lds = _memref.ViewOp(lds_f32_type, lds_i8, byte_zero, []).result

        lane_id = tid_i32 % arith.constant(WARP_SIZE, type=i32)
        wave_id = tid_i32 // arith.constant(WARP_SIZE, type=i32)

        # Phase 1: load x, accumulate stats, save f32 in registers
        _saved_x_f32 = []
        partial_sum = _v(c_zero_f32)
        partial_sum_sq = _v(c_zero_f32)

        for it in range_constexpr(NUM_ITERS):
            col = tid_i32 * arith.constant(VEC, type=i32) + arith.constant(
                it * BLOCK * VEC, type=i32
            )
            off = row_off + col

            x_vec = buffer_ops.buffer_load(x_rsrc, off, vec_width=VEC, dtype=bf16)
            x_f32 = arith_ops.ExtFOp(vec_f32_t, _v(x_vec)).result
            _saved_x_f32.append(x_f32)

            if const_expr(not is_rms):
                v_sum = _vector.ReductionOp(
                    f32, _vector.CombiningKind.ADD, x_f32
                ).result
                partial_sum = arith_ops.AddFOp(partial_sum, v_sum).result
            x_sq = arith_ops.MulFOp(x_f32, x_f32).result
            v_sum_sq = _vector.ReductionOp(f32, _vector.CombiningKind.ADD, x_sq).result
            partial_sum_sq = arith_ops.AddFOp(partial_sum_sq, v_sum_sq).result

        # Intra-wave shuffle reduction
        width_c = _v(arith.constant(WARP_SIZE, type=i32))
        w_sum = partial_sum
        w_sq = partial_sum_sq
        for sh in [32, 16, 8, 4, 2, 1]:
            off_sh = _v(arith.constant(sh, type=i32))
            if const_expr(not is_rms):
                peer_sum = _gpu.ShuffleOp(
                    w_sum, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
                ).shuffleResult
                w_sum = arith_ops.AddFOp(w_sum, peer_sum).result
            peer_sq = _gpu.ShuffleOp(
                w_sq, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
            ).shuffleResult
            w_sq = arith_ops.AddFOp(w_sq, peer_sq).result

        # Cross-wave LDS reduction
        lane_0 = arith.cmpi(CmpIPredicate.eq, lane_id, arith.constant(0, type=i32))
        wave_idx = arith_ops.IndexCastOp(ir.IndexType.get(), _v(wave_id)).result

        _if_lane0 = scf.IfOp(lane_0)
        with ir.InsertionPoint(_if_lane0.then_block):
            if const_expr(not is_rms):
                _memref.StoreOp(w_sum, lds, [wave_idx])
            sq_slot = arith_ops.AddIOp(
                wave_idx,
                arith_ops.ConstantOp(
                    ir.IndexType.get(),
                    ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES),
                ).result,
            ).result
            _memref.StoreOp(w_sq, lds, [sq_slot])
            scf.YieldOp([])
        _gpu.BarrierOp()

        wave_0 = arith.cmpi(CmpIPredicate.eq, wave_id, arith.constant(0, type=i32))
        active = arith.andi(
            wave_0,
            arith.cmpi(CmpIPredicate.ult, lane_id, arith.constant(NUM_WAVES, type=i32)),
        )
        lane_idx = arith_ops.IndexCastOp(ir.IndexType.get(), _v(lane_id)).result
        lane_idx_sq = arith_ops.AddIOp(
            lane_idx,
            arith_ops.ConstantOp(
                ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES)
            ).result,
        ).result

        if const_expr(is_rms):
            _if_active = scf.IfOp(active, [f32], has_else=True)
            with ir.InsertionPoint(_if_active.then_block):
                sq_val = _memref.LoadOp(lds, [lane_idx_sq]).result
                scf.YieldOp([sq_val])
            with ir.InsertionPoint(_if_active.else_block):
                scf.YieldOp([_v(c_zero_f32)])
            loaded_sq = _if_active.results[0]
            loaded_sum = _v(c_zero_f32)
        else:
            _if_active = scf.IfOp(active, [f32, f32], has_else=True)
            with ir.InsertionPoint(_if_active.then_block):
                s_val = _memref.LoadOp(lds, [lane_idx]).result
                sq_val = _memref.LoadOp(lds, [lane_idx_sq]).result
                scf.YieldOp([s_val, sq_val])
            with ir.InsertionPoint(_if_active.else_block):
                scf.YieldOp([_v(c_zero_f32), _v(c_zero_f32)])
            loaded_sum = _if_active.results[0]
            loaded_sq = _if_active.results[1]

        final_sum = loaded_sum
        final_sq = loaded_sq
        for sh in [32, 16, 8, 4, 2, 1]:
            off_sh = _v(arith.constant(sh, type=i32))
            if const_expr(not is_rms):
                ps = _gpu.ShuffleOp(
                    final_sum, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
                ).shuffleResult
                final_sum = arith_ops.AddFOp(final_sum, ps).result
            pq = _gpu.ShuffleOp(
                final_sq, off_sh, width_c, mode=_gpu.ShuffleMode.XOR
            ).shuffleResult
            final_sq = arith_ops.AddFOp(final_sq, pq).result

        both_0 = arith.andi(wave_0, lane_0)
        final_sum_slot = arith_ops.ConstantOp(
            ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES * 2)
        ).result
        final_sq_slot = arith_ops.ConstantOp(
            ir.IndexType.get(),
            ir.IntegerAttr.get(ir.IndexType.get(), NUM_WAVES * 2 + 1),
        ).result
        _if_both = scf.IfOp(both_0)
        with ir.InsertionPoint(_if_both.then_block):
            if const_expr(not is_rms):
                _memref.StoreOp(final_sum, lds, [final_sum_slot])
            _memref.StoreOp(final_sq, lds, [final_sq_slot])
            scf.YieldOp([])
        _gpu.BarrierOp()

        if const_expr(not is_rms):
            total_sum = _memref.LoadOp(lds, [final_sum_slot]).result
        else:
            total_sum = _v(c_zero_f32)
        total_sq = _memref.LoadOp(lds, [final_sq_slot]).result

        d_f = _v(D_float)
        eps_v = _v(eps_val)
        if const_expr(is_rms):
            var = arith_ops.DivFOp(total_sq, d_f).result
            var_eps = arith_ops.AddFOp(var, eps_v).result
            rstd = math_ops.RsqrtOp(var_eps).result
            mean = _v(c_zero_f32)
        else:
            mean = arith_ops.DivFOp(total_sum, d_f).result
            mean_sq = arith_ops.MulFOp(mean, mean).result
            var = arith_ops.SubFOp(
                arith_ops.DivFOp(total_sq, d_f).result, mean_sq
            ).result
            var_eps = arith_ops.AddFOp(var, eps_v).result
            rstd = math_ops.RsqrtOp(var_eps).result

        # Phase 2: normalize from register cache + scale_shift → single output
        mean_splat = _vector.BroadcastOp(vec_f32_t, mean).result
        rstd_splat = _vector.BroadcastOp(vec_f32_t, rstd).result
        one_splat = _vector.BroadcastOp(vec_f32_t, _v(c_one_f32)).result

        for it in range_constexpr(NUM_ITERS):
            col = tid_i32 * arith.constant(VEC, type=i32) + arith.constant(
                it * BLOCK * VEC, type=i32
            )
            off = row_off + col

            x_f32 = _saved_x_f32[it]

            if const_expr(is_rms):
                x_hat = arith_ops.MulFOp(x_f32, rstd_splat).result
            else:
                centered = arith_ops.SubFOp(x_f32, mean_splat).result
                x_hat = arith_ops.MulFOp(centered, rstd_splat).result

            if const_expr(has_weight):
                w_vec = buffer_ops.buffer_load(w_rsrc, col, vec_width=VEC, dtype=bf16)
                w_f32 = arith_ops.ExtFOp(vec_f32_t, _v(w_vec)).result
                x_hat = arith_ops.MulFOp(x_hat, w_f32).result
                b_vec = buffer_ops.buffer_load(b_rsrc, col, vec_width=VEC, dtype=bf16)
                b_f32 = arith_ops.ExtFOp(vec_f32_t, _v(b_vec)).result
                x_hat = arith_ops.AddFOp(x_hat, b_f32).result

            sc_off = scale_row_off + col
            sc_vec = buffer_ops.buffer_load(sc_rsrc, sc_off, vec_width=VEC, dtype=bf16)
            sc_f32 = arith_ops.ExtFOp(vec_f32_t, _v(sc_vec)).result
            sc_p1 = arith_ops.AddFOp(one_splat, sc_f32).result
            x_hat = arith_ops.MulFOp(x_hat, sc_p1).result

            sh_off = shift_row_off + col
            sh_vec = buffer_ops.buffer_load(sh_rsrc, sh_off, vec_width=VEC, dtype=bf16)
            sh_f32 = arith_ops.ExtFOp(vec_f32_t, _v(sh_vec)).result
            y_f32 = arith_ops.AddFOp(x_hat, sh_f32).result

            y_bf16 = arith_ops.TruncFOp(vec_bf16_t, y_f32).result
            buffer_ops.buffer_store(y_bf16, y_rsrc, off)

    @flyc.jit
    def launch_norm_ss(
        y: fx.Tensor,
        x: fx.Tensor,
        weight: fx.Tensor,
        bias: fx.Tensor,
        scale: fx.Tensor,
        shift: fx.Tensor,
        total_rows: fx.Int32,
        scale_stride: fx.Int32,
        shift_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        grid_x = arith.index_cast(T.index, total_rows)
        launcher = flydsl_norm_scale_shift_kernel(
            y,
            x,
            weight,
            bias,
            scale,
            shift,
            total_rows,
            scale_stride,
            shift_stride,
        )
        LDS_BYTES = (NUM_WAVES * 2 + 2) * 4
        launcher.launch(
            grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), smem=LDS_BYTES, stream=stream
        )

    return launch_norm_ss


_NSS_COMPILE_CACHE = {}


def _get_or_compile_nss(D, is_rms, has_weight, args):
    key = ("nss", D, is_rms, has_weight)
    if key not in _NSS_COMPILE_CACHE:
        launcher = _build_norm_scale_shift_module(D, is_rms, has_weight)
        cf = flyc.compile(launcher, *args)
        _NSS_COMPILE_CACHE[key] = cf
    return _NSS_COMPILE_CACHE[key]


@torch.library.custom_op("sglang::flydsl_norm_scale_shift", mutates_args=())
def flydsl_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, L, C = x.shape
    rows = B * L
    bf16 = torch.bfloat16

    x_2d = _ensure_bf16_contig(x).reshape(rows, C)
    y = torch.empty_like(x_2d)

    has_weight = weight is not None
    weight_c = (
        _ensure_bf16_contig(weight)
        if has_weight
        else torch.empty(C, device=x.device, dtype=bf16)
    )
    bias_c = (
        _ensure_bf16_contig(bias)
        if bias is not None
        else torch.zeros(C, device=x.device, dtype=bf16)
    )

    sc_slices, sc_stride = _prep_slices(scale, B, L, C)
    sh_slices, sh_stride = _prep_slices(shift, B, L, C)

    is_rms = norm_type == "rms"
    stream = torch.cuda.current_stream()

    dummy_args = (
        y[:L],
        x_2d[:L],
        weight_c,
        bias_c,
        sc_slices[0],
        sh_slices[0],
        L,
        sc_stride,
        sh_stride,
        stream,
    )
    cf = _get_or_compile_nss(C, is_rms, has_weight, dummy_args)

    for b in range(B):
        s, e = b * L, (b + 1) * L
        cf(
            y[s:e],
            x_2d[s:e],
            weight_c,
            bias_c,
            sc_slices[b],
            sh_slices[b],
            L,
            sc_stride,
            sh_stride,
            stream,
        )

    return y.view(B, L, C)


@flydsl_norm_scale_shift.register_fake
def _fake_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps=1e-6):
    B, L, C = x.shape
    return torch.empty(B, L, C, device=x.device, dtype=torch.bfloat16)
