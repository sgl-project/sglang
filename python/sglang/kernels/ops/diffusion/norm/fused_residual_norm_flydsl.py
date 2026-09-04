"""FlyDSL fused normalization kernels for AMD ROCm (gfx950).

Provides two fused kernels:
  - flydsl_fused_residual_norm_scale_shift:
        residual_add + gate_mul + RMSNorm/LayerNorm + scale·shift
  - flydsl_norm_scale_shift:
        RMSNorm/LayerNorm + scale·shift

Both kernels use register-cache optimization: Phase 2 (scale·shift)
reuses f32 intermediate values from Phase 1 (norm) registers instead
of re-reading from HBM, saving ~20% bandwidth.

Written against the FlyDSL v0.3.0 stable public API only; see
docs/api_stability.md in the FlyDSL tree for the classification rules.
Raw MLIR dialect builders, compiler-internal contexts, private expr
submodules, and anything under the FlyDSL source-only kernel tree are all
outside the shipped wheel or the stability contract -- do not reintroduce
them here. CI greps this file for those names, so avoid spelling them even
in comments.
"""

from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr

WARP_SIZE = 64
_VEC = 8
_NUM_WAVES = 10
FLYDSL_NORM_MIN_ALIGNED_DIM = WARP_SIZE * _NUM_WAVES * _VEC  # 5120

# Kernel-side epsilon. The public `eps` argument is intentionally ignored, as it
# was before the stable-API migration; changing that is a separate correctness fix.
_EPS = 1e-6

# 128-bit vector copies over bf16 elements.
_ELEM_BITS = 16

# Reduction order is part of the numerical contract; keep the descending sequence.
_SHUFFLE_OFFSETS = (32, 16, 8, 4, 2, 1)


def _require_stable_api() -> None:
    """Fail as ImportError when the installed FlyDSL predates the v0.3.0 surface.

    Callers in multimodal_gen/runtime/layers/layernorm.py guard this module with
    `except ImportError` and fall back to the native path, so a missing symbol
    must surface as ImportError rather than AttributeError.
    """
    required = {
        "flydsl.expr": (
            "Tensor",
            "Stream",
            "Int32",
            "Float32",
            "BFloat16",
            "ReductionOp",
            "SharedAllocator",
            "struct",
            "Array",
            "make_layout",
            "logical_divide",
            "slice",
            "make_copy_atom",
            "copy_atom_call",
            "make_rmem_tensor",
            "memref_load_vec",
            "memref_store_vec",
            "memref_load",
            "memref_store",
        ),
        "flydsl.expr.gpu": ("barrier", "shuffle_xor", "block_idx", "thread_idx"),
        "flydsl.expr.math": ("rsqrt",),
        "flydsl.expr.rocdl": ("make_buffer_tensor", "BufferCopy128b"),
        "flydsl.compiler": ("kernel", "jit", "compile"),
    }
    roots = {
        "flydsl.expr": fx,
        "flydsl.expr.gpu": fx.gpu,
        "flydsl.expr.math": fx.math,
        "flydsl.expr.rocdl": fx.rocdl,
        "flydsl.compiler": flyc,
    }
    missing = [
        f"{mod}.{name}"
        for mod, names in required.items()
        for name in names
        if not hasattr(roots[mod], name)
    ]
    if missing:
        raise ImportError(
            "FlyDSL is too old for sglang's fused norm kernels; missing stable "
            f"v0.3.0 APIs: {', '.join(missing)}"
        )


_require_stable_api()


def _make_reduction_storage(slots: int):
    """LDS layout for the two-stage block reduction.

    s_sum/s_sq hold one partial per wave; s_final holds the two broadcast values
    so the final write never aliases a slot that is still being read.
    """

    @fx.struct
    class SharedStorage:
        s_sum: fx.Array[fx.Float32, slots, 16]
        s_sq: fx.Array[fx.Float32, slots, 16]
        s_final: fx.Array[fx.Float32, 2, 16]

    return SharedStorage


def _load_vec(copy_atom, div_tensor, idx):
    r = fx.make_rmem_tensor(_VEC, fx.BFloat16)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, div_tensor, idx, val):
    r = fx.make_rmem_tensor(_VEC, fx.BFloat16)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _row_div(tensor, row):
    """Vector-partitioned view of one row of a rank-2 tensor."""
    return fx.logical_divide(
        fx.slice(fx.rocdl.make_buffer_tensor(tensor), (row, None)),
        fx.make_layout(_VEC, 1),
    )


def _flat_div(tensor):
    """Vector-partitioned view of a rank-1 tensor."""
    return fx.logical_divide(
        fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(_VEC, 1)
    )


def _bcast_row(row, stride):
    """Row index honoring a runtime broadcast stride.

    A broadcast operand arrives as a dense (1, C) tensor and the host signals it
    with stride == 0, so selecting row 0 reproduces the previous `row * stride`
    addressing exactly.
    """
    return (stride != 0).select(row, 0)


def _build_fused_norm_module(D: int, is_rms: bool, has_gate: bool, has_weight: bool):
    VEC = _VEC
    NUM_WAVES = _NUM_WAVES
    BLOCK = NUM_WAVES * WARP_SIZE
    assert D % FLYDSL_NORM_MIN_ALIGNED_DIM == 0, (
        f"FlyDSL fused_residual_norm requires D % {FLYDSL_NORM_MIN_ALIGNED_DIM} == 0, got D={D}"
    )
    NUM_ITERS = D // (BLOCK * VEC)
    SharedStorage = _make_reduction_storage(NUM_WAVES)

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
        total_rows: fx.Int32,
        gate_stride: fx.Int32,
        scale_stride: fx.Int32,
        shift_stride: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane_id = tid % WARP_SIZE
        wave_id = tid // WARP_SIZE

        n_float = float(D)
        c_zero = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(NUM_WAVES, 1))
        s_sq = lds.s_sq.view(fx.make_layout(NUM_WAVES, 1))
        s_final = lds.s_final.view(fx.make_layout(2, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), _ELEM_BITS)

        y_div = _row_div(y_ptr, row)
        ro_div = _row_div(res_out_ptr, row)
        r_div = _row_div(res_ptr, row)
        x_div = _row_div(x_ptr, row)
        if const_expr(has_gate):
            g_div = _row_div(gate_ptr, _bcast_row(row, gate_stride))
        if const_expr(has_weight):
            w_div = _flat_div(weight_ptr)
            b_div = _flat_div(bias_ptr)
        sc_div = _row_div(scale_ptr, _bcast_row(row, scale_stride))
        sh_div = _row_div(shift_ptr, _bcast_row(row, shift_stride))

        def wave_reduce_add(val):
            w = val
            for i in range_constexpr(len(_SHUFFLE_OFFSETS)):
                w = w + fx.gpu.shuffle_xor(w, _SHUFFLE_OFFSETS[i], WARP_SIZE)
            return w

        # Phase 1: residual + gate*x, accumulate stats, keep f32 in registers.
        saved_ro = []
        partial_sum = c_zero
        partial_sq = c_zero

        for it in range_constexpr(NUM_ITERS):
            idx = tid + it * BLOCK

            r_f32 = _load_vec(copy_atom, r_div, idx).to(fx.Float32)
            x_f32 = _load_vec(copy_atom, x_div, idx).to(fx.Float32)

            if const_expr(has_gate):
                g_f32 = _load_vec(copy_atom, g_div, idx).to(fx.Float32)
                ro_f32 = r_f32 + g_f32 * x_f32
            else:
                ro_f32 = r_f32 + x_f32

            _store_vec(copy_atom, ro_div, idx, ro_f32.to(fx.BFloat16))
            saved_ro.append(ro_f32)

            if const_expr(not is_rms):
                partial_sum = partial_sum + ro_f32.reduce(fx.ReductionOp.ADD)
            partial_sq = partial_sq + (ro_f32 * ro_f32).reduce(fx.ReductionOp.ADD)

        # Stage 1: intra-wave shuffle reduction.
        if const_expr(not is_rms):
            w_sum = wave_reduce_add(partial_sum)
        w_sq = wave_reduce_add(partial_sq)

        if lane_id == 0:
            if const_expr(not is_rms):
                fx.memref_store(w_sum, s_sum, wave_id)
            fx.memref_store(w_sq, s_sq, wave_id)
        fx.gpu.barrier()

        # Stage 2: wave 0 folds the per-wave partials and publishes the results.
        if wave_id == 0:
            in_range = lane_id < NUM_WAVES
            lane_safe = in_range.select(lane_id, 0)
            if const_expr(not is_rms):
                v_sum = wave_reduce_add(
                    in_range.select(fx.memref_load(s_sum, lane_safe), c_zero)
                )
            v_sq = wave_reduce_add(
                in_range.select(fx.memref_load(s_sq, lane_safe), c_zero)
            )
            if lane_id == 0:
                if const_expr(not is_rms):
                    fx.memref_store(v_sum, s_final, 0)
                fx.memref_store(v_sq, s_final, 1)
        fx.gpu.barrier()

        total_sq = fx.memref_load(s_final, 1)

        if const_expr(is_rms):
            rstd = fx.math.rsqrt(total_sq / n_float + _EPS)
        else:
            mean = fx.memref_load(s_final, 0) / n_float
            var = total_sq / n_float - mean * mean
            rstd = fx.math.rsqrt(var + _EPS)

        # Phase 2: normalize from the register cache (no HBM re-read).
        for it in range_constexpr(NUM_ITERS):
            idx = tid + it * BLOCK
            ro_f32 = saved_ro[it]

            if const_expr(is_rms):
                x_hat = ro_f32 * rstd
            else:
                x_hat = (ro_f32 - mean) * rstd

            if const_expr(has_weight):
                x_hat = x_hat * _load_vec(copy_atom, w_div, idx).to(fx.Float32)
                x_hat = x_hat + _load_vec(copy_atom, b_div, idx).to(fx.Float32)

            sc_f32 = _load_vec(copy_atom, sc_div, idx).to(fx.Float32)
            x_hat = x_hat * (sc_f32 + 1.0)
            y_f32 = x_hat + _load_vec(copy_atom, sh_div, idx).to(fx.Float32)

            _store_vec(copy_atom, y_div, idx, y_f32.to(fx.BFloat16))

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
        launcher.launch(grid=(total_rows, 1, 1), block=(BLOCK, 1, 1), stream=stream)

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
    assert D % FLYDSL_NORM_MIN_ALIGNED_DIM == 0, (
        f"FlyDSL norm_scale_shift requires D % {FLYDSL_NORM_MIN_ALIGNED_DIM} == 0, got D={D}"
    )
    NUM_ITERS = D // (BLOCK * VEC)
    SharedStorage = _make_reduction_storage(NUM_WAVES)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def flydsl_norm_scale_shift_kernel(
        y_ptr: fx.Tensor,
        x_ptr: fx.Tensor,
        weight_ptr: fx.Tensor,
        bias_ptr: fx.Tensor,
        scale_ptr: fx.Tensor,
        shift_ptr: fx.Tensor,
        total_rows: fx.Int32,
        scale_stride: fx.Int32,
        shift_stride: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane_id = tid % WARP_SIZE
        wave_id = tid // WARP_SIZE

        n_float = float(D)
        c_zero = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(NUM_WAVES, 1))
        s_sq = lds.s_sq.view(fx.make_layout(NUM_WAVES, 1))
        s_final = lds.s_final.view(fx.make_layout(2, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), _ELEM_BITS)

        y_div = _row_div(y_ptr, row)
        x_div = _row_div(x_ptr, row)
        if const_expr(has_weight):
            w_div = _flat_div(weight_ptr)
            b_div = _flat_div(bias_ptr)
        sc_div = _row_div(scale_ptr, _bcast_row(row, scale_stride))
        sh_div = _row_div(shift_ptr, _bcast_row(row, shift_stride))

        def wave_reduce_add(val):
            w = val
            for i in range_constexpr(len(_SHUFFLE_OFFSETS)):
                w = w + fx.gpu.shuffle_xor(w, _SHUFFLE_OFFSETS[i], WARP_SIZE)
            return w

        # Phase 1: load x, accumulate stats, keep f32 in registers.
        saved_x = []
        partial_sum = c_zero
        partial_sq = c_zero

        for it in range_constexpr(NUM_ITERS):
            idx = tid + it * BLOCK
            x_f32 = _load_vec(copy_atom, x_div, idx).to(fx.Float32)
            saved_x.append(x_f32)

            if const_expr(not is_rms):
                partial_sum = partial_sum + x_f32.reduce(fx.ReductionOp.ADD)
            partial_sq = partial_sq + (x_f32 * x_f32).reduce(fx.ReductionOp.ADD)

        # Stage 1: intra-wave shuffle reduction.
        if const_expr(not is_rms):
            w_sum = wave_reduce_add(partial_sum)
        w_sq = wave_reduce_add(partial_sq)

        if lane_id == 0:
            if const_expr(not is_rms):
                fx.memref_store(w_sum, s_sum, wave_id)
            fx.memref_store(w_sq, s_sq, wave_id)
        fx.gpu.barrier()

        # Stage 2: wave 0 folds the per-wave partials and publishes the results.
        if wave_id == 0:
            in_range = lane_id < NUM_WAVES
            lane_safe = in_range.select(lane_id, 0)
            if const_expr(not is_rms):
                v_sum = wave_reduce_add(
                    in_range.select(fx.memref_load(s_sum, lane_safe), c_zero)
                )
            v_sq = wave_reduce_add(
                in_range.select(fx.memref_load(s_sq, lane_safe), c_zero)
            )
            if lane_id == 0:
                if const_expr(not is_rms):
                    fx.memref_store(v_sum, s_final, 0)
                fx.memref_store(v_sq, s_final, 1)
        fx.gpu.barrier()

        total_sq = fx.memref_load(s_final, 1)

        if const_expr(is_rms):
            rstd = fx.math.rsqrt(total_sq / n_float + _EPS)
        else:
            mean = fx.memref_load(s_final, 0) / n_float
            var = total_sq / n_float - mean * mean
            rstd = fx.math.rsqrt(var + _EPS)

        # Phase 2: normalize from the register cache + scale_shift.
        for it in range_constexpr(NUM_ITERS):
            idx = tid + it * BLOCK
            x_f32 = saved_x[it]

            if const_expr(is_rms):
                x_hat = x_f32 * rstd
            else:
                x_hat = (x_f32 - mean) * rstd

            if const_expr(has_weight):
                x_hat = x_hat * _load_vec(copy_atom, w_div, idx).to(fx.Float32)
                x_hat = x_hat + _load_vec(copy_atom, b_div, idx).to(fx.Float32)

            sc_f32 = _load_vec(copy_atom, sc_div, idx).to(fx.Float32)
            x_hat = x_hat * (sc_f32 + 1.0)
            y_f32 = x_hat + _load_vec(copy_atom, sh_div, idx).to(fx.Float32)

            _store_vec(copy_atom, y_div, idx, y_f32.to(fx.BFloat16))

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
        launcher.launch(grid=(total_rows, 1, 1), block=(BLOCK, 1, 1), stream=stream)

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
