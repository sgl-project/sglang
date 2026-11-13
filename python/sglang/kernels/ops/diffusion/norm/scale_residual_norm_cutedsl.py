import functools
from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from sglang.kernels.ops.diffusion.norm.cutedsl_common.norm_fusion import (
    broadcast_tensor_for_bsfd,
    normalize_chunk,
    row_norm_stats,
    tensor_slice_for_bsfd,
)
from sglang.kernels.ops.diffusion.norm.cutedsl_utils import (
    WARP_SIZE,
    to_fake_cute_args,
)

_COMPILE_CACHE = {}


# columns one warp covers per 128-bit load: 32 lanes x 8 elements
CHUNK = WARP_SIZE * 8


class ScaleResidualNormScaleShift:
    @classmethod
    def make_hash_key(cls, *inputs):
        """
        Compile-time values:
          - D: hidden dimension (size of the last dimension)
          - norm_type: layer norm or RMS norm
          - chunks per warp
          - tensor dtype
          - tensor rank (i.e., tensor.ndim)

        Runtime values:
          - all other inputs

        This hash key defines the compile-time specialization boundary for
        ScaleResidualNormScaleShift kernels.
        """

        def _sig(val):
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(self, D: int, norm_type: str, chunks_per_warp: int):
        self.D = D
        self.norm_type = norm_type  # "layer" or "rms"
        # a row is D // CHUNK chunks; each warp walks chunks_per_warp of them, so the
        # reduction order is fixed by D alone
        self.num_chunks = D // CHUNK
        self.chunks_per_warp = chunks_per_warp
        self.num_threads = self.num_chunks // chunks_per_warp * WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mY,
        mResOut,
        mRes,
        mX,
        mGate,
        mWeight,
        mBias,
        mScale,
        mShift,
        eps: cutlass.Float32 = cutlass.Float32(1e-5),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        B, S, _ = mX.shape  # (batch, seq_len, hidden_dim)
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        # one warp per chunk: 8 consecutive elements per lane (LDG.128)
        tiled_copy = cute.make_tiled_copy_tv(
            atom_copy, cute.make_layout(WARP_SIZE), cute.make_layout(8)
        )

        self.kernel(
            mY,
            mResOut,
            mRes,
            mX,
            mGate,
            mWeight,
            mBias,
            mScale,
            mShift,
            tiled_copy,
            eps,
        ).launch(
            grid=[B * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY,
        mResOut,
        mRes,
        mX,
        mGate,
        mWeight,
        mBias,
        mScale,
        mShift,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        _, S, _ = mX.shape
        D = self.D
        tidx, _, _ = cute.arch.thread_idx()  # thread index
        bid, _, _ = cute.arch.block_idx()  # cta index
        bidx = cutlass.Int32(bid // S)  # batch index
        bidy = cutlass.Int32(bid % S)  # seq_len index
        lane_id = tidx & (WARP_SIZE - 1)
        warp_id = tidx // WARP_SIZE
        thr_copy = tiled_copy.get_slice(lane_id)
        chunk0 = warp_id * self.chunks_per_warp

        # For norm_scale_shift, output:
        # - y = norm(x, weight, bias) * (1 + scale) + shift
        # For scale_residual_norm_scale_shift, output:
        # - residual_out = residual + gate * x
        # - y = norm(residual_out, weight, bias) * (1 + scale) + shift
        # 1. load every chunk of this warp: residual, x, gate
        tXrXs, tRrRs, tGrGs = [], [], []
        for j in cutlass.range_constexpr(self.chunks_per_warp):
            tRgR, tRrR = _slice(mRes, thr_copy, bidx, bidy, chunk0 + j, S, D)
            tXgX, tXrX = _slice(mX, thr_copy, bidx, bidy, chunk0 + j, S, D)
            tGgG, tGrG = _slice(mGate, thr_copy, bidx, bidy, chunk0 + j, S, D)
            _copy(tRgR, tRrR)  # gmem -> rmem
            _copy(tXgX, tXrX)
            _copy(tGgG, tGrG)
            tXrXs.append(tXrX)
            tRrRs.append(tRrR)
            tGrGs.append(tGrG)
        # 2. value = <gate> * x + <residual>; store residual_out
        tNrNs = []
        for j in cutlass.range_constexpr(self.chunks_per_warp):
            value = tXrXs[j].load()
            if cutlass.const_expr(isinstance(tGrGs[j], cute.Tensor)):
                value = tGrGs[j].load() * value
            if cutlass.const_expr(isinstance(tRrRs[j], cute.Tensor)):
                value = value + tRrRs[j].load()
            tROgRO, tROrRO = _slice(mResOut, thr_copy, bidx, bidy, chunk0 + j, S, D)
            if cutlass.const_expr(isinstance(tROrRO, cute.Tensor)):
                tROrRO.store(value.to(tROrRO.element_type))
                _copy(tROrRO, tROgRO)  # rmem -> gmem
            tNrN = cute.make_rmem_tensor_like(tXrXs[j], tXrXs[j].element_type)
            tNrN.store(value.to(tNrN.element_type))
            tNrNs.append(tNrN)
        # 3. row statistics over every chunk
        mean, factor = row_norm_stats(
            self.norm_type, tNrNs, self.num_chunks, warp_id, lane_id, D, eps
        )
        # 4. y = norm(value) * <weight> + <bias>, then * (1 + <scale>) + <shift>
        for j in cutlass.range_constexpr(self.chunks_per_warp):
            tWgW, tWrW = _slice(mWeight, thr_copy, bidx, bidy, chunk0 + j, S, D)
            tBgB, tBrB = _slice(mBias, thr_copy, bidx, bidy, chunk0 + j, S, D)
            _copy(tWgW, tWrW)
            _copy(tBgB, tBrB)
            tNrN = normalize_chunk(self.norm_type, tNrNs[j], tWrW, tBrB, mean, factor)
            value = tNrN.load()
            tSCgSC, tSCrSC = _slice(mScale, thr_copy, bidx, bidy, chunk0 + j, S, D)
            tSHgSH, tSHrSH = _slice(mShift, thr_copy, bidx, bidy, chunk0 + j, S, D)
            _copy(tSCgSC, tSCrSC)
            _copy(tSHgSH, tSHrSH)
            if cutlass.const_expr(isinstance(tSCrSC, cute.Tensor)):
                value = value * (1 + tSCrSC.load())
            if cutlass.const_expr(isinstance(tSHrSH, cute.Tensor)):
                value = value + tSHrSH.load()
            tYgY, tYrY = _slice(mY, thr_copy, bidx, bidy, chunk0 + j, S, D)
            tYrY.store(value.to(tYrY.element_type))
            _copy(tYrY, tYgY)  # rmem -> gmem


@cute.jit
def _slice(mV, thr_copy, bidx, bidy, chunk_id, S, D: cutlass.Constexpr):
    # per-thread gmem/rmem slice of one chunk of row (bidx, bidy); scalars pass through
    if cutlass.const_expr(isinstance(mV, cute.Tensor)):
        return tensor_slice_for_bsfd(mV, thr_copy, bidx, bidy, chunk_id, S, D, CHUNK)
    return mV, mV


@cute.jit
def _copy(src, dst):
    if cutlass.const_expr(
        isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)
    ):
        cute.autovec_copy(src, dst)  # LDG.128 / STG.128


@functools.lru_cache
def _chunks_per_warp(D: int, gated_affine: bool) -> int:
    """Chunks each warp walks: ~5 warps per row (B300 at Wan's 5120 is fastest there), 10 when
    the row also streams a gate and an affine weight, whose registers then cap occupancy."""
    num_chunks = D // CHUNK
    min_warps = 10 if gated_affine else 5
    per_warp = 1
    for c in range(1, num_chunks + 1):
        if num_chunks % c == 0 and num_chunks // c >= min_warps:
            per_warp = c
    return per_warp


def validate_x(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (B, S, D):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def validate_weight_bias(t: Optional[torch.Tensor], D: int):
    if t is None:
        return
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (D,):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def validate_scale_shift(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    failed = False
    if t.ndim == 1 and (t.shape[0] not in (1, D)):
        failed = True
    elif t.ndim == 2 and ((t.shape[0] not in (1, B)) or t.shape[1] != D):
        failed = True
    elif t.ndim == 3 and (
        (t.shape[0] not in (1, B)) or (t.shape[1] not in (1, S) or t.shape[2] != D)
    ):
        failed = True
    elif t.ndim == 4:
        F = t.shape[1]
        if t.shape[0] != B or t.shape[2] != 1 or t.shape[3] != D:
            failed = True
        elif S % F != 0:
            raise ValueError(f"Validate failed: S({S}) must be divisible by F({F}).")
    if failed:
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError("Validate failed: not contiguous on dim D.")


def validate_gate(t: Union[torch.Tensor, int], B: int, S: int, D: int):
    if not isinstance(t, torch.Tensor):
        return
    validate_scale_shift(t, B, S, D)


@torch.library.custom_op("sglang::fused_norm_scale_shift", mutates_args=())
def fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fuse: norm(x) * (1 + scale) + shift
      where norm is either layernorm or rmsnorm.

    Expects:
      - x: [B, S, D]
      - weight/bias: None, [D]
      - scale/shift: [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    D must be a multiple of 256 and <= 8192 to enable LDG.128 vectorized loads per
    thread and avoid predicated loads (e.g., bounds checks such as `index < D`).
    """
    from sglang.kernels.kda_kernels.norm_scale_shift_jit import (
        try_fused_norm_scale_shift as _try_qwen_native_norm_scale_shift,
    )

    native_y = _try_qwen_native_norm_scale_shift(
        x, weight, bias, scale, shift, norm_type, eps
    )
    if native_y is not None:
        return native_y
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Tensor Validation
    BSD = x.shape
    validate_x(x, *BSD)
    validate_weight_bias(weight, BSD[-1])
    validate_weight_bias(bias, BSD[-1])
    validate_scale_shift(scale, *BSD)
    validate_scale_shift(shift, *BSD)

    if norm_type == "layer" or norm_type == "rms":
        D = x.shape[-1]
        if D % 256 != 0 or D > 8192:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)  # create output tensor
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        # Use scalar placeholders for None tensors as a workaround, since the CuTe DSL
        # TVM-FFI backend does not support None parameters. scalar values do not result
        # in code generation and have no impact on runtime performance.
        weight = 1 if weight is None else weight
        bias = 0 if bias is None else bias
        ResOut, Residual, Gate = 0, 0, 1
        torch_tensors = [y, ResOut, Residual, x, Gate, weight, bias, scale, shift]
        chunks_per_warp = _chunks_per_warp(D, gated_affine=False)
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, chunks_per_warp, *torch_tensors
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D, norm_type, chunks_per_warp)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*torch_tensors, eps, stream)
        return y
    else:
        raise ValueError('norm_type must be one of "layer" and "rms"')


@fused_norm_scale_shift.register_fake
def _fused_norm_scale_shift_fake(x, weight, bias, scale, shift, norm_type, eps=1e-5):
    y = x.new_empty(x.shape)
    return y


@torch.library.custom_op(
    "sglang::fused_scale_residual_norm_scale_shift", mutates_args=()
)
def fused_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],  # Union[Optional[torch.Tensor], int] indeed
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse: norm(residual + gate * x) * (1 + scale) + shift
      where norm is either layernorm or rmsnorm.

    Expects:
      - residual, x: [B, S, D]
      - gate: None, [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - weight/bias: None, [D]
      - scale/shift: [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    D must be a multiple of 256 and <= 8192 to enable LDG.128 vectorized loads per
    thread and avoid predicated loads (e.g., bounds checks such as `index < D`).
    """
    from sglang.kernels.kda_kernels.norm_scale_shift_jit import (
        try_fused_scale_residual_norm_scale_shift as _try_qwen_native_residual_path,
    )

    native_out = _try_qwen_native_residual_path(
        residual, x, gate, weight, bias, scale, shift, norm_type, eps
    )
    if native_out is not None:
        return native_out
    # Tensor Validation
    BSD = x.shape
    validate_x(x, *BSD)
    validate_x(residual, *BSD)
    validate_gate(gate, *BSD)
    validate_weight_bias(weight, BSD[-1])
    validate_weight_bias(bias, BSD[-1])
    validate_scale_shift(scale, *BSD)
    validate_scale_shift(shift, *BSD)
    if norm_type == "layer" or norm_type == "rms":
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        D = x.shape[-1]
        if D % 256 != 0 or D > 8192:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)  # create output tensor
        resi_out = torch.empty_like(x)  # create output tensor
        gate = broadcast_tensor_for_bsfd(gate, *x.shape)  # handle various shapes
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        # Use scalar placeholders for None tensors as a workaround, since the CuTe DSL
        # TVM-FFI backend does not support None parameters. scalar values do not result
        # in code generation and have no impact on runtime performance.
        chunks_per_warp = _chunks_per_warp(
            D,
            gated_affine=isinstance(gate, torch.Tensor)
            and gate.numel() > 1
            and weight is not None,
        )
        gate = 1 if gate is None else gate
        weight = 1 if weight is None else weight
        bias = 0 if bias is None else bias
        torch_tensors = [y, resi_out, residual, x, gate, weight, bias, scale, shift]
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, chunks_per_warp, *torch_tensors
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D, norm_type, chunks_per_warp)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*torch_tensors, eps, stream)
        return y, resi_out
    else:
        raise ValueError('norm_type must be one of "layer" and "rms"')


@fused_scale_residual_norm_scale_shift.register_fake
def _fused_scale_residual_norm_scale_shift_fake(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps=1e-5
):
    y = x.new_empty(x.shape)
    residual_out = x.new_empty(x.shape)
    return y, residual_out
