from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from sglang.jit_kernel.diffusion.cutedsl.common.norm_fusion import (
    apply_norm_cta,
    broadcast_tensor_for_bsfd,
    tensor_slice_for_bsfd,
)
from sglang.jit_kernel.diffusion.cutedsl.utils import TORCH_TO_CUTE_DTYPE, WARP_SIZE

_COMPILE_CACHE = {}


def to_cute_arg(
    t,
    *,
    assume_aligned: Optional[int] = 32,
    use_32bit_stride: bool = False,
    enable_tvm_ffi: bool = True,
):
    """
    Convert a Python value into a CuTeDSL value.
    """
    if isinstance(t, torch.Tensor):
        return cute.runtime.from_dlpack(
            t,
            assumed_align=assume_aligned,
            use_32bit_stride=use_32bit_stride,
            enable_tvm_ffi=enable_tvm_ffi,
        )
    if isinstance(t, int):
        return cutlass.Int32(t)
    if isinstance(t, float):
        return cutlass.Float32(t)
    return t


def to_fake_cute_args(t: torch.Tensor):
    if isinstance(t, torch.Tensor):
        # Only keep the last dim as compile-time value to maximum compiled kernel reuse
        # e.g. (1,2,1536):(3027,1536,1) -> (?,?,1536):(?,?,1)
        D = t.shape[-1]
        dtype = TORCH_TO_CUTE_DTYPE[t.dtype]
        shape = (*(cute.sym_int() for _ in range(t.ndim - 1)), D)
        stride = (*(cute.sym_int(divisibility=D) for _ in range(t.ndim - 1)), 1)
        fake_t = cute.runtime.make_fake_tensor(
            dtype, shape, stride, memspace=cute.AddressSpace.gmem, assumed_align=32
        )
        return fake_t
    return to_cute_arg(t)


class ScaleResidualNormScaleShift:
    @classmethod
    def make_hash_key(cls, *inputs):
        """
        Compile-time values:
          - D: hidden dimension (size of the last dimension)
          - norm_type: layer norm or RMS norm
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

    def __init__(self, D: int, norm_type: str):
        self.D = D
        self.norm_type = norm_type  # "layer" or "rms"
        self.num_warps = self.D // 256  # num of warps per cta
        self.num_threads = self.num_warps * WARP_SIZE  # num of threads per cta

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
        # Tensor shapes
        B, S, _ = mX.shape  # (batch, seq_len, hidden_dim)
        # Vectorized copy configuration
        num_vectorized = 8  # maximum num of elem per copy
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        # Thread/value layouts for tiled copy
        t_layout = cute.make_layout(self.num_threads)  # thread layout within a CTA
        v_layout = cute.make_layout(num_vectorized)  # per-thread vector layout
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)

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
        tidx, _, _ = cute.arch.thread_idx()  # thread index
        bid, _, _ = cute.arch.block_idx()  # cta index
        bidx = cutlass.Int32(bid // S)  # batch index
        bidy = cutlass.Int32(bid % S)  # seq_len index
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def slice_if(mV):
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                return tensor_slice_for_bsfd(mV, thr_copy, bidx, bidy, S, self.D)
            return mV, mV

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(
                isinstance(src, cute.Tensor) and isinstance(src, cute.Tensor)
            ):
                cute.autovec_copy(src, dst)  # LDG.128

        @cute.jit
        def norm(x, weight, bias):
            return apply_norm_cta(
                self.norm_type, self.num_warps, tidx, x, weight, bias, self.D, eps
            )

        # Slice: retrieve the per-thread data slices for both global memory (gmem)
        # and register memory (rmem). The layouts are:
        # - ((4,2),(1)):((1,4),(0)) for fp32
        # - ((8,1),(1)):((1,0),(0)) for fp16/bf16
        tRgR, tRrR = slice_if(mRes)  # residual
        tXgX, tXrX = slice_if(mX)  # x
        tGgG, tGrG = slice_if(mGate)  # gate
        tROgRO, tROrRO = slice_if(mResOut)  # residual_out
        tWgW, tWrW = slice_if(mWeight)  # weight
        tBgB, tBrB = slice_if(mBias)  # bias
        tSCgSC, tSCrSC = slice_if(mScale)  # scale
        tSHgSH, tSHrSH = slice_if(mShift)  # shift
        tYgY, tYrY = slice_if(mY)  # y
        # Load: load tensor from global memory to registers
        copy_if(tRgR, tRrR)  # gmem -> rmem
        copy_if(tXgX, tXrX)  # gmem -> rmem
        copy_if(tGgG, tGrG)  # gmem -> rmem
        copy_if(tWgW, tWrW)  # gmem -> rmem
        copy_if(tBgB, tBrB)  # gmem -> rmem

        # For norm_scale_shift, output:
        # - y = norm(x, weight, bias) * (1 + scale) + shift
        # For scale_residual_norm_scale_shift, output:
        # - residual_out = residual + gate * x
        # - y = norm(residual_out, weight, bias) * (1 + scale) + shift
        # Compute: value = <gate> * x
        value = tXrX.load()
        if cutlass.const_expr(isinstance(tGrG, cute.Tensor)):
            value = tGrG.load() * value
        # Compute: value = value + <residual>
        if cutlass.const_expr(isinstance(tRrR, cute.Tensor)):
            value = value + tRrR.load()
        # Store: residual_out
        if cutlass.const_expr(isinstance(tROrRO, cute.Tensor)):
            tROrRO.store(value.to(tROrRO.element_type))
            copy_if(tROrRO, tROgRO)  # rmem -> gmem
        # Compute: value = norm(value) * <weight> + <bias>
        tNrN = cute.make_rmem_tensor_like(tXrX, tXrX.element_type)
        tNrN.store(value.to(tNrN.element_type))
        tNrN = norm(tNrN, tWrW, tBrB)
        # Compute: value = value * (1 + <scale>) + <shift>
        value = tNrN.load()
        copy_if(tSCgSC, tSCrSC)  # gmem -> rmem
        copy_if(tSHgSH, tSHrSH)  # gmem -> rmem
        if cutlass.const_expr(isinstance(tSCrSC, cute.Tensor)):
            value = value * (1 + tSCrSC.load())
        if cutlass.const_expr(isinstance(tSHrSH, cute.Tensor)):
            value = value + tSHrSH.load()
        # Store: y
        tYrY.store(value.to(tYrY.element_type))
        copy_if(tYrY, tYgY)  # rmem -> gmem


def validate_x(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (B, S, D):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"Validate failed: not contiguous on dim D.")


def validate_weight_bias(t: Optional[torch.Tensor], B: int, S: int, D: int):
    if t is None:
        return
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (D,):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"Validate failed: not contiguous on dim D.")


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
    elif t.ndim == 4 and (t.shape[0] != B or t.shape[2] != 1 or t.shape[3] != D):
        F = t.shape[1]
        if S % F != 0:
            raise ValueError(f"Validate failed: S({S}) must be divisible by F({F}).")
        failed = True
    if failed:
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"Validate failed: not contiguous on dim D.")


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
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Tensor Validation
    BSD = x.shape
    validate_x(x, *BSD)
    validate_weight_bias(weight, *BSD)
    validate_weight_bias(bias, *BSD)
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
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(norm_type, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*torch_tensors, eps, stream)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_norm_scale_shift.register_fake
def _fused_norm_scale_shift_fake(x, weight, bias, scale, shift, norm_type, eps):
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
    # Tensor Validation
    BSD = x.shape
    validate_x(x, *BSD)
    validate_x(residual, *BSD)
    validate_gate(gate, *BSD)
    validate_weight_bias(weight, *BSD)
    validate_weight_bias(bias, *BSD)
    validate_scale_shift(scale, *BSD)
    validate_scale_shift(shift, *BSD)
    if norm_type == "layer" or norm_type == "rms":
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # if norm_type == "layer" or norm_type == "rms":
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
        gate = 1 if gate is None else gate
        weight = 1 if weight is None else weight
        bias = 0 if bias is None else bias
        torch_tensors = [y, resi_out, residual, x, gate, weight, bias, scale, shift]
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(norm_type, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*torch_tensors, eps, stream)
        return y, resi_out
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_scale_residual_norm_scale_shift.register_fake
def _fused_scale_residual_norm_scale_shift_fake(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    y = x.new_empty(x.shape)
    residual_out = x.new_empty(x.shape)
    return y, residual_out
