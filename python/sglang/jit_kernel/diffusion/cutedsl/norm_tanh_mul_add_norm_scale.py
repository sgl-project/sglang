from typing import Optional, Tuple

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


class NormTanhMulAddNormScale:
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
        NormTanhMulAddNormScale kernels.
        """

        def _sig(val):
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(self, D: int, norm_type: str, is_norm2: bool):
        self.D = D
        self.norm_type = norm_type  # "layer" or "rms"
        self.is_norm2 = is_norm2 # single norm or double norm
        self.num_warps = self.D // 256  # num of warps per cta
        self.num_threads = self.num_warps * WARP_SIZE  # num of threads per cta

    @cute.jit
    def __call__(
        self,
        mY,
        mY2,
        mX,
        mWeight,
        mBias,
        mScale,
        mShift,
        mWeight2,
        mBias2,
        mScale2,
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
            mY2,
            mX,
            mWeight,
            mBias,
            mScale,
            mShift,
            mWeight2,
            mBias2,
            mScale2,
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
        mY2,
        mX,
        mWeight,
        mBias,
        mScale,
        mShift,
        mWeight2,
        mBias2,
        mScale2,
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
        tXgX, tXrX = slice_if(mX)  # x
        tWgW, tWrW = slice_if(mWeight)  # weight
        tBgB, tBrB = slice_if(mBias)  # bias
        tSCgSC, tSCrSC = slice_if(mScale)  # scale
        tSHgSH, tSHrSH = slice_if(mShift)  # shift
        tYgY, tYrY = slice_if(mY)  # y
        if cutlass.const_expr(self.is_norm2):
            tYgY2, tYrY2 = slice_if(mY2)  # y2
            tWgW2, tWrW2 = slice_if(mWeight2)  # weight2
            tBgB2, tBrB2 = slice_if(mBias2)  # bias2
            tSCgSC2, tSCrSC2 = slice_if(mScale2)  # scale2
        # Load: load tensor from global memory to registers
        copy_if(tXgX, tXrX)  # gmem -> rmem
        copy_if(tWgW, tWrW)  # gmem -> rmem
        copy_if(tBgB, tBrB)  # gmem -> rmem
        tNrN = norm(tXrX, tWrW, tBrB)
        # Compute: value = value * tanh(<scale>) + <shift>
        copy_if(tSCgSC, tSCrSC)  # gmem -> rmem
        copy_if(tSHgSH, tSHrSH)  # gmem -> rmem
        value = tNrN.load() * cute.tanh(tSCrSC.load()) + tSHrSH.load()
        # Store: y
        tYrY.store(value.to(tYrY.element_type))
        copy_if(tYrY, tYgY)  # rmem -> gmem
        if cutlass.const_expr(self.is_norm2):
            copy_if(tWgW2, tWrW2)  # gmem -> rmem
            copy_if(tBgB2, tBrB2)  # gmem -> rmem
            tNrN2 = norm(tYrY, tWrW2, tBrB2)
            # Compute: value2 = value2 * (1 + <scale2>)
            copy_if(tSCgSC2, tSCrSC2)  # gmem -> rmem
            value2 = tNrN2.load() * (1 + tSCrSC2.load())
            # Store: y2
            tYrY2.store(value2.to(tYrY2.element_type))
            copy_if(tYrY2, tYgY2)  # rmem -> gmem
            


def validate_3d(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.ndim != 3 or (t.shape[0] not in (1, B)) or (t.shape[1] not in (1, S) or t.shape[2] != D):
        raise ValueError(f"Validate failed: unsupported 3d-tensor: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"Validate failed: not contiguous on dim D.")


def validate_weight_bias(t: Optional[torch.Tensor], D: int):
    if t is None:
        return
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (D,):
        raise ValueError(f"Validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"Validate failed: not contiguous on dim D.")


@torch._dynamo.disable  # Disable Dynamo tracing
def fused_norm_tanh_mul_add(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
    stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse: norm(x) * tanh(scale) + shift
      where norm is either layernorm or rmsnorm.

    Expects:
      - x: [B, S, D]
      - weight/bias: None, [D]
      - scale/shift: [1/B, 1/S, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    D must be a multiple of 256 and <= 8192 to enable LDG.128 vectorized loads per
    thread and avoid predicated loads (e.g., bounds checks such as `index < D`).
    """
    # Tensor Validation
    BSD = x.shape
    validate_3d(x, *BSD)
    validate_weight_bias(weight, BSD[2])
    validate_weight_bias(bias, BSD[2])
    validate_3d(scale, *BSD)
    validate_3d(shift, *BSD)
    if norm_type == "layer" or norm_type == "rms":
        D = x.shape[-1]
        if D % 256 != 0 or D > 8192:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)  # create output tensor
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        # y2, weight2, bias2, scale2 is None
        torch_tensors = [y, None, x, weight, bias, scale, shift, None, None, None]
        cute_tensor_args = [to_cute_arg(t) for t in torch_tensors]
        # Compile cache
        hash_key = NormTanhMulAddNormScale.make_hash_key(norm_type, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = NormTanhMulAddNormScale(D, norm_type, is_norm2=False)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*cute_tensor_args, eps, stream)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@torch._dynamo.disable  # Disable Dynamo tracing
def fused_norm_tanh_mul_add_norm_scale(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight2: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    scale2: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
    stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse:
      y = norm(x) * tanh(scale) + shift
      y2 = norm(y) * (1 + scale2)
      where norm is either layernorm or rmsnorm.

    Expects:
      - x: [B, S, D]
      - weight/bia/weight2/bias2: None, [D]
      - scale/shift/scale2: [1/B, 1/S, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    D must be a multiple of 256 and <= 8192 to enable LDG.128 vectorized loads per
    thread and avoid predicated loads (e.g., bounds checks such as `index < D`).
    """
    # Tensor Validation
    BSD = x.shape
    validate_3d(x, *BSD)
    validate_weight_bias(weight, BSD[2])
    validate_weight_bias(bias, BSD[2])
    validate_3d(scale, *BSD)
    validate_3d(shift, *BSD)
    validate_weight_bias(weight2, BSD[2])
    validate_weight_bias(bias2, BSD[2])
    validate_3d(scale2, *BSD)
    if norm_type == "layer" or norm_type == "rms":
        D = x.shape[-1]
        if D % 256 != 0 or D > 8192:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)  # create output tensor
        y2 = torch.empty_like(x)  # create output tensor
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        scale2 = broadcast_tensor_for_bsfd(scale2, *x.shape)  # handle various shapes
        torch_tensors = [y, y2, x, weight, bias, scale, shift, weight2, bias2, scale2]
        cute_tensor_args = [to_cute_arg(t) for t in torch_tensors]
        # Compile cache
        hash_key = NormTanhMulAddNormScale.make_hash_key(norm_type, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = NormTanhMulAddNormScale(D, norm_type, is_norm2=True)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel, *fake_sig_args, options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*cute_tensor_args, eps, stream)
        return y, y2
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
