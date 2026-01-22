from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.jit_kernel.cutedsl.common.norm_fusion import (
    apply_norm_cta,
    broadcast_tensor_for_bsfd,
    tensor_slice_for_bsfd,
)

_COMPILE_CACHE = {}


def _tensor_sig(tensor):
    if not isinstance(tensor, torch.Tensor):
        return type(tensor)
    return (tensor.dtype, tensor.shape, tensor.stride())


def _make_hash_key(
    norm_type: str,
    *tensors,
):
    # TODO: After upgrading the CuTe DSL version, use `make_fake_tensor` to compile
    #       the kernel with only hidden_dim (D) as a compile-time constant,
    #       avoiding specialization on B, S, or F.
    return (norm_type, *(_tensor_sig(t) for t in tensors))


class ScaleResidualNormScaleShift:
    def __init__(self, D: int):
        self.D = D

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mResOut: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mX: cute.Tensor,
        mGate: Union[Optional[cute.Tensor], cutlass.Int32, int],
        mWeight: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        mScale: Optional[cute.Tensor],
        mShift: Optional[cute.Tensor],
        norm_type: cutlass.Constexpr = "rms",
        eps: cutlass.Float32 = cutlass.Float32(1e-6),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        # Tensor shapes
        B, S, _ = mX.shape  # (batch, seq_len, hidden_dim)
        F = self.infer_frame(mGate, mScale, mShift)  # num of frame
        self.len_f = cutlass.Int32(S // F)  # len of frame
        self.norm_type = norm_type  # layernorm or rmsnorm
        # Vectorized copy configuration
        num_vectorized = 8  # maximum num of elem per copy
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        # Thread/value layouts for tiled copy
        self.num_threads = self.D // 256 * 32  # num of threads per cta
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
            S,
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
        mY: cute.Tensor,
        mResOut: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mX: cute.Tensor,
        mGate: Union[Optional[cute.Tensor], cutlass.Int32],
        mWeight: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        mScale: Optional[cute.Tensor],
        mShift: Optional[cute.Tensor],
        S: cutlass.Int32,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        tidx = cutlass.Int32(cute.arch.thread_idx()[0])  # thread index
        bid = cutlass.Int32(cute.arch.block_idx()[0])  # cta index
        bidx = cutlass.Int32(bid // S)  # batch index
        bidy = cutlass.Int32(bid % S)  # seq_len index
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def slice_if(mV):
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                return tensor_slice_for_bsfd(
                    mV, thr_copy, bidx, bidy, self.D, self.len_f
                )
            return mV, mV

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(
                isinstance(src, cute.Tensor) and isinstance(src, cute.Tensor)
            ):
                cute.autovec_copy(src, dst)

        @cute.jit
        def norm(x, weight, bias):
            return apply_norm_cta(
                self.norm_type, self.num_threads, tidx, x, weight, bias, self.D, eps
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
        elif cutlass.const_expr(isinstance(tGrG, cutlass.Int32)):
            value = tGrG * value
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

    @cute.jit
    def infer_frame(self, *tensors) -> cutlass.Int32:
        num_of_frame = 1
        for t in tensors:
            if cutlass.const_expr(isinstance(t, cute.Tensor) and len(t.shape) == 4):
                num_of_frame = t.shape[1]
        return num_of_frame


def validate_x(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (B, S, D):
        raise ValueError(f"validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"validate failed: not contiguous on dim D.")


def validate_weight_bias(t: Optional[torch.Tensor], B: int, S: int, D: int):
    if t is None:
        return
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"validate failed: unsupported dtype: {t.dtype}")
    if t.shape != (D,):
        raise ValueError(f"validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"validate failed: not contiguous on dim D.")


def validate_scale_shift(t: torch.Tensor, B: int, S: int, D: int):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"validate failed: unsupported dtype: {t.dtype}")
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
        failed = True
    if failed:
        raise ValueError(f"validate failed: unsupported tensor shape: {t.shape}.")
    if t.stride()[-1] != 1:
        raise ValueError(f"validate failed: not contiguous on dim D.")


def validate_gate(t: Union[torch.Tensor, int], B: int, S: int, D: int):
    if not isinstance(t, torch.Tensor):
        return
    validate_scale_shift(t, B, S, D)


def fused_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse: norm(x) * (1 + scale) + shift
      where norm is either layernorm or rmsnorm.

    Expects:
      - x: B, S, D]
      - weight/bias: None, [D]
      - scale/shift: [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    Supported D values (must be 256's multiple and <= 8192, etc.)
    """
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
        y = torch.empty_like(x)
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)
        # Use None as placeholders for ResOut, Residual, and Gate
        torch_tensors = [y, None, None, x, None, weight, bias, scale, shift]
        cute_tensors = [
            # TODO: Enable tvm ffi to reduce host-side overhead of `from_dlpack`
            #       once the CuTe DSL version is updated.
            from_dlpack(t, assumed_align=32) if isinstance(t, torch.Tensor) else t
            for t in torch_tensors
        ]
        # Compile cache
        hash_key = _make_hash_key(
            norm_type,
            x,
            weight,
            bias,
            scale,
            shift,
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D)
            compiled_fn = cute.compile(kernel, *cute_tensors, norm_type)
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*cute_tensors, eps=eps)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


def fused_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Union[torch.Tensor, int],
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
      - gate: 1, [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - weight/bias: None, [D]
      - scale/shift: [1], [D], [1/B, D], [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5

    Supported D values (must be 256's multiple and <= 8192, etc).
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
        D = x.shape[-1]
        if D % 256 != 0 or D > 8192:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)
        resi_out = torch.empty_like(x)
        gate = broadcast_tensor_for_bsfd(gate, *x.shape)
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)
        torch_tensors = [y, resi_out, residual, x, gate, weight, bias, scale, shift]
        cute_tensors = [
            # TODO: Enable tvm ffi to reduce host-side overhead of `from_dlpack`
            #       once the CuTe DSL version is updated.
            from_dlpack(t, assumed_align=32) if isinstance(t, torch.Tensor) else t
            for t in torch_tensors
        ]
        # Compile cache
        hash_key = _make_hash_key(
            norm_type,
            x,
            residual,
            gate,
            weight,
            bias,
            scale,
            shift,
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(D)
            compiled_fn = cute.compile(kernel, *cute_tensors, norm_type)
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(*cute_tensors, eps=eps)
        return y, resi_out
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
