from enum import Enum
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

class KernelEnum(Enum):
    NormScaleShift = 0 # NormScaleShift
    ScaleResidualNormScaleShift = 1 # ScaleResidualNormScaleShift
    DualNormScaleShift = 2 # DualNormScaleShift
    DualScaleResidualNormScaleShift = 3 # DualScaleResidualNormScaleShift

# CuTeDSL annotation
# y, x, weight, bias, scale, shift
NormScaleShiftParams = Tuple[*([cute.Tensor] * 6)]
DualNormScaleShiftParams = Tuple[*([cute.Tensor] * 6 * 2)]
# y, res_out, res, x, gate, weight, bias, scale, shift
ScaleResidualNormScaleShiftParams = Tuple[*([cute.Tensor] * 9)]
DualScaleResidualNormScaleShiftParams = Tuple[*([cute.Tensor] * 9 * 2)]

class ScaleResidualNormScaleShift():
    @classmethod
    def make_hash_key(cls, *inputs):
        def _sig(val):
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(self, batch: int, D: int, norm_type: str):
        self.batch = batch
        self.D = D
        self.num_vectorized = 8  # maximum num of elem per copy
        self.norm_type = norm_type  # "layer" or "rms"
        self.num_threads = self.heuristic_threads()  # num of threads per cta
        self.num_warps = self.num_threads // WARP_SIZE  # num of warps per cta

    def heuristic_threads(self):
        elems_per_warp = self.num_vectorized * WARP_SIZE
        heu_warps = (self.D + elems_per_warp - 1) // elems_per_warp // 4
        heu_warps = max(heu_warps, 1) # at least one warp
        heu_warps = (heu_warps + 1) // 2 * 2 # be multiple of 2
        heu_threads = heu_warps * 32
        return heu_threads

    @cute.jit
    def get_tiled_copy(self, dtype):
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128,
        )
        t_layout = cute.make_layout(self.num_threads)
        v_layout = cute.make_layout(self.num_vectorized)
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)
        return tiled_copy

    @cute.jit
    def norm_scale_shift(
        self,
        tensors: NormScaleShiftParams,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel_enum = KernelEnum.NormScaleShift
        mY = tensors[0]
        _, S, _ = mY.shape
        tiled_copy = self.get_tiled_copy(mY.element_type)
        self.kernel(tensors, tiled_copy, eps).launch(
            grid=[self.batch * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def dual_norm_scale_shift(
        self,
        tensors: DualNormScaleShiftParams,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel_enum = KernelEnum.DualNormScaleShift
        mY1, mY2 = tensors[0], tensors[6] # [batch, s1/s2, hidden]
        S1, S2 = mY1.shape[1], mY2.shape[1]
        num_cta_1 = self.batch * S1
        tiled_copy = self.get_tiled_copy(mY1.element_type)
        self.kernel(tensors + [num_cta_1], tiled_copy, eps).launch(
            grid=[self.batch * (S1 + S2), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def scale_residual_norm_scale_shift(
        self,
        tensors: ScaleResidualNormScaleShiftParams,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel_enum = KernelEnum.ScaleResidualNormScaleShift
        mY = tensors[0]
        _, S, _ = mY.shape
        tiled_copy = self.get_tiled_copy(mY.element_type)
        self.kernel(tensors, tiled_copy, eps).launch(
            grid=[self.batch * S, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def dual_scale_residual_norm_scale_shift(
        self,
        tensors: DualScaleResidualNormScaleShiftParams,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.kernel_enum = KernelEnum.DualScaleResidualNormScaleShift
        mY1, mY2 = tensors[0], tensors[9] # [batch, s1/s2, hidden]
        S1, S2 = mY1.shape[1], mY2.shape[1]
        num_cta_1 = self.batch * S1
        tiled_copy = self.get_tiled_copy(mY1.element_type)
        self.kernel(tensors + [num_cta_1], tiled_copy, eps).launch(
            grid=[self.batch * (S1 + S2), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tensors,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        has_res = (
            self.kernel_enum == KernelEnum.ScaleResidualNormScaleShift or
            self.kernel_enum == KernelEnum.DualScaleResidualNormScaleShift
        )
        tidx, _, _ = cute.arch.thread_idx()  # thread index
        bid, _, _ = cute.arch.block_idx()  # cta index
        if cutlass.const_expr(
            self.kernel_enum == KernelEnum.NormScaleShift
        ):
            mY, mX, mWeight, mBias, mScale, mShift = tensors
        elif cutlass.const_expr(
            self.kernel_enum == KernelEnum.ScaleResidualNormScaleShift
        ):
            mY, mResOut, mRes, mX, mGate, mWeight, mBias, mScale, mShift = tensors
        elif cutlass.const_expr(
            self.kernel_enum == KernelEnum.DualNormScaleShift
        ):
            mY, mX, mWeight, mBias, mScale, mShift = tensors[:6]
            num_cta_1 = tensors[12]
            if bid >= num_cta_1:
                bid -= num_cta_1
                mY, mX, mWeight, mBias, mScale, mShift = tensors[6:12]
        elif cutlass.const_expr(
            self.kernel_enum == KernelEnum.DualScaleResidualNormScaleShift
        ):
            num_cta_1 = tensors[18]
            mY, mResOut, mRes, mX, mGate, mWeight, mBias, mScale, mShift = tensors[:9]
            if bid >= num_cta_1:
                bid -= num_cta_1
                mY, mResOut, mRes, mX, mGate, mWeight, mBias, mScale, mShift = tensors[9:18]
        _, S, _ = mX.shape
        
        if cutlass.const_expr(self.batch == 1):
            bidx, bidy = 0, bid
        else:
            bidx = cutlass.Int32(bid // S)  # batch index
            bidy = cutlass.Int32(bid % S)  # seq_len index
        thr_copy = tiled_copy.get_slice(tidx)

        @cute.jit
        def slice_if(mV):
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                return tensor_slice_for_bsfd(mV, thr_copy, bidx, bidy, S, self.D)
            return mV, mV

        @cute.jit
        def copy_if(src, dst, pred, fill_val=None):
            if cutlass.const_expr(
                isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)
            ):
                for i in range(cute.size(src, mode=[1])):
                    if pred[i]:
                        cute.autovec_copy(src[None, i], dst[None, i])
                    else:
                        if cutlass.const_expr(fill_val is not None):
                            dst.fill(fill_val)
        
        if cutlass.const_expr(has_res):
            tRgR, tRrR = slice_if(mRes)  # residual
            tGgG, tGrG = slice_if(mGate)  # gate
            tROgRO, _ = slice_if(mResOut)  # residual_out
        tXgX, tXrX = slice_if(mX)  # x
        tWgW, tWrW = slice_if(mWeight)  # weight
        tBgB, tBrB = slice_if(mBias)  # bias
        tSCgSC, tSCrSC = slice_if(mScale)  # scale
        tSHgSH, tSHrSH = slice_if(mShift)  # shift
        tYgY, tYrY = slice_if(mY)  # y

        pred = cute.make_rmem_tensor(cute.size(tYgY, mode=[1]), cutlass.Boolean)
        for i in range(cute.size(pred)):
            offset = (i * self.num_threads + tidx) * self.num_vectorized
            pred[i] = offset < self.D
        if cutlass.const_expr(has_res):
            copy_if(tGgG, tGrG, pred, fill_val=0.0)
            copy_if(tRgR, tRrR, pred, fill_val=0.0)
        copy_if(tXgX, tXrX, pred, fill_val=0.0)
        copy_if(tWgW, tWrW, pred)
        copy_if(tBgB, tBrB, pred)

        if cutlass.const_expr(has_res):
            if cutlass.const_expr(isinstance(tGrG, cute.Tensor)):
                value = tGrG.load().to(cutlass.Float32) * tXrX.load()
                tXrX.store(value.to(tXrX.element_type))
            if cutlass.const_expr(isinstance(tRrR, cute.Tensor)):
                value = tXrX.load().to(cutlass.Float32) + tRrR.load()
                tXrX.store(value.to(tXrX.element_type))
            if cutlass.const_expr(isinstance(tROgRO, cute.Tensor)):
                copy_if(tXrX, tROgRO, pred)
        tNrN = apply_norm_cta(
            self.norm_type, self.num_warps, tidx, tXrX, tWrW, tBrB, pred, self.D, eps
        )
        copy_if(tSCgSC, tSCrSC, pred)
        copy_if(tSHgSH, tSHrSH, pred)
        value = tNrN.load()
        if cutlass.const_expr(isinstance(tSCrSC, cute.Tensor)):
            value = value * (1 + tSCrSC.load())
        if cutlass.const_expr(isinstance(tSHrSH, cute.Tensor)):
            value = value + tSHrSH.load()
        tYrY.store(value.to(tYrY.element_type))
        copy_if(tYrY, tYgY, pred)


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
        batch, _, D = x.shape
        if D % 8 != 0 or D > 16384:
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
        torch_tensors = [y, x, weight, bias, scale, shift]
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, batch, *torch_tensors
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(batch, D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel.norm_scale_shift,
                fake_sig_args,
                eps,  # eps: runtime value
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(torch_tensors, eps, stream)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_norm_scale_shift.register_fake
def _fused_norm_scale_shift_fake(x, weight, bias, scale, shift, norm_type, eps):
    y = torch.empty_like(x)
    return y


@torch.library.custom_op("sglang::fused_dual_norm_scale_shift", mutates_args=())
def fused_dual_norm_scale_shift(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    x_2: torch.Tensor,
    weight_2: Optional[torch.Tensor],
    bias_2: Optional[torch.Tensor],
    scale_2: torch.Tensor,
    shift_2: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        batch, _, D = x.shape
        if D % 8 != 0 or D > 16384:
            raise ValueError(
                f"D={D} not supported, must be multiple of 8 and <= 16384"
            )
        y = torch.empty_like(x)  # create output tensor
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        weight = 1 if weight is None else weight
        bias = 0 if bias is None else bias
        torch_tensors = [y, x, weight, bias, scale, shift]
        y_2 = torch.empty_like(x_2)  # create output tensor
        scale_2 = broadcast_tensor_for_bsfd(scale_2, *x_2.shape)  # handle various shapes
        shift_2 = broadcast_tensor_for_bsfd(shift_2, *x_2.shape)  # handle various shapes
        weight_2 = 1 if weight_2 is None else weight_2
        bias_2 = 0 if bias_2 is None else bias_2
        torch_tensors += [y_2, x_2, weight_2, bias_2, scale_2, shift_2]
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, batch, *torch_tensors
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(batch, D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel.dual_norm_scale_shift,
                fake_sig_args,
                eps,  # eps: runtime value
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(torch_tensors, eps, stream)
        return y, y_2
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_dual_norm_scale_shift.register_fake
def _fused_dual_norm_scale_shift_fake(
    x, weight, bias, scale, shift,
    x_2, weight_2, bias_2, scale_2, shift_2,
    norm_type, eps
):
    y_1 = torch.empty_like(x)
    y_2 = torch.empty_like(x_2)
    return y_1, y_2


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
        batch, _, D = x.shape
        if D % 8 != 0 or D > 16384:
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
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, batch, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(batch, D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel.scale_residual_norm_scale_shift,
                fake_sig_args,
                eps,  # eps: runtime value
                stream,
                options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(torch_tensors, eps, stream)
        return y, resi_out
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_scale_residual_norm_scale_shift.register_fake
def _fused_scale_residual_norm_scale_shift_fake(
    residual, x, gate, weight, bias, scale, shift, norm_type, eps
):
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    return y, residual_out


@torch.library.custom_op(
    "sglang::fused_dual_scale_residual_norm_scale_shift", mutates_args=()
)
def fused_dual_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],  # Union[Optional[torch.Tensor], int] indeed
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    residual_2: torch.Tensor,
    x_2: torch.Tensor,
    gate_2: Optional[torch.Tensor],  # Union[Optional[torch.Tensor], int] indeed
    weight_2: Optional[torch.Tensor],
    bias_2: Optional[torch.Tensor],
    scale_2: torch.Tensor,
    shift_2: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        batch, _, D = x.shape
        if D % 8 != 0 or D > 16384:
            raise ValueError(
                f"D={D} not supported, must be multiple of 256 and <= 8192"
            )
        y = torch.empty_like(x)  # create output tensor
        resi_out = torch.empty_like(x)  # create output tensor
        gate = broadcast_tensor_for_bsfd(gate, *x.shape)  # handle various shapes
        scale = broadcast_tensor_for_bsfd(scale, *x.shape)  # handle various shapes
        shift = broadcast_tensor_for_bsfd(shift, *x.shape)  # handle various shapes
        gate = 1 if gate is None else gate
        weight = 1 if weight is None else weight
        bias = 0 if bias is None else bias
        y_2 = torch.empty_like(x_2)  # create output tensor
        resi_out_2 = torch.empty_like(x_2)  # create output tensor
        gate_2 = broadcast_tensor_for_bsfd(gate_2, *x_2.shape)  # handle various shapes
        scale_2 = broadcast_tensor_for_bsfd(scale_2, *x_2.shape)  # handle various shapes
        shift_2 = broadcast_tensor_for_bsfd(shift_2, *x_2.shape)  # handle various shapes
        gate_2 = 1 if gate_2 is None else gate_2
        weight_2 = 1 if weight_2 is None else weight_2
        bias_2 = 0 if bias_2 is None else bias_2
        torch_tensors = [y, resi_out, residual, x, gate, weight, bias, scale, shift]
        torch_tensors += [
            y_2, resi_out_2, residual_2, x_2, gate_2, weight_2, bias_2, scale_2, shift_2
        ]
        # Compile cache
        hash_key = ScaleResidualNormScaleShift.make_hash_key(
            norm_type, batch, *torch_tensors)
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            kernel = ScaleResidualNormScaleShift(batch, D, norm_type)
            fake_sig_args = [to_fake_cute_args(t) for t in torch_tensors]
            compiled_fn = cute.compile(
                kernel.dual_scale_residual_norm_scale_shift,
                fake_sig_args,
                eps,  # eps: runtime value
                stream,
                options="--enable-tvm-ffi"
            )
            _COMPILE_CACHE[hash_key] = compiled_fn
        # Execute
        compiled_fn(torch_tensors, eps, stream)
        return y, resi_out, y_2, resi_out_2
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')


@fused_dual_scale_residual_norm_scale_shift.register_fake
def _fused_dual_scale_residual_norm_scale_shift_fake(
    residual, x, gate, weight, bias, scale, shift,
    residual_2, x_2, gate_2, weight_2, bias_2, scale_2, shift_2,
    norm_type, eps
):
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    y_2 = torch.empty_like(x_2)
    residual_out_2 = torch.empty_like(x_2)
    return y, residual_out, y_2, residual_out_2

