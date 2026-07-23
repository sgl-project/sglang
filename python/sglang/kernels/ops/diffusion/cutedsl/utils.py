from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

WARP_SIZE = 32

TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


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
