from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fp8_cvt_module() -> Module:
    return load_jit(
        make_name("fp8_cvt"),
        cuda_files=["deepseek_v4/fp8_cvt.cuh"],
        cuda_wrappers=[("cvt_fp8_e4m3", "cvt_fp8_e4m3")],
    )


def cvt_fp8_e4m3(src: torch.Tensor) -> torch.Tensor:
    """Cast fp32 to fp8 e4m3 through the same ``pack_fp8`` the fp8 stores use.

    Nothing in the serving path calls this -- it is here so the conversion can be
    compared against torch on its own. Inside the fused kernels every value goes
    through a quantization scale first, which turns a wrong conversion byte into
    "fp8 is a bit lossy" rather than a failure.

    Args:
        src: contiguous 1D fp32 CUDA/HIP tensor of even length -- the conversion runs
            two values at a time.

    Returns:
        uint8 tensor of the same length holding the raw e4m3 bytes.

    Raises:
        RuntimeError: if the length is zero or odd, or a tensor does not match.
    """
    dst = torch.empty_like(src, dtype=torch.uint8)
    _jit_fp8_cvt_module().cvt_fp8_e4m3(dst, src)
    return dst
