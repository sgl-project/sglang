from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _find_cutlass_include() -> str:
    """Find CUTLASS include directory from installed packages."""
    try:
        import deep_gemm

        path = pathlib.Path(deep_gemm.__file__).parent / "include"
        if (path / "cutlass" / "cutlass.h").exists():
            return str(path)
    except ImportError:
        pass
    try:
        import flashinfer

        path = pathlib.Path(flashinfer.__file__).parent / "data" / "cutlass" / "include"
        if (path / "cutlass" / "cutlass.h").exists():
            return str(path)
    except ImportError:
        pass
    raise RuntimeError(
        "Cannot find CUTLASS include path. Install deep_gemm or flashinfer."
    )


@cache_once
def _jit_nvfp4_mm_module() -> Module:
    sm_major, sm_minor = torch.cuda.get_device_capability()
    # Blackwell (SM100+) requires the 'a' suffix to enable FP4 MMA instructions
    arch = f"{sm_major}.{sm_minor}a" if sm_major >= 10 else f"{sm_major}.{sm_minor}"

    cutlass_include = _find_cutlass_include()

    # Override TVM_FFI_CUDA_ARCH_LIST so load_jit uses the 'a' variant arch flags
    prev_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    try:
        return load_jit(
            "nvfp4_scaled_mm",
            cuda_files=["gemm/nvfp4_scaled_mm_kernels.cuh"],
            cuda_wrappers=[("cutlass_scaled_fp4_mm", "cutlass_scaled_fp4_mm")],
            extra_include_paths=[cutlass_include],
        )
    finally:
        if prev_arch is None:
            if "TVM_FFI_CUDA_ARCH_LIST" in os.environ:
                del os.environ["TVM_FFI_CUDA_ARCH_LIST"]
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = prev_arch


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    module = _jit_nvfp4_mm_module()
    module.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out
