"""Public interface of sglang.jit_kernel.utils.

The implementation lives in the submodules:
- common: caching decorator, CI test gating, and runtime detection
- arch: CUDA/ROCm architecture detection and default compile target flags
- compile: load_jit, the build cache, and C++ template arguments
- deps: header-only dependency registration (flashinfer, cutlass, mathdx, ...)

Only the names below are public. Internal helpers (DEFAULT_* flag lists,
ArchInfo, the dependency registry, ...) stay in their submodule; the rare
internal consumer imports them from there directly.
"""

from sglang.jit_kernel.utils.arch import (
    get_jit_cuda_arch,
    is_arch_support_pdl,
    override_jit_cuda_arch,
)
from sglang.jit_kernel.utils.common import (
    cache_once,
    get_ci_test_range,
    is_hip_runtime,
    should_run_full_tests,
)
from sglang.jit_kernel.utils.compile import (
    CPP_DTYPE_MAP,
    KERNEL_PATH,
    load_jit,
    make_cpp_args,
)
from sglang.jit_kernel.utils.deps import register_dependency

__all__ = [
    "should_run_full_tests",
    "get_ci_test_range",
    "cache_once",
    "is_hip_runtime",
    "make_cpp_args",
    "load_jit",
    "override_jit_cuda_arch",
    "get_jit_cuda_arch",
    "is_arch_support_pdl",
    "register_dependency",
    "KERNEL_PATH",
    "CPP_DTYPE_MAP",
]
