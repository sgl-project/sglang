"""Public interface of sglang.jit_kernel.utils.

The implementation lives in the submodules:
- common: caching decorator, CI test gating, and runtime detection
- arch: CUDA/ROCm architecture detection and default compile target flags
- compile: load_jit, the build cache, and C++ template arguments
- deps: header-only dependency registration (flashinfer, cutlass, mathdx, ...)
"""

from sglang.jit_kernel.utils.arch import (
    ArchInfo,
    _get_default_target_flags,
    get_jit_cuda_arch,
    is_arch_support_pdl,
    override_jit_cuda_arch,
)
from sglang.jit_kernel.utils.common import (
    cache_once,
    get_ci_test_range,
    is_hip_runtime,
    is_musa_runtime,
    should_run_full_tests,
)
from sglang.jit_kernel.utils.compile import (
    CPP_DTYPE_MAP,
    CPP_TEMPLATE_TYPE,
    DEFAULT_CFLAGS,
    DEFAULT_INCLUDE,
    DEFAULT_LDFLAGS,
    KERNEL_PATH,
    CPPArgList,
    load_jit,
    make_cpp_args,
)
from sglang.jit_kernel.utils.deps import (
    _REGISTERED_DEPENDENCIES,
    get_cutlass_include_paths,
    get_flashinfer_include_paths,
    get_mathdx_include_paths,
    get_mathdx_root,
    register_dependency,
)

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
]
