"""JIT compilation: source layout, ninja generation, the build cache, load_jit.

The package owns the whole path from a ``load_jit`` call to a loaded module,
including the ``build.ninja`` it compiles through — only tvm-ffi's headers,
its shared library, and ``tvm_ffi.load_module`` are consumed from outside.

Modules, in dependency order (there are no cycles)::

    paths      where the in-tree sources live, and the default flags
    cpp_args   rendering Python values as C++ template arguments
    spec       BuildSpec: one fully-resolved build
    toolchain  compilers, tvm-ffi locations, platform base flags
    ninja      generating and running build.ninja, reading its depfiles
    cache      build_key / deps_key, cache layout, publication
    loader     load_jit
"""

from sglang.kernels.jit.utils.compile.cpp_args import (
    CPP_DTYPE_MAP,
    CPP_TEMPLATE_TYPE,
    CPPArgList,
    make_cpp_args,
)
from sglang.kernels.jit.utils.compile.loader import load_jit
from sglang.kernels.jit.utils.compile.paths import (
    DEFAULT_CFLAGS,
    DEFAULT_INCLUDE,
    DEFAULT_LDFLAGS,
    KERNEL_PATH,
)
from sglang.kernels.jit.utils.compile.spec import BuildSpec

__all__ = [
    "BuildSpec",
    "CPPArgList",
    "CPP_DTYPE_MAP",
    "CPP_TEMPLATE_TYPE",
    "DEFAULT_CFLAGS",
    "DEFAULT_INCLUDE",
    "DEFAULT_LDFLAGS",
    "KERNEL_PATH",
    "load_jit",
    "make_cpp_args",
]
