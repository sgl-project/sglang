"""One fully-resolved JIT build, described in one place.

``BuildSpec`` is the hand-off between the halves of ``load_jit``: the cache has
to see *every* input that could change the output in order to key it, and the
ninja generator has to feed those same inputs to the compiler.

Anything added here that affects the generated code must also reach
``cache.compute_build_key``.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional, Tuple

import msgspec

from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

_MODULE_NAME_PREFIX = "sgl_kernel_jit_"


class TranslationUnit(msgspec.Struct, frozen=True):
    """One file handed to the compiler.

    ``source`` is set for the wrapper units sglang generates and left None for
    sources that already exist on disk; the ninja layer writes the former into
    the build directory and points at the latter where they are.
    """

    filename: str
    is_cuda: bool
    source: Optional[str] = None

    @property
    def stem(self) -> str:
        return pathlib.Path(self.filename).stem


class BuildSpec(msgspec.Struct, frozen=True):
    """Everything needed to either key or run one build."""

    module_args: Tuple[str, ...]
    cpp_files: Tuple[str, ...]
    cuda_files: Tuple[str, ...]
    cpp_wrappers: Tuple[Tuple[str, str], ...]
    cuda_wrappers: Tuple[Tuple[str, str], ...]
    cflags: Tuple[str, ...]
    cuda_cflags: Tuple[str, ...]
    ldflags: Tuple[str, ...]
    include_paths: Tuple[str, ...]
    header_only: bool

    @property
    def module_name(self) -> str:
        """Derived: the args are the module's identity, the name just spells it."""
        return _MODULE_NAME_PREFIX + "_".join(self.module_args)

    @property
    def sources(self) -> Tuple[str, ...]:
        return self.cpp_files + self.cuda_files

    @property
    def wrappers(self) -> Tuple[Tuple[str, str], ...]:
        return self.cpp_wrappers + self.cuda_wrappers

    def translation_units(self) -> List[TranslationUnit]:
        """What the compiler is actually invoked on.

        Header-only modules are compiled through a generated wrapper that
        includes the sources and exports the requested entry points; everything
        else is compiled in place and exports from the C++ side itself.
        """
        if not self.header_only:
            return [
                TranslationUnit(filename=path, is_cuda=path.endswith(".cu"))
                for path in self.sources
            ]

        units: List[TranslationUnit] = []
        for name, is_cuda, files, wrappers in (
            ("main.cpp", False, self.cpp_files, self.cpp_wrappers),
            ("cuda.cu", True, self.cuda_files, self.cuda_wrappers),
        ):
            if not files and not wrappers:
                continue
            units.append(
                TranslationUnit(
                    filename=name,
                    is_cuda=is_cuda,
                    source=_wrapper_source(files, wrappers),
                )
            )
        return units


# What tvm-ffi's own `_decorate_with_tvm_ffi` prepends to every generated unit.
# The wrapper below uses TVM_FFI_DLL_EXPORT_TYPED_FUNC, so it has to include the
# header that defines it rather than rely on the kernel's own include chain
# happening to drag it in — that dependency held for every kernel in tree, but
# it is not something a new kernel's author would know to preserve.
_FFI_INCLUDES = (
    "#include <tvm/ffi/container/tensor.h>",
    "#include <tvm/ffi/dtype.h>",
    "#include <tvm/ffi/error.h>",
    "#include <tvm/ffi/extra/c_env_api.h>",
    "#include <tvm/ffi/function.h>",
)


def _wrapper_source(
    files: Tuple[str, ...], wrappers: Tuple[Tuple[str, str], ...]
) -> str:
    lines = list(_FFI_INCLUDES)
    lines += [f'#include "{path}"' for path in files]
    lines.append("namespace sglang {")
    lines += [
        f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"
        for export_name, kernel_name in wrappers
    ]
    lines.append("}  // namespace sglang")
    return "\n".join(lines) + "\n"


def resolve_sources(files: List[str] | None) -> Tuple[str, ...]:
    """Absolute paths pass through; anything else is relative to ``csrc/``."""
    return tuple(
        str(
            path.resolve()
            if path.is_absolute()
            else (KERNEL_PATH / "csrc" / path).resolve()
        )
        for path in map(pathlib.Path, files or [])
    )
