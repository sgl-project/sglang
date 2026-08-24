"""``load_jit``: resolve a request, reuse a cached build, or make one."""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import pathlib
import shutil
import uuid
from typing import TYPE_CHECKING, List, Tuple

import torch

from sglang.kernels.jit.utils.arch import get_default_target_flags
from sglang.kernels.jit.utils.common import is_hip_runtime
from sglang.kernels.jit.utils.compile import cache, ninja
from sglang.kernels.jit.utils.compile.paths import (
    DEFAULT_CFLAGS,
    DEFAULT_INCLUDE,
    DEFAULT_LDFLAGS,
)
from sglang.kernels.jit.utils.compile.spec import BuildSpec, resolve_sources
from sglang.kernels.jit.utils.deps import REGISTERED_DEPENDENCIES

if TYPE_CHECKING:
    from tvm_ffi import Module

    _DISABLE_TORCH_COMPILE = lambda f: f
else:
    # NOTE: this is not friendly to type checking
    _DISABLE_TORCH_COMPILE = torch.compiler.disable


logger = logging.getLogger(__name__)

_LOCK_FILE = ".lock"


# JIT compilation is pure Python/filesystem plumbing (path `.resolve()` calls
# `os.lstat`, etc.) that Dynamo cannot trace. When a lazily-loaded kernel is
# first reached from inside a `@torch.compile`d region, tracing into it produces
# spurious "Dynamo does not know how to trace the builtin `posix.lstat`" graph
# breaks. The load happens once and is memoized, so keep it out of the graph.
@_DISABLE_TORCH_COMPILE
def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    header_only: bool = True,
) -> Module:
    """Load a JIT module, compiling it if no cached build still applies.

    A wrapper is a ``(export_name, kernel_name)`` pair: ``export_name`` is what
    Python calls, ``kernel_name`` is the C++ class or function it resolves to.

    :param args: Unique marker of the module. Must differ between kernels.
    :param cpp_files: C++ sources. Relative names resolve against the in-tree
                      ``csrc/`` directory; pass an absolute path for a source
                      outside the tree.
    :param cuda_files: CUDA sources, resolved like `cpp_files`.
    :param cpp_wrappers: C++ exports to generate.
    :param cuda_wrappers: CUDA exports to generate.
    :param extra_cflags: Extra host compiler flags.
    :param extra_cuda_cflags: Extra device compiler flags.
    :param extra_ldflags: Extra linker flags.
    :param extra_include_paths: Extra include paths.
    :param extra_dependencies: Registered header-only dependencies, e.g. cutlass.
    :param header_only: Compile through a generated wrapper that exports the
                        given entry points. Otherwise the sources must export
                        from the C++ side themselves.
    """
    if is_hip_runtime():
        extra_cuda_cflags = [
            flag
            for flag in (extra_cuda_cflags or [])
            if flag not in ("--use_fast_math", "-use_fast_math")
        ]

    includes = list(DEFAULT_INCLUDE) + (extra_include_paths or [])
    for dep in sorted(set(extra_dependencies or [])):
        if dep not in REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        includes += REGISTERED_DEPENDENCIES[dep]()

    module_args = tuple(str(arg) for arg in args)
    spec = BuildSpec(
        module_args=module_args,
        cpp_files=resolve_sources(cpp_files),
        cuda_files=resolve_sources(cuda_files),
        cpp_wrappers=tuple(cpp_wrappers or ()),
        cuda_wrappers=tuple(cuda_wrappers or ()),
        cflags=tuple(DEFAULT_CFLAGS + (extra_cflags or [])),
        cuda_cflags=tuple(get_default_target_flags() + (extra_cuda_cflags or [])),
        ldflags=tuple(DEFAULT_LDFLAGS + (extra_ldflags or [])),
        include_paths=tuple(includes),
        header_only=header_only,
    )

    # Generated once and threaded through: the cache key is taken over this
    # exact text, and this exact text is what gets compiled.
    build_file = ninja.generate(spec)

    build_key = cache.compute_build_key(spec, build_file=build_file)
    scope = cache.build_key_dir(module_name=spec.module_name, build_key=build_key)

    prebuilt = cache.find_prebuilt(scope=scope, module_name=spec.module_name)
    if prebuilt is not None:
        try:
            return _load(prebuilt)
        except Exception as e:
            # Also the benign case where a concurrent GC unlinked the leaf
            # between the lookup and the load.
            logger.warning(
                "Cached JIT module %s failed to load; rebuilding. " "Got error: %s",
                spec.module_name,
                e,
            )

    with _build_lock(scope):
        # Re-check: whoever held the lock before us has very likely just
        # published exactly what we were about to build. This is what turns N
        # tensor-parallel ranks starting together into one compile plus N-1
        # cache hits instead of N identical compiles.
        prebuilt = cache.find_prebuilt(scope=scope, module_name=spec.module_name)
        if prebuilt is not None:
            try:
                return _load(prebuilt)
            except Exception as e:
                logger.warning(
                    "Cached JIT module %s failed to load; rebuilding. Got error: %s",
                    spec.module_name,
                    e,
                )
                # A leaf is named after the dependency list it was built from,
                # so the rebuild below lands on this exact name and publishing
                # it would find the directory already there and keep the broken
                # one -- leaving every later process to fail twice and rebuild
                # for nothing. We hold the lock, so drop it now.
                shutil.rmtree(prebuilt.parent, ignore_errors=True)

        # Build into a private staging directory, then publish it by renaming.
        # Building in place would let another process observe a leaf that exists
        # but is still being linked — and that process is on the fast path, so it
        # takes no lock. A rename makes the leaf appear complete or not at all.
        #
        # The staging name is random rather than pid-based: the cache root is
        # meant to be a shared mount, where two machines can hold the same pid.
        staging = scope / f".staging-{uuid.uuid4().hex}"
        try:
            library = ninja.build(spec=spec, build_dir=staging, build_file=build_file)
            # Loaded before publishing, so a broken artifact never becomes a
            # leaf that later runs have to discover and discard.
            module = _load(library)
            cache.commit_build(
                spec,
                scope=scope,
                staging=staging,
                dependencies=ninja.scan_dependencies(staging),
            )
            return module
        finally:
            shutil.rmtree(staging, ignore_errors=True)


@contextlib.contextmanager
def _build_lock(scope: pathlib.Path):
    """Serialize builds of one module variant across processes.

    Every tensor-parallel rank on a node reaches the same cold cache at the same
    moment. Without this they each run a full compile: measured at 8 ranks,
    14.0 s of CPU-oversubscribed nvcc against 9.0 s when one builds and the rest
    wait and then hit the cache.

    The lock only saves duplicated work; it is not what makes publication safe
    (the atomic rename is), and readers on the fast path deliberately do not
    take it — that would serialize every warm load. Deadlock is not reachable:
    one lock, never nested, and the kernel drops it if the holder dies.
    """
    scope.mkdir(parents=True, exist_ok=True)
    handle = os.open(scope / _LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        os.close(handle)  # releases the lock


def _load(library: pathlib.Path) -> Module:
    from tvm_ffi import load_module

    return load_module(str(library))
