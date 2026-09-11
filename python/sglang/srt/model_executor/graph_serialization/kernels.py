# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kernel identity and re-resolution (design sections 6.5 and 8).

A saved ``KernelNode`` never stores a ``CUfunction``. Handles are process
local and, under the default lazy module loading, a fresh process cannot even
look a kernel up by name until it holds a module or library handle (fact 5).
The save side therefore maps every live handle to a durable
:class:`~sglang.srt.model_executor.graph_serialization.format.KernelIdentity`
through a chain of *providers*, and the load side turns the identity back into
a ``(CUfunction, CUkernel)`` pair through the same providers, tried in
:data:`RESOLUTION_ORDER`.

Kernel names repeat across translation units (fact 11), so the primary key is
``(container sha256, kernel name)``. Kernels with no bytes on disk (cuBLASLt
``nvjet_*`` and CuTe-DSL, fact 12) carry a live-only identity made of the
owning module's name-set digest, the name and the function attributes; they
are either captured through CUPTI at save (fact 20) or harvested from
throwaway captures at load. Function attributes live on the ``CUfunction``
and must be re-applied after re-resolution (fact 8).

In this draft the four providers are documented skeletons.
:class:`KernelResolver` (ordering, fall-through, error reporting) is pure and
unit tested with fake providers; :meth:`KernelResolver.apply_func_attrs` is
the one small, guarded driver call.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from cuda.bindings import driver as cuda_drv
except ImportError:
    cuda_drv = None

from sglang.srt.model_executor.graph_serialization.format import KernelIdentity
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)

logger = logging.getLogger(__name__)

# Load-time order (design section 6.5): bytes captured at save, then bytes found
# on disk (ELF sections, JIT caches), then live handles harvested from throwaway
# captures. A provider whose ``name`` is not listed here is tried last, in
# construction order.
RESOLUTION_ORDER = ("captured_bytes", "elf_section", "jit_cache", "live_harvest")

# The ``CU_FUNC_ATTRIBUTE_*`` values ``cuFuncSetAttribute`` accepts. Every other
# attribute (``NUM_REGS``, ``MAX_THREADS_PER_BLOCK``, ``PTX_VERSION``, ...) is
# read-only: the resolver records them for the live-harvest confirmation
# (design section 6.5) and skips them when re-applying.
SETTABLE_FUNC_ATTR_NAMES = (
    "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
    "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED",
    "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
)


def _not_implemented(where: str, what: str, section: str) -> NotImplementedError:
    return NotImplementedError(
        f"{where}: {what} is not implemented in this draft; see "
        f"DESIGN_cuda_graph_serialization.md section {section}"
    )


class KernelUnresolved(RuntimeError):
    """No provider could identify a live handle or resolve a saved identity.

    At save this makes the graph ``needs_recapture``; at load it makes the
    shape fall back to capture (design section 6.5, resolution order).
    """


class KernelProvider(Protocol):
    """One tier of the kernel identity chain (design section 6.5)."""

    name: str

    def identify(self, func: int, kern: int) -> KernelIdentity | None:
        """SAVE: map a live ``(CUfunction, CUkernel)`` pair to a durable
        identity, or ``None`` when this provider cannot (for example no bytes
        on disk)."""

    def resolve(
        self, ident: KernelIdentity, images: Mapping[str, bytes]
    ) -> tuple[int, int] | None:
        """LOAD: return ``(CUfunction, CUkernel or 0)`` for ``ident`` using the
        content-addressed ``images`` (``sha256 -> container bytes``), or
        ``None`` when this provider does not know the identity."""


class ElfSectionProvider:
    """Tier: kernels whose container is a fatbin section of a mapped shared
    object (design section 6.5).

    SAVE: ``cuFuncGetModule`` / ``cuKernelGetLibrary``, enumerate the module's
    kernel names, and match the unique container among the ``.nv_fatbin`` and
    ``.cask_resource`` sections of the shared objects mapped in
    ``/proc/self/maps`` (per-file section index cached on disk by path, size
    and mtime). Records ``image_sha256`` and the container bytes into the
    artifact's image store. A module's name-set identified exactly one
    container in every probe, but names alone repeat across translation units,
    so the identity key is ``(sha256 of container bytes, kernel name)``
    (fact 11). Foundry's fatbin-size walk over-reads concatenated sections;
    the section table is authoritative.

    LOAD: ``cuLibraryLoadData(bytes)`` once per used image in the primary
    context, then ``cuLibraryGetKernel`` and ``cuKernelGetFunction``; the
    caller re-applies ``func_attrs``. About 0.5 ms and 85 KiB per image, zero
    device memory until a function is materialized (fact 11). Covers
    libtorch_cuda, sgl_kernel, flashinfer AOT and JIT ``.so`` files, cuBLAS,
    cuBLASLt cutlass kernels and cuDNN.
    """

    name = "elf_section"

    def __init__(self, *, index_cache_dir: str | None = None) -> None:
        self.index_cache_dir = index_cache_dir

    def identify(self, func: int, kern: int) -> KernelIdentity | None:
        raise _not_implemented(
            "ElfSectionProvider.identify",
            "matching a live module's name-set against the fatbin sections of "
            "mapped shared objects",
            "6.5",
        )

    def resolve(
        self, ident: KernelIdentity, images: Mapping[str, bytes]
    ) -> tuple[int, int] | None:
        raise _not_implemented(
            "ElfSectionProvider.resolve",
            "cuLibraryLoadData of the saved container and cuLibraryGetKernel by name",
            "6.5",
        )


class JitCacheProvider:
    """Tier: kernels compiled at runtime whose container is on disk (design
    section 6.5).

    Sources, each with its own lookup: Triton (cubin path reported by the
    ``triton.knobs`` ``kernel_load_start_hook``, which must be armed before
    capture); DeepGEMM under ``SGLANG_DG_CACHE_DIR`` (defaults under
    ``SGLANG_CACHE_DIR``); ``sglang.kernels.jit`` under ``SGLANG_JIT_CACHE_DIR``
    (default ``~/.cache/sglang/jit``, independent of ``SGLANG_CACHE_DIR``, so
    the ``.so`` path the loader returns is recorded rather than derived; fact
    22); FlashInfer JIT under ``$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/
    <ver>/<arch>/cached_ops``, valid only when sglang is imported before
    flashinfer. The container bytes are embedded in the artifact's image store
    so a load never depends on the cache directory still existing.
    """

    name = "jit_cache"

    def __init__(
        self,
        *,
        recorded_paths: Mapping[str, str] | None = None,
        cache_dirs: Sequence[str] = (),
    ) -> None:
        # kernel name -> container path, as reported by the loader hooks.
        self.recorded_paths: dict[str, str] = dict(recorded_paths or {})
        self.cache_dirs: tuple[str, ...] = tuple(cache_dirs)

    def identify(self, func: int, kern: int) -> KernelIdentity | None:
        raise _not_implemented(
            "JitCacheProvider.identify",
            "mapping a live handle to a Triton / DeepGEMM / sglang JIT / "
            "FlashInfer JIT container on disk",
            "6.5",
        )

    def resolve(
        self, ident: KernelIdentity, images: Mapping[str, bytes]
    ) -> tuple[int, int] | None:
        raise _not_implemented(
            "JitCacheProvider.resolve",
            "cuLibraryLoadData of the embedded JIT container and lookup by name",
            "6.5",
        )


class CapturedBytesProvider:
    """Tier: module bytes captured in-process at save through CUPTI (design
    section 6.5, verified in fact 20). Optional.

    ``CUPTI_CBID_RESOURCE_MODULE_LOADED`` delivers the decompressed cubin of
    every module at materialization, including ``nvjet`` and CuTe-DSL kernels
    that have no bytes on disk. cuda-python has no CUPTI binding; the
    subscription goes through ``ctypes`` against torch's own ``libcupti`` and
    must be armed at process start, before the CUDA context, because modules
    materialized before arming are never delivered.

    Every delivered cubin is copied into a content-addressed in-memory store
    (dedupe by sha256: the same ``nvjet`` module materializes twice) but only
    images that a graph node with a live-only identity resolves to are
    persisted. The match is the module's ``cuFuncGetModule`` name-set against
    ``cuLibraryEnumerateKernels`` of the captured bytes, because CUPTI reports
    a ``uint32`` module id, never a ``CUmodule``. Delivered bytes are
    decompressed (24 MB for a 6.8 MB fatbin), so capture-all is never written.

    Constraints: ``cuptiSubscribe`` is process-exclusive and the conflict with
    ``torch.profiler`` (kineto) is sticky in both directions, so arming is
    refused while profiling is enabled; CUPTI error 39 is a loud fallback to
    :class:`LiveHarvestProvider`; the callback runs on a non-Python thread and
    must be implemented in C for production (a Python callback would deadlock
    on the GIL; the probe paid about 1.3 ms per MB hashing synchronously). An
    opt-in interposer shared object remains the alternative. CUPTI is never
    needed at load: ``resolve`` only loads the persisted bytes from ``images``.
    """

    name = "captured_bytes"

    def __init__(self, *, profiling_enabled: bool = False) -> None:
        self.profiling_enabled = profiling_enabled
        self.armed = False
        # sha256 -> decompressed cubin, in-memory only until a graph node needs it.
        self.store: dict[str, bytes] = {}

    def arm(self) -> None:
        """Subscribe to CUPTI module-load callbacks before the CUDA context."""
        raise _not_implemented(
            "CapturedBytesProvider.arm",
            "the ctypes CUPTI subscription on torch's libcupti",
            "6.5",
        )

    def identify(self, func: int, kern: int) -> KernelIdentity | None:
        raise _not_implemented(
            "CapturedBytesProvider.identify",
            "matching a live module's name-set against captured cubins",
            "6.5",
        )

    def resolve(
        self, ident: KernelIdentity, images: Mapping[str, bytes]
    ) -> tuple[int, int] | None:
        raise _not_implemented(
            "CapturedBytesProvider.resolve",
            "cuLibraryLoadData of the persisted captured cubin and lookup by name",
            "6.5",
        )


class LiveHarvestProvider:
    """Tier: LOAD fallback for live-only identities (design section 6.5).

    One throwaway ``keep_graph`` capture per (phase, variant) at the largest
    shape seeds a registry of live handles; then, for each shape with
    still-unresolved kernels, one throwaway capture of that shape. Lazy module
    loading is module-granular, so one launch resolves every sibling kernel of
    the module (fact 5). Handles are matched by ``(module_names_digest,
    name)`` and confirmed against the saved ``func_attrs`` (``NUM_REGS``) and
    ``param_layout``. ``forwards_used`` reports the forward count for the
    :class:`~sglang.srt.model_executor.graph_serialization.safety.CoverageReport`.

    Without byte capture, ``nvjet`` GEMM kernels force one harvest capture per
    shape whose tile pick is new (largest first, so a handful; design section
    8). ``identify`` always answers ``None``: a live handle with no bytes gets
    its live-only identity from the codec, not from this provider.
    """

    name = "live_harvest"

    def __init__(self, *, harvest_capture: Callable[[Any], int] | None = None) -> None:
        # Callable that runs one throwaway capture of a shape and returns its
        # raw CUgraph; supplied by the materializer.
        self.harvest_capture = harvest_capture
        self.forwards_used = 0
        # (module_names_digest, name) -> (CUfunction, CUkernel)
        self.registry: dict[tuple[str, str], tuple[int, int]] = {}

    def identify(self, func: int, kern: int) -> KernelIdentity | None:
        raise _not_implemented(
            "LiveHarvestProvider.identify",
            "the live-only identity (module name-set digest, name, function "
            "attributes) of a handle with no bytes",
            "6.5",
        )

    def resolve(
        self, ident: KernelIdentity, images: Mapping[str, bytes]
    ) -> tuple[int, int] | None:
        raise _not_implemented(
            "LiveHarvestProvider.resolve",
            "throwaway harvest captures and the (module_names_digest, name) match",
            "6.5",
        )


def _resolution_rank(provider: KernelProvider) -> int:
    try:
        return RESOLUTION_ORDER.index(provider.name)
    except ValueError:
        return len(RESOLUTION_ORDER)


class KernelResolver:
    """Chain of :class:`KernelProvider` tiers (design section 6.5).

    ``identify`` tries providers in construction order (the save side decides
    which tier owns a handle); ``resolve`` tries them in
    :data:`RESOLUTION_ORDER` by ``name``, unknown names last. The first
    non-``None`` answer wins; when none answers, :class:`KernelUnresolved`
    names the identity and every provider tried so the graph verdict can
    carry the reason.
    """

    def __init__(
        self,
        providers: Sequence[KernelProvider],
        images: Mapping[str, bytes] | None = None,
    ) -> None:
        self._providers: tuple[KernelProvider, ...] = tuple(providers)
        self._images: dict[str, bytes] = dict(images or {})

    @property
    def providers(self) -> tuple[KernelProvider, ...]:
        return self._providers

    @property
    def images(self) -> Mapping[str, bytes]:
        return self._images

    def ordered_providers(self) -> tuple[KernelProvider, ...]:
        """Providers in load order: :data:`RESOLUTION_ORDER`, then the rest in
        construction order (stable sort)."""
        return tuple(sorted(self._providers, key=_resolution_rank))

    def identify(self, func: int, kern: int) -> KernelIdentity:
        for provider in self._providers:
            ident = provider.identify(func, kern)
            if ident is not None:
                return ident
        tried = ", ".join(p.name for p in self._providers) or "<no providers>"
        raise KernelUnresolved(
            f"no kernel provider identified func={int(func):#x} "
            f"kern={int(kern):#x}; tried: {tried}"
        )

    def resolve(self, ident: KernelIdentity) -> tuple[int, int]:
        tried: list[str] = []
        for provider in self.ordered_providers():
            tried.append(provider.name)
            handles = provider.resolve(ident, self._images)
            if handles is not None:
                func, kern = handles
                return int(func), int(kern)
        raise KernelUnresolved(
            f"kernel {ident.name!r} (image_sha256={ident.image_sha256}, "
            f"module_names_digest={ident.module_names_digest!r}) could not be "
            f"resolved; tried: {', '.join(tried) or '<no providers>'}"
        )

    def apply_func_attrs(self, func: int, ident: KernelIdentity) -> None:
        """Re-apply the saved function attributes to a re-resolved function.

        Attributes live on the ``CUfunction``, not the graph node, so a
        function obtained from a fresh ``cuLibraryLoadData`` starts at its
        defaults (fact 8). Only the attributes ``cuFuncSetAttribute`` accepts
        (:data:`SETTABLE_FUNC_ATTR_NAMES`) are written; read-only ones such as
        ``NUM_REGS`` are confirmation data for the live-harvest tier. An
        attribute whose live value already equals the saved one is left alone,
        which also avoids ``CUDA_ERROR_NOT_PERMITTED`` on kernels with
        compile-time ``__cluster_dims__``.
        """
        if cuda_drv is None:
            raise _not_implemented(
                "KernelResolver.apply_func_attrs",
                "cuFuncSetAttribute without cuda.bindings",
                "6.5",
            )
        settable: dict[int, Any] = {}
        for attr_name in SETTABLE_FUNC_ATTR_NAMES:
            attr = getattr(cuda_drv.CUfunction_attribute, attr_name, None)
            if attr is not None:
                settable[int(attr)] = attr
        for attr, value in ident.func_attrs:
            attr = int(attr)
            if attr not in settable:
                continue
            attrib = settable[attr]
            current = checkCudaErrors(cuda_drv.cuFuncGetAttribute(attrib, func))
            if int(current) == int(value):
                continue
            checkCudaErrors(cuda_drv.cuFuncSetAttribute(func, attrib, int(value)))
