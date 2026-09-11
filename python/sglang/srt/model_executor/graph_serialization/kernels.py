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

A saved kernel node must find its ``CUfunction`` again in a fresh process. The
identity is ``(container sha256, kernel name)``: names repeat across
translation units (fact 11), and the driver never checks identity on
``cuGraphExecUpdate`` (fact 8), so the loader compares names itself. Kernels
with no bytes on disk (cuBLASLt ``nvjet_*``, CuTe-DSL; fact 12) carry a
live-only identity (module name-set digest plus attributes) and are found by
harvesting live handles at load.

Resolution tiers, tried in this order at load: module bytes captured in the
saving process (CUPTI, fact 20), fatbin sections of mapped shared objects
(fact 11), JIT caches on disk (Triton, DeepGEMM, flashinfer), and a live
harvest capture (fact 5). A kernel none of them resolves makes its shape
recapture.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

import msgspec

TIERS = ("captured_bytes", "elf_section", "jit_cache", "live_harvest")


class KernelIdentity(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    image_sha256: Optional[str] = None  # None: live-only identity
    module_digest: str = ""  # sha256 of the owning module's sorted kernel names
    attrs: tuple[tuple[int, int], ...] = ()  # (CU_FUNC_ATTRIBUTE_*, value); re-applied


class KernelSource(Protocol):
    """One resolution tier (``name`` in :data:`TIERS`)."""

    name: str

    def identify(self, func: int) -> Optional[KernelIdentity]:
        """SAVE: durable identity of a live ``CUfunction``, or ``None``."""

    def resolve(self, ident: KernelIdentity) -> Optional[int]:
        """LOAD: the ``CUfunction`` for ``ident``, or ``None``."""


class KernelUnresolved(RuntimeError):
    pass


class KernelResolver:
    """Chain of :class:`KernelSource` tiers."""

    def __init__(self, sources: Sequence[KernelSource]) -> None:
        self.sources = sorted(
            sources,
            key=lambda s: TIERS.index(s.name) if s.name in TIERS else len(TIERS),
        )

    def identify(self, func: int) -> KernelIdentity:
        for s in self.sources:
            ident = s.identify(func)
            if ident is not None:
                return ident
        raise KernelUnresolved(f"no tier identifies CUfunction 0x{func:x}")

    def resolve(self, ident: KernelIdentity) -> int:
        for s in self.sources:
            func = s.resolve(ident)
            if func is not None:
                apply_attrs(func, ident)
                return func
        raise KernelUnresolved(
            f"{ident.name}: no tier resolves it ({', '.join(s.name for s in self.sources)})"
        )


def apply_attrs(func: int, ident: KernelIdentity) -> None:
    """Re-apply the saved function attributes (``MAX_DYNAMIC_SHARED_SIZE_BYTES``,
    carveout, ...) to a re-resolved ``CUfunction``; they live on the function,
    not on the node (fact 8)."""
    raise NotImplementedError(
        "apply_attrs: cuFuncSetAttribute is not implemented in this draft; see "
        "DESIGN_cuda_graph_serialization.md section 6.5"
    )
