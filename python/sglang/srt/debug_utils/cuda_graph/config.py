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
"""Configuration for CUDA-graph-compatible dumping.

The knobs live on `DumperConfig` so they inherit its env-var machinery
(`DUMPER_CUDA_GRAPH_ENABLE`, `..._FILTER`, `..._BUDGET_MB`, `..._STRICT`,
`..._TAPS`) and its `/dumper/configure` HTTP surface.  This module only projects
them into a small value object, so the rest of the package needs neither
`DumperConfig` nor a live dumper to be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_MB = 1024 * 1024

TAP_HOOK = "t1"
TAP_COMPILED = "t3"
_ALL_TAPS = (TAP_HOOK, TAP_COMPILED)
_EVERYTHING = ("all", "*")
_NOTHING = ("none", "off")


@dataclass(frozen=True)
class CudaGraphDumpConfig:
    """Projection of the `cuda_graph_*` fields of `DumperConfig`."""

    enable: bool = False
    # Comma-separated *prefixes* of fully-expanded dump names.  Empty means
    # "every name", which is only sane for small models -- one graph-resident
    # buffer is allocated per (name, occurrence), so a full leaf sweep over a
    # large model will hit `budget_mb` and start dropping names.
    filter: Optional[str] = None
    budget_mb: int = 4096
    # Turn the two soft failure modes (budget exhausted, zero taps observed)
    # into exceptions instead of warnings.  Recommended in CI.
    strict: bool = False
    # Which channels to arm; None/empty/`all` arms every one.  See `taps`.
    taps: Optional[str] = None

    @classmethod
    def from_dumper_config(cls, dumper_config: Any) -> CudaGraphDumpConfig:
        return cls(
            enable=bool(getattr(dumper_config, "cuda_graph_enable", False)),
            filter=getattr(dumper_config, "cuda_graph_filter", None),
            budget_mb=int(getattr(dumper_config, "cuda_graph_budget_mb", 4096)),
            strict=bool(getattr(dumper_config, "cuda_graph_strict", False)),
            taps=getattr(dumper_config, "cuda_graph_taps", None),
        )

    @property
    def prefixes(self) -> tuple[str, ...]:
        if not self.filter:
            return ()
        return tuple(p for p in (s.strip() for s in self.filter.split(",")) if p)

    @property
    def budget_bytes(self) -> int:
        """Negative means unbounded."""
        return -1 if self.budget_mb < 0 else self.budget_mb * _MB

    @property
    def armed_taps(self) -> frozenset[str]:
        """The channels the user asked for; every one when unset.

        Unset means "arm everything", not "arm the cheap ones": a debugging tool
        that quietly under-covers by default is the exact failure this package
        exists to close.  Narrowing is explicit, and `report()` names what was
        armed so a narrowed run cannot be mistaken for a complete one.
        """
        raw = (self.taps or "").strip().lower()
        if not raw or raw in _EVERYTHING:
            return frozenset(_ALL_TAPS)
        if raw in _NOTHING:
            return frozenset()
        return frozenset(
            t for t in (s.strip() for s in raw.split(",")) if t in _ALL_TAPS
        )

    def arms(self, tap: str) -> bool:
        return tap in self.armed_taps

    def accepts(self, name: str) -> bool:
        prefixes = self.prefixes
        if not prefixes:
            return True
        return any(name.startswith(prefix) for prefix in prefixes)
