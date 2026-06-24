# Copyright 2023-2024 SGLang Team
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
"""Python orchestration-layer tree components that own the custom cache-management
logic for each component type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.mem_cache._mem_cache_core import ComponentType
from sglang.srt.mem_cache.rust_unified_cache_components.rust_full_component import (
    RustFullComponent,
)
from sglang.srt.mem_cache.rust_unified_cache_components.rust_mamba_component import (
    RustMambaComponent,
)
from sglang.srt.mem_cache.rust_unified_cache_components.rust_swa_component import (
    RustSWAComponent,
)
from sglang.srt.mem_cache.rust_unified_cache_components.rust_tree_component import (
    RustTreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.rust_unified_radix_cache import RustUnifiedRadixCache

__all__ = [
    "RustTreeComponent",
    "RustFullComponent",
    "RustSWAComponent",
    "RustMambaComponent",
    "build_components",
]


def build_components(
    cache: "RustUnifiedRadixCache",
) -> "dict[ComponentType, RustTreeComponent]":
    """One handler per configured component, keyed by `ComponentType`."""
    components: dict[ComponentType, RustTreeComponent] = {
        ComponentType.Full: RustFullComponent(cache)
    }
    if cache.sliding_window_size is not None:
        components[ComponentType.Swa] = RustSWAComponent(cache)
    if cache.mamba_cache_chunk_size is not None:
        components[ComponentType.Mamba] = RustMambaComponent(cache)
    return components
