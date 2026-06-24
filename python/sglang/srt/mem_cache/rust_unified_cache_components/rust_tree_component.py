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
"""Base pool-side per-component handler for the Rust radix cache.

Parallels `unified_cache_components/tree_component.py`, but is the POOL-HALF only:
the Rust crate's `Component`/`Slot` own the tree-side logic (walks, LRU, lock-ref),
so these handlers own just the allocator ops Python must perform. The methods here
therefore do NOT mirror `TreeComponent` one-for-one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.mem_cache._mem_cache_core import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.rust_unified_radix_cache import RustUnifiedRadixCache


class RustTreeComponent:
    """Pool-side handler for one `ComponentType`. Defaults are no-ops."""

    component_type: ComponentType
    # DeferredAction tags this component handles in `stage_insert_action`.
    insert_action_tags: tuple[str, ...] = ()

    def __init__(self, cache: "RustUnifiedRadixCache"):
        self.cache = cache

    def free_evicted(self, freed_bin: list[torch.Tensor]) -> None:
        """Free this component's evicted tensors into its pool."""
        raise NotImplementedError

    def stage_insert_action(self, action: tuple) -> None:
        """Apply one insert DeferredAction (may buffer for `commit`)."""

    def commit_insert_actions(self) -> None:
        """Flush anything `stage_insert_action` buffered. Default no-op."""

    def evictable_size(self) -> int:
        return 0

    def protected_size(self) -> int:
        return 0
