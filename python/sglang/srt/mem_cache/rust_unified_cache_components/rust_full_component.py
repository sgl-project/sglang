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
"""Pool-side handler for FULL attention (always-present baseline)."""

from __future__ import annotations

from sglang.srt.mem_cache._mem_cache_core import ComponentType, RadixCacheRuntimePyError
from sglang.srt.mem_cache.rust_unified_cache_components.rust_tree_component import (
    RustTreeComponent,
)


class RustFullComponent(RustTreeComponent):
    component_type = ComponentType.Full

    def free_evicted(self, freed_bin):
        alloc = self.cache.token_to_kv_pool_allocator
        if alloc is None:
            return
        for t in freed_bin:
            alloc.free(t)

    def stage_insert_action(self, action):
        # (ComponentType.Full, "FullFree", full_to_free)
        if action[1] != "FullFree":
            raise RadixCacheRuntimePyError(
                f"RustFullComponent: unknown insert action {action[1]!r}"
            )
        self.cache.token_to_kv_pool_allocator.free(action[2])

    def evictable_size(self):
        return self.cache._rust_radix.evictable_token_size()

    def protected_size(self):
        return self.cache._rust_radix.protected_token_size()
