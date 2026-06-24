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
"""Pool-side handler for Mamba / linear attention."""

from __future__ import annotations

from sglang.srt.mem_cache._mem_cache_core import ComponentType
from sglang.srt.mem_cache.rust_unified_cache_components.rust_tree_component import (
    RustTreeComponent,
)


class RustMambaComponent(RustTreeComponent):
    component_type = ComponentType.Mamba

    def free_evicted(self, freed_bin):
        mamba_allocator = self.cache.req_to_token_pool.mamba_allocator
        for t in freed_bin:
            mamba_allocator.free(t)

    def evictable_size(self):
        return self.cache._rust_radix.mamba_evictable_token_size()

    def protected_size(self):
        return self.cache._rust_radix.mamba_protected_token_size()

    def total_size(self):
        # Mamba's unit is slots, not tokens.
        return self.cache._rust_radix.mamba_total_size()
