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
"""Pool-side handler for sliding-window attention."""

from __future__ import annotations

import torch
from sglang.srt.mem_cache._mem_cache_core import ComponentType
from sglang.srt.mem_cache.rust_unified_cache_components.rust_tree_component import (
    RustTreeComponent,
)


class RustSWAComponent(RustTreeComponent):
    component_type = ComponentType.Swa
    insert_action_tags = ("SwaRecover", "SwaStamp")

    def __init__(self, cache):
        super().__init__(cache)
        self._node_indices: list[int] = []
        self._values: list[torch.Tensor] = []

    def free_evicted(self, freed_bin):
        alloc = self.cache.token_to_kv_pool_allocator
        if alloc is None:
            return
        for t in freed_bin:
            alloc.free_swa(t)

    def stage_insert_action(self, action):
        alloc = self.cache.token_to_kv_pool_allocator
        if action[0] == "SwaRecover":
            _, node_idx, old_full_to_free, new_full_value = action
            alloc.free(old_full_to_free)
            full_value = new_full_value
        else:  # SwaStamp
            _, node_idx, full_value = action
        self._node_indices.append(node_idx)
        self._values.append(alloc.translate_loc_from_full_to_swa(full_value))

    def commit_insert_actions(self):
        if self._node_indices:
            self.cache._rust_radix.apply_swa_writes(self._node_indices, self._values)
            self._node_indices = []
            self._values = []

    def evictable_size(self):
        return self.cache._rust_radix.swa_evictable_token_size()

    def protected_size(self):
        return self.cache._rust_radix.swa_protected_token_size()
