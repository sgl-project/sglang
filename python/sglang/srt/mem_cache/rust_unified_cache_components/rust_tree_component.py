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
"""Base orchestration-side per-component handler for the Rust radix cache.

Parallels `unified_cache_components/tree_component.py`, but owns only the POOL
ops: the Rust crate's `Component`/`Slot` own the tree-side logic (walks, LRU,
lock-ref), so these handlers run just the allocator ops Python must perform. The
methods here therefore do NOT mirror `TreeComponent` one-for-one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from sglang.srt.mem_cache._mem_cache_core import ComponentType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
    from sglang.srt.mem_cache.rust_unified_radix_cache import RustUnifiedRadixCache


class RustTreeComponent:
    """Pool-side handler for one `ComponentType`. Defaults are no-ops."""

    component_type: ComponentType

    def __init__(self, cache: "RustUnifiedRadixCache"):
        self.cache = cache

    # --- Required: every component frees into its own pool (raise if not). ---
    def free_evicted(self, freed_bin: list[torch.Tensor]) -> None:
        """Free this component's evicted tensors into its pool. `evict()`
        guarantees the pool allocator is live before calling."""
        raise NotImplementedError

    # --- Generic size accounting, keyed by this component's type. ---
    def evictable_size(self) -> int:
        return self.cache._rust_radix.component_evictable_size(self.component_type)

    def protected_size(self) -> int:
        return self.cache._rust_radix.component_protected_size(self.component_type)

    def total_size(self) -> int:
        return self.cache._rust_radix.component_total_size(self.component_type)

    # --- Optional hooks: the orchestrator loops these over every component, so
    #     only the participating component overrides; the rest stay no-ops. ---
    def stage_insert_action(self, action: tuple) -> None:
        """Apply one routed insert DeferredAction (may buffer for `commit`)."""

    def commit_insert_actions(self) -> None:
        """Flush anything `stage_insert_action` buffered. Called once per insert,
        after every action is staged (so batched writes go out in one call).
        Default no-op."""

    def finalize_match_result(self, params: Any, rust_result: Any) -> None:
        """Pool-side match post-processing (e.g. Mamba CoW). Default no-op."""

    def prepare_for_caching_req(
        self,
        req: "Req",
        insert_params: "InsertParams",
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        """Build this component's insert value (set on `insert_params`); return
        the cache length it wants (min-reduced across components), or `None` to
        skip caching this request. Default: cache the full length."""
        return token_ids_len

    def cleanup_after_caching_req(
        self,
        req: "Req",
        is_finished: bool,
        inserted: bool = False,
        insert_result: "Optional[InsertResult]" = None,
        insert_params: "Optional[InsertParams]" = None,
        disabled: bool = False,
    ) -> None:
        """Release this component's per-request pool slots. Default no-op."""
