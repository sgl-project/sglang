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
"""Orchestration-side handler for Mamba / linear attention.

Owns the Mamba SSM-slot pool ops: eviction frees, the match copy-on-write, and
the per-request caching lifecycle (extract / fork / release), which the
orchestrator drives via the generic `prepare`/`cleanup_after_caching_req` hooks.
"""

from __future__ import annotations

from sglang.srt.mem_cache._mem_cache_core import ComponentType
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.rust_unified_cache_components.rust_tree_component import (
    RustTreeComponent,
)


class RustMambaComponent(RustTreeComponent):
    component_type = ComponentType.Mamba

    def __init__(self, cache):
        super().__init__(cache)
        self.enable_extra_buffer = cache.enable_mamba_extra_buffer

    def free_evicted(self, freed_bin):
        mamba_allocator = self.cache.req_to_token_pool.mamba_allocator
        for t in freed_bin:
            mamba_allocator.free(t)

    def finalize_match_result(self, params, rust_result):
        """Copy the matched node's SSM state into a request-local slot."""
        if not params.cow_mamba or rust_result.mamba_value is None:
            return
        req = params.req
        src_index = rust_result.mamba_value
        if req.mamba_pool_idx is None:
            forked = self._fork_from(
                src_index, protect_node_idx=rust_result.last_device_node_idx
            )
            req.mamba_pool_idx = forked[0]
        else:
            self.cache.req_to_token_pool.mamba_pool.copy_from(
                src_index, req.mamba_pool_idx.unsqueeze(0)
            )

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        if is_finished:
            cache_len = (
                (req.mamba_last_track_seqlen or 0)
                if self.enable_extra_buffer
                else token_ids_len
            )
            insert_params.mamba_value, _ = self._extract_value(req)
            return cache_len

        # Unfinished: cache only up to the chunk-aligned boundary, and fork so
        # decode mutations don't alias the cached state.
        if self.enable_extra_buffer:
            cache_len = req.mamba_last_track_seqlen
            if cache_len is None:
                return None  # no chunk-aligned boundary yet — skip caching
        else:
            cache_len = token_ids_len
        if req.mamba_pool_idx is not None:
            mamba_value_src, _ = self._extract_value(req)
            assert (
                mamba_value_src is not None
            ), "mamba_value_src must be present when req.mamba_pool_idx is not None"
            insert_params.mamba_value = self._fork_from(mamba_value_src)
        return cache_len

    def cleanup_after_caching_req(
        self, req, is_finished, inserted=False, insert_result=None,
        insert_params=None, disabled=False,
    ):
        if disabled:
            self.cache.req_to_token_pool.free_mamba_cache(req)
            return
        if is_finished:
            if inserted:
                mamba_exist = insert_result.mamba_exist
            else:
                mamba_exist = (
                    insert_params is not None and insert_params.mamba_value is not None
                )
            # extra_buffer always frees the orphaned primary (keeping the
            # surviving ping-pong slot); no_buffer frees only when rejected.
            if self.enable_extra_buffer and req.mamba_pool_idx is not None:
                keep = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            else:
                keep = None
            if mamba_exist:
                keep = None
            if self.enable_extra_buffer or mamba_exist:
                self.cache.req_to_token_pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep
                )
        else:
            forked = insert_params.mamba_value if insert_params is not None else None
            if (
                forked is not None
                and insert_result is not None
                and insert_result.mamba_exist
            ):
                self.cache.req_to_token_pool.mamba_allocator.free(forked)
            # Clear the chunk-aligned marker so the next call recomputes it.
            req.mamba_last_track_seqlen = None

    def _extract_value(self, req):
        """Build the `mamba_value` tensor to insert into the radix tree."""
        if req.mamba_pool_idx is None:
            return None, None
        if not self.enable_extra_buffer:
            return req.mamba_pool_idx.unsqueeze(-1).clone(), None
        # extra_buffer mode: also return the ping-pong index to release later.
        track_buffer_to_keep = self.cache.req_to_token_pool.get_mamba_ping_pong_other_idx(
            req.mamba_next_track_idx
        )
        mamba_value = (
            req.mamba_ping_pong_track_buffer[track_buffer_to_keep].unsqueeze(-1).clone()
        )
        return mamba_value, track_buffer_to_keep

    def _fork_from(self, mamba_value, protect_node_idx=None):
        """Fork `mamba_value` in the pool, evicting only if alloc fails."""
        mamba_allocator = self.cache.req_to_token_pool.mamba_allocator
        mamba_pool = self.cache.req_to_token_pool.mamba_pool
        dst = mamba_allocator.alloc(1)
        if dst is None:
            if protect_node_idx is not None:
                self.cache.inc_lock_ref(protect_node_idx)
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            dst = mamba_allocator.alloc(1)
            if protect_node_idx is not None:
                self.cache.dec_lock_ref(protect_node_idx)
            assert dst is not None, "Can not alloc mamba cache"
        mamba_pool.copy_from(mamba_value, dst)
        return dst
