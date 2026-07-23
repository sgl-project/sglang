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

"""Radix-cache backend with fuzzy (non-exact) prefix reuse.

``FuzzyRadixCache`` extends the standard ``RadixCache`` with a pluggable
:class:`FuzzyMatchProvider`. When exact prefix matching leaves part of the
prompt uncovered, the provider may nominate donor KV from a previously
finished request. The donor slots are surfaced through
``MatchResult.fuzzy_matched_len`` and position-corrected (RoPE) by the model
runner before the forward pass.

Select it with ``--radix-cache-backend fuzzy_match``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    create_fuzzy_match_provider,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.registry import TreeCacheBuildContext

logger = logging.getLogger(__name__)


class FuzzyRadixCache(RadixCache):
    """RadixCache that falls back to provider-driven fuzzy matching.

    The exact radix tree stays authoritative: pool indices are owned by tree
    nodes, and donors are addressed by ``TreeNode.id`` through a node
    registry maintained alongside tree mutations. A fuzzy hit pins the donor
    node (``inc_lock_ref``) for the lifetime of the recipient request so LRU
    eviction cannot free donor KV mid-forward.
    """

    def __init__(self, params):
        # Set before super().__init__ because it invokes self.reset().
        self._node_registry: Dict[int, TreeNode] = {}
        self.fuzzy_config: Optional[FuzzyMatchConfig] = None
        self.fuzzy_match_provider: Optional[FuzzyMatchProvider] = None
        self._fuzzy_cache_enabled: bool = False
        super().__init__(params)

    def init_fuzzy_match(
        self, config: FuzzyMatchConfig, provider: FuzzyMatchProvider
    ) -> None:
        """Attach the fuzzy-match provider and its configuration."""
        self.fuzzy_config = config
        self.fuzzy_match_provider = provider
        self._fuzzy_cache_enabled = config.cache_fuzzy_results
        logger.info(
            "Fuzzy matching initialized: provider=%s, min_match_length=%d, "
            "cache_results=%s",
            config.fuzzy_match_provider,
            config.fuzzy_min_match_length,
            config.cache_fuzzy_results,
        )

    ##### Public API #####

    def reset(self):
        if self.fuzzy_match_provider is not None:
            try:
                self.fuzzy_match_provider.on_cache_reset()
            except Exception as e:
                logger.warning("[FUZZY RADIX] on_cache_reset failed: %s", e)
        super().reset()
        self._node_registry.clear()
        self._register_node(self.root_node)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = super().match_prefix(params)
        # Fuzzy matching needs a request to attach realization state to;
        # internal re-matches (e.g. cache_unfinished_req) pass req=None and
        # must stay exact-only.
        if self.disable or self.fuzzy_match_provider is None or params.req is None:
            return result

        key = params.key.page_aligned(self.page_size)
        total_len = len(key)
        exact_matched_len = len(result.device_indices)
        if total_len == 0 or exact_matched_len >= total_len:
            return result

        fuzzy_result = self._match_prefix_fuzzy(
            key=key, req=params.req, exact_matched_len=exact_matched_len
        )
        if fuzzy_result is None:
            return result

        return self._apply_fuzzy_result(
            result=result,
            fuzzy_result=fuzzy_result,
            req=params.req,
            exact_matched_len=exact_matched_len,
            total_len=total_len,
        )

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ):
        self._reclaim_realization_slots(req)
        super().cache_finished_req(
            req, is_insert=is_insert, kv_len_to_handle=kv_len_to_handle
        )
        self._release_donor(req)

    ##### Internal Helper Functions #####

    def _match_prefix_fuzzy(
        self, key: RadixKey, req: Req, exact_matched_len: int
    ) -> Optional[FuzzyMatchResult]:
        """Ask the provider for a donor covering the missed prompt suffix."""
        # Short suffixes cannot amortize the semantic lookup; skip before
        # paying the provider's embedding cost.
        suffix_len = len(key) - exact_matched_len
        if suffix_len < self.fuzzy_config.fuzzy_min_suffix_tokens:
            return None

        min_match_length = self.fuzzy_config.fuzzy_min_match_length
        if 0 < exact_matched_len < min_match_length:
            return None

        try:
            fuzzy_result = self.fuzzy_match_provider.match_on_prefix_miss(
                prompt_token_ids=key.token_ids,
                already_matched_len=exact_matched_len,
                request=req,
                extra_key=key.extra_key,
            )
        except Exception:
            logger.exception("[FUZZY RADIX] provider match failed")
            return None

        if fuzzy_result is None:
            logger.debug(
                "[FUZZY RADIX] no fuzzy match: rid=%s exact=%d total=%d",
                req.rid,
                exact_matched_len,
                len(key),
            )
            return None

        quality = fuzzy_result.quality_signals
        quality_msg = (
            ", reuse=%.3f, cosine=%.3f, tier=%s, gate=%s"
            % (
                quality.reuse_ratio,
                quality.cosine_similarity,
                quality.confidence_tier,
                quality.passed_quality_gate,
            )
            if quality is not None
            else ""
        )
        logger.info(
            "[FUZZY RADIX] fuzzy match success: rid=%s cached=%d prompt=%d "
            "offset=%d%s",
            req.rid,
            fuzzy_result.cached_token_count,
            fuzzy_result.prompt_token_count,
            fuzzy_result.position_offset,
            quality_msg,
        )
        return fuzzy_result

    def _apply_fuzzy_result(
        self,
        result: MatchResult,
        fuzzy_result: FuzzyMatchResult,
        req: Req,
        exact_matched_len: int,
        total_len: int,
    ) -> MatchResult:
        """Validate a provider result and merge it into the match result.

        Any validation failure falls back to the exact-only ``result``
        without mutating request state.
        """
        fuzzy_matched_len = fuzzy_result.cached_token_count
        if fuzzy_matched_len <= 0:
            return result

        # A donor whose tree node vanished (cache reset / eviction race)
        # cannot be safely reused.
        donor_node: Optional[TreeNode] = None
        if fuzzy_result.donor_last_node_id is not None:
            donor_node = self._node_registry.get(fuzzy_result.donor_last_node_id)
            if donor_node is None:
                logger.warning(
                    "[FUZZY RADIX] dropping fuzzy match for rid=%s: donor "
                    "node %d is gone (cache reset or evicted)",
                    req.rid,
                    fuzzy_result.donor_last_node_id,
                )
                return result

        # device_indices is prefix-shaped; reject results that cannot be
        # represented by that contract.
        fuzzy_kv_indices = fuzzy_result.kv_cache_indices.detach().to(
            device=result.device_indices.device,
            dtype=result.device_indices.dtype,
        )
        if len(fuzzy_kv_indices) != fuzzy_matched_len:
            logger.warning(
                "[FUZZY RADIX] provider returned %d kv indices but "
                "cached_token_count=%d; falling back to exact-only",
                len(fuzzy_kv_indices),
                fuzzy_matched_len,
            )
            return result

        # Position-aligned reuse (donor span already at the target positions)
        # would adopt donor-owned slots into the recipient's tree branch —
        # two nodes owning the same pool slots. Token- and position-aligned
        # content is the exact radix tree's job; skip such matches.
        if (
            fuzzy_result.segments is None
            and fuzzy_result.cached_start_pos == exact_matched_len
        ):
            logger.debug(
                "[FUZZY RADIX] dropping position-aligned fuzzy match for "
                "rid=%s (exact radix matching owns aligned content)",
                req.rid,
            )
            return result

        # Donor KV carries RoPE for its original positions; the model runner
        # must realize it into freshly allocated slots. Allocate before
        # mutating request state so a capacity failure falls back cleanly.
        realized_locs = self.token_to_kv_pool_allocator.alloc(fuzzy_matched_len)
        if realized_locs is None:
            logger.debug(
                "[FUZZY RADIX] no pool capacity for %d fuzzy tokens; "
                "falling back to exact-only",
                fuzzy_matched_len,
            )
            return result
        # Chunked-prefill resume: release slots from a previous round.
        self._reclaim_realization_slots(req)
        req.fuzzy_realized_locs = realized_locs

        if donor_node is not None and donor_node is not req.fuzzy_donor_node:
            # Release any prior donor pin before acquiring a new one
            # (chunked-prefill / re-scheduling case). A re-fire on the same
            # donor keeps the existing single pin.
            if req.fuzzy_donor_node is not None:
                self.dec_lock_ref(req.fuzzy_donor_node)
            self.inc_lock_ref(donor_node)
            req.fuzzy_donor_node = donor_node

        req.fuzzy_match_result = fuzzy_result

        logger.info(
            "[FUZZY RADIX] match_prefix: rid=%s exact=%d fuzzy=%d miss=%d "
            "total=%d cached_start_pos=%d realized_locs=pre-allocated",
            req.rid,
            exact_matched_len,
            fuzzy_matched_len,
            total_len - exact_matched_len - fuzzy_matched_len,
            total_len,
            fuzzy_result.cached_start_pos,
        )

        merged = torch.cat([result.device_indices, fuzzy_kv_indices])
        return result._replace(
            device_indices=merged,
            fuzzy_matched_len=fuzzy_matched_len,
            cache_protected_len=exact_matched_len + fuzzy_matched_len,
        )

    def _on_finished_insert(
        self,
        req: Req,
        insert_result: InsertResult,
        kv_indices: torch.Tensor,
        token_ids: List[int],
    ) -> None:
        """Register the finished request as a donor while its KV is live."""
        if not self._fuzzy_cache_enabled or self.fuzzy_match_provider is None:
            return
        try:
            self.fuzzy_match_provider.cache_on_request_finished(
                request=req,
                token_ids=token_ids,
                kv_cache=kv_indices,
                cache_start_pos=0,
                cache_end_pos=len(token_ids),
                radix_tree=self,
            )
            if insert_result.last_device_node is not None:
                self.fuzzy_match_provider.on_donor_inserted(
                    request=req,
                    donor_last_node_id=insert_result.last_device_node.id,
                )
        except Exception:
            logger.exception("[FUZZY RADIX] donor registration failed: rid=%s", req.rid)

    def _reclaim_realization_slots(self, req: Req) -> None:
        """Free pre-allocated realization slots the forward pass never used."""
        if req.fuzzy_realized_locs is not None:
            self.token_to_kv_pool_allocator.free(req.fuzzy_realized_locs)
            req.fuzzy_realized_locs = None

    def _release_donor(self, req: Req) -> None:
        """Drop the donor pin acquired in match_prefix."""
        if req.fuzzy_donor_node is not None:
            self.dec_lock_ref(req.fuzzy_donor_node)
            req.fuzzy_donor_node = None

    # --- node registry maintenance -------------------------------------
    # Donors are addressed by TreeNode.id; keep an id -> node map in sync
    # with every tree mutation so stale donor references are detectable.

    def _register_node(self, node: TreeNode) -> None:
        self._node_registry[node.id] = node

    def _insert_helper(self, node, key, value, priority=0, chunked=False):
        prefix_len, last_node = super()._insert_helper(
            node, key, value, priority, chunked
        )
        self._register_node(last_node)
        return prefix_len, last_node

    def _split_node(self, key, child, split_len):
        new_node = super()._split_node(key, child, split_len)
        self._register_node(new_node)
        self._register_node(child)
        return new_node

    def _delete_leaf(self, node):
        super()._delete_leaf(node)
        self._node_registry.pop(node.id, None)


def fuzzy_match_backend_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    """Build a ``FuzzyRadixCache`` for ``--radix-cache-backend fuzzy_match``."""
    if ctx.disable_radix_cache:
        raise ValueError(
            "--radix-cache-backend fuzzy_match requires the radix cache; "
            "remove --disable-radix-cache"
        )
    if ctx.params.is_eagle:
        raise ValueError(
            "--radix-cache-backend fuzzy_match does not support EAGLE "
            "speculative decoding yet"
        )
    config = FuzzyMatchConfig.from_server_args(ctx.server_args)
    provider = create_fuzzy_match_provider(config)
    cache = FuzzyRadixCache(params=ctx.params)
    if provider is not None:
        cache.init_fuzzy_match(config=config, provider=provider)
    return cache
