# Copyright 2025 SGLang Team
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
"""Session radix cache mixin.

Per-session KV held as ordinary evictable radix tags, bulk-freed on close
(``--enable-session-radix-cache``). Mixed into RadixCache alongside
KVCacheEventMixin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class SessionRadixCacheMixin:
    """Per-session KV as evictable radix tags, bulk-freed on close.

    A session is a *tag*, not an eviction class: tagged KV competes as ordinary LRU
    radix (no floor priority), there is no open/registration step, and close scans
    ``evictable_leaves`` for the tag and drains the freed leaf->branch chains a
    bounded number per scheduler iteration.

    Coupled to the host RadixCache for: ``match_prefix``, ``root_node``,
    ``evictable_leaves``, ``token_to_kv_pool_allocator``, ``_delete_leaf``, and the
    ``self._pending_release`` list (init'd in the host ``__init__`` / ``reset``).
    """

    def _tag_session_leaf(self, req: Req, radix_key, node=None) -> None:
        """Stamp this request's leaf node with its session_id so release_session
        can find it at close. No-op for non-session requests. No open/registration:
        the tag is the only session state, and release scans for it."""
        sid = getattr(req, "session_id", None)
        if sid is None:
            return
        if node is None:
            node = self.match_prefix(MatchPrefixParams(key=radix_key)).last_device_node
        if node is not None and node is not self.root_node:
            node.session_id = sid
            logger.debug("tag session %s: node=%d len=%d", sid, node.id, len(node.key))

    def release_session(self, session_id: str) -> int:
        """Queue a session's tagged leaves for deferred free (drained by
        drain_pending_release). Each seed frees the session's unique leaf->branch
        chain; shared prefixes are branch points and are preserved. Finds the
        leaves by scanning evictable_leaves for the session tag -- no per-session
        index, so a request that finishes after close has nothing to leak."""
        seeds = [
            n
            for n in self.evictable_leaves
            if getattr(n, "session_id", None) == session_id and n.lock_ref == 0
        ]
        logger.info("release_session %s: %d seeds queued", session_id, len(seeds))
        self._pending_release.extend(seeds)
        return len(seeds)

    def drain_pending_release(self, max_nodes: int = 64) -> int:
        """Free up to ``max_nodes`` queued session nodes per call so a close never
        blocks. `node not in evictable_leaves` guards against double-free of a node
        already reclaimed by normal eviction."""
        freed_nodes = 0
        while self._pending_release and freed_nodes < max_nodes:
            node = self._pending_release[-1]
            if (
                node is self.root_node
                or node.lock_ref != 0
                or len(node.children) != 0
                or node not in self.evictable_leaves
            ):
                # Chain finished, hit a branch/lock, or already evicted -> drop it.
                self._pending_release.pop()
                continue
            parent = node.parent
            self.token_to_kv_pool_allocator.free(node.value)
            self._delete_leaf(node)
            freed_nodes += 1
            # Walk up: the parent may now be a freeable leaf of this chain.
            self._pending_release[-1] = parent
        if freed_nodes:
            logger.debug("drain_pending_release: freed %d nodes", freed_nodes)
        return freed_nodes
