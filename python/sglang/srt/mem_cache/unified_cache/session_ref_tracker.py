"""Session ref tracking for UnifiedRadixCache (``--enable-session-radix-cache``):
tag each request's KV by session_id for each tree component; ``release_radix_session``
(close) releases a session's tagged reference.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.unified_tree_core import NodeId

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.components.base import (
        TreeComponent,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

logger = logging.getLogger(__name__)

# Bounded guard against a request finishing after close. If a session id falls
# out of this LRU after 8192 later closes, an extremely late finish can tag
# again; explicit register_session clears the tombstone for intentional id reuse.
_CLOSED_SESSION_TOMBSTONE_LIMIT = 8192


@dataclass(kw_only=True)
class UnifiedSessionRefTracker:
    """Tags radix KV by session id; ``release_radix_session`` (close) releases a
    session's tagged reference. Each component maintains its own session_ids,
    session_ref and so on."""

    components: tuple[TreeComponent, ...]
    tree_core: UnifiedTreeCore
    enable_session_radix_cache: bool

    def __post_init__(self) -> None:
        self._ref_prompt_tail = envs.SGLANG_OPT_SESSION_REF_PROMPT_TAIL.get()
        self.reset()

    def reset(self) -> None:
        self._closed_session_ids: OrderedDict[str, None] = OrderedDict()
        self._session_incarnation_counter: int = 0
        self._session_generations: dict[str, int] = {}
        for component in self.components:
            component.reset_session_state()

    def session_id_for_req(self, req: Req) -> Optional[str]:
        session_id = req.session_id
        if session_id is None and req.session is not None:
            session_id = req.session.session_id
        return session_id

    def register_session_ref(self, req: Req, leaf: NodeId) -> None:
        """Register the leaf a finished request's insert ended on with each
        component; the request's lock sits on a different node."""
        if not self.enable_session_radix_cache:
            return

        session = req.session
        if session is not None and session.streaming:
            return

        session_id = self.session_id_for_req(req)
        if session_id is None or session_id in self._closed_session_ids:
            return

        current_generation = self._session_generations.get(session_id)
        if current_generation is None or req.session_generation != current_generation:
            logger.warning("register_session_ref called for stale request; Skip it.")
            return

        node = self.tree_core.node_by_id(leaf)
        # A server-managed session (session_params) extends the previous output.
        if self._ref_prompt_tail and session is None:
            self._register_prompt_tail_ref(req, session_id, node)
            return
        if node is self.tree_core.root_node:
            return

        for component in self.components:
            component_leaf = component.resolve_session_leaf(req, node)
            component.register_session_leaf(session_id, component_leaf)

    def _register_prompt_tail_ref(self, req: Req, session_id: str, node) -> None:
        """Move the session's single reference to the end of this request's
        prompt, releasing its older prompt ends and output branches."""
        node = self._prompt_tail_node(req, node)
        for component in self.components:
            component_leaf = (
                None
                if node is self.tree_core.root_node
                else component.resolve_session_leaf(req, node)
            )
            component.register_session_leaf(session_id, component_leaf)
            component.release_other_session_leaves(session_id, keep=component_leaf)

    def _prompt_tail_node(self, req: Req, node):
        """The deepest node on the root path of ``node`` that ends within the
        request's prompt."""
        root = self.tree_core.root_node
        # A bigram (EAGLE) key is one shorter than the tokens it spans.
        prompt_len = len(req.origin_input_ids) - int(self.tree_core.is_eagle)
        end, cur = 0, node
        while cur is not root:
            end += len(cur.key)
            cur = cur.parent
        cur = node
        while cur is not root and end > prompt_len:
            end -= len(cur.key)
            cur = cur.parent
        return cur

    def _remember_closed_session(self, session_id: str) -> None:
        self._closed_session_ids[session_id] = None
        self._closed_session_ids.move_to_end(session_id)
        while len(self._closed_session_ids) > _CLOSED_SESSION_TOMBSTONE_LIMIT:
            self._closed_session_ids.popitem(last=False)

    def open_radix_session(self, session_id: str) -> Optional[int]:
        self._closed_session_ids.pop(session_id, None)
        self._session_incarnation_counter += 1
        self._session_generations[session_id] = self._session_incarnation_counter
        return self._session_incarnation_counter

    def ensure_session_generation(self, session_id: str) -> int:
        generation = self._session_generations.get(session_id)
        if generation is None:
            generation = self.open_radix_session(session_id)
        return generation

    def release_radix_session(self, session_id: str) -> int:
        if not self.enable_session_radix_cache or session_id is None:
            return 0

        if session_id in self._closed_session_ids:
            return 0

        self._remember_closed_session(session_id)
        self._session_generations.pop(session_id, None)

        indexed = 0
        for component in self.components:
            indexed += component.release_session(session_id)

        logger.info(
            "release_session %s: indexed %d component leaves",
            session_id,
            indexed,
        )
        return 0
