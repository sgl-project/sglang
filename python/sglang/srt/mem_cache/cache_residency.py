"""Opt-in residency policies for UnifiedRadixCache under HiCache write_back.

They change only which cached states stay on host and in which order host
entries are evicted; no KV or state value changes. Both need the Rust TreeCore
built with ``cache_residency_features`` (see rust_tree_core/__init__.py).
"""

from __future__ import annotations

import logging
import time
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Bounds on the sub-agent tracker's memory; arbitrary, well above the session
# counts of an hour-long agentic replay (~20k sessions per worker).
_MAX_SESSIONS = 1 << 17
_MAX_NODES_PER_SESSION = 256


class CacheResidency:
    """The cache's residency switches, their counters and engagement logs."""

    def __init__(self, *, tree_core, is_write_back: bool) -> None:
        self.keep_backed_mamba_states = (
            envs.SGLANG_HICACHE_KEEP_BACKED_MAMBA_STATES.get()
        )
        self.subagent_host_evict_first = (
            envs.SGLANG_HICACHE_SUBAGENT_HOST_EVICT_FIRST.get()
        )
        self.batched_write_back_eviction = (
            envs.SGLANG_HICACHE_WRITE_BACK_BATCHED_EVICTION.get()
        )
        self._tree_core = tree_core
        self._log_interval = envs.SGLANG_HICACHE_RESIDENCY_LOG_INTERVAL.get()
        self._next_log = time.monotonic() + self._log_interval
        self.counts: Counter[str] = Counter()
        self._logged: set[str] = set()
        self.subagents: Optional[SubagentEvictFirstTracker] = None
        if not self.any_enabled:
            return
        if not is_write_back:
            raise ValueError(
                "SGLANG_HICACHE_KEEP_BACKED_MAMBA_STATES, "
                "SGLANG_HICACHE_SUBAGENT_HOST_EVICT_FIRST and "
                "SGLANG_HICACHE_WRITE_BACK_BATCHED_EVICTION need "
                "--hicache-write-policy write_back"
            )
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

        if isinstance(tree_core, RustUnifiedTreeCore):
            # Raises AttributeError on an extension built without these switches.
            tree_core.configure_cache_residency(
                keep_backed_mamba_states=self.keep_backed_mamba_states,
                host_evict_first=self.subagent_host_evict_first,
            )
        elif self.keep_backed_mamba_states or self.subagent_host_evict_first:
            raise RuntimeError(
                "SGLANG_HICACHE_KEEP_BACKED_MAMBA_STATES and "
                "SGLANG_HICACHE_SUBAGENT_HOST_EVICT_FIRST need the Rust TreeCore, "
                f"not {type(tree_core).__name__}"
            )
        if self.subagent_host_evict_first:
            self.subagents = SubagentEvictFirstTracker(
                mark_host_evict_first=tree_core.mark_host_evict_first,
                counts=self.counts,
                note_once=self.note_once,
            )
        logger.info(
            "Cache residency switches: keep_backed_mamba_states=%s "
            "subagent_host_evict_first=%s write_back_batched_eviction=%s "
            "(tree core %s)",
            self.keep_backed_mamba_states,
            self.subagent_host_evict_first,
            self.batched_write_back_eviction,
            type(tree_core).__name__,
        )

    @property
    def any_enabled(self) -> bool:
        return (
            self.keep_backed_mamba_states
            or self.subagent_host_evict_first
            or self.batched_write_back_eviction
        )

    def note_once(self, key: str, message: str, *args) -> None:
        if key not in self._logged:
            self._logged.add(key)
            logger.info(message, *args)

    def maybe_log(self) -> None:
        """Cumulative counters, at most once per log interval."""
        if not self.any_enabled or self._log_interval <= 0:
            return
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self._log_interval
        rust = self._tree_core.cache_residency_stats()
        logger.info(
            "Cache residency counters: rust %s | controller %s",
            " ".join(f"{k}={rust[k]}" for k in sorted(rust)),
            " ".join(f"{k}={self.counts[k]}" for k in sorted(self.counts)) or "-",
        )


class SubagentEvictFirstTracker:
    """Ranks a finished sub-agent session's nodes first for host eviction.

    A request whose ``parent_session_id`` names another session is a sub-agent
    child. When the parent session sends its next request, each child with no
    request in flight is finished: the deepest nodes its requests ended on are
    marked host evict-first. A marked node is still written back and demoted
    as usual; a child that returns clears the mark by matching its nodes.
    """

    def __init__(
        self,
        *,
        mark_host_evict_first: Callable[[Sequence[int]], int],
        counts: Counter,
        note_once: Callable[..., None],
    ) -> None:
        self._mark = mark_host_evict_first
        self._counts = counts
        self._note_once = note_once
        self._admitted: OrderedDict[str, str] = OrderedDict()  # rid -> session
        self._inflight: Counter[str] = Counter()
        self._children: OrderedDict[str, set[str]] = OrderedDict()
        self._nodes: OrderedDict[str, list[int]] = OrderedDict()

    def on_admit(self, req: Req) -> None:
        session = req.session_id
        if session is None or req.rid in self._admitted:
            return
        _put_bounded(self._admitted, req.rid, session)
        self._inflight[session] += 1
        parent = req.parent_session_id
        if parent is not None and parent != session:
            self._note_once(
                "parent_session_id",
                "SGLANG_HICACHE_SUBAGENT_HOST_EVICT_FIRST: first sub-agent "
                "request (session %s, parent %s)",
                session,
                parent,
            )
            self._counts["subagent_requests"] += 1
            children = self._children.get(parent)
            if children is None:
                children = set()
                _put_bounded(self._children, parent, children)
            children.add(session)
        self._demote_finished_children(session)

    def on_release(self, req: Req) -> None:
        session = self._admitted.pop(req.rid, None)
        if session is None:
            return
        self._inflight[session] -= 1
        if self._inflight[session] <= 0:
            del self._inflight[session]
        if req.parent_session_id is None or req.last_node is None:
            return
        nodes = self._nodes.get(session)
        if nodes is None:
            nodes = []
            _put_bounded(self._nodes, session, nodes)
        nodes.append(req.last_node)
        del nodes[:-_MAX_NODES_PER_SESSION]

    def _demote_finished_children(self, parent: str) -> None:
        children = self._children.get(parent)
        if not children:
            return
        finished = sorted(c for c in children if self._inflight[c] <= 0)
        node_ids: list[int] = []
        for child in finished:
            children.discard(child)
            node_ids.extend(self._nodes.pop(child, ()))
        if not node_ids:
            return
        marked = self._mark(node_ids)
        self._counts["subagent_children_demoted"] += len(finished)
        self._counts["subagent_nodes_marked"] += marked
        self._note_once(
            "subagent_mark",
            "SGLANG_HICACHE_SUBAGENT_HOST_EVICT_FIRST: first finished sub-agents "
            "ranked host evict-first (parent %s: %d children, %d nodes marked)",
            parent,
            len(finished),
            marked,
        )


def _put_bounded(table: OrderedDict, key, value) -> None:
    table[key] = value
    if len(table) > _MAX_SESSIONS:
        table.popitem(last=False)
