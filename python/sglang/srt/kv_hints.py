# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
        SessionCacheEvictResult,
        UnifiedSessionRefTracker,
    )
    from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector

logger = logging.getLogger(__name__)

KV_HINT_DEREF_V1 = "kv_hint.deref.v1"
_DEREF_NEXT_REQUEST_LIMIT = 8192
_DEREF_ACTION_LIMIT = 8192


class KvHintStruct(msgspec.Struct, kw_only=True):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class DerefHint(KvHintStruct):
    action_id: str = ""


class KvHints(KvHintStruct):
    deref: Optional[DerefHint] = None


class KvHintManager:
    """Owns supported KV hint handlers and their request lifecycle hooks."""

    def __init__(
        self,
        session_refs: Optional[UnifiedSessionRefTracker] = None,
        metrics_collector: Optional[RadixCacheMetricsCollector] = None,
    ) -> None:
        self._session_refs = session_refs
        self._metrics_collector = metrics_collector
        self._deref_next_sessions: OrderedDict[str, int] = OrderedDict()
        self._deref_next_requests: OrderedDict[str, str] = OrderedDict()
        self._applied_deref_actions: OrderedDict[tuple[str, str], None] = OrderedDict()

    def capabilities(self) -> list[str]:
        if self._session_refs is None:
            return []
        return [KV_HINT_DEREF_V1]

    def on_request(self, req: Req, hints: KvHints) -> None:
        """Accept supported hints after the request's session is resolved."""
        if hints.deref is None:
            return
        if self._session_refs is None:
            logger.warning("Ignoring KV DEREF because session radix cache is disabled")
            return
        if req.session_id is None or req.session_generation is None:
            logger.warning("Ignoring KV DEREF without a radix-native session")
            return

        req.kv_hints = hints
        logger.info(
            "Accepted KV DEREF session_id=%s generation=%s",
            req.session_id,
            req.session_generation,
        )

    def on_request_success(self, req: Req, *, has_reusable_leaf: bool) -> None:
        """Apply DEREF on request success and track ordinary request leaves."""
        if self._session_refs is None:
            return

        deref = req.kv_hints.deref if req.kv_hints is not None else None
        if deref is not None:
            action_key = self._deref_action_key(req, deref)
            if action_key is not None and action_key in self._applied_deref_actions:
                self._applied_deref_actions.move_to_end(action_key)
                self._record_duplicate_deref()
                logger.info(
                    "Skipped duplicate KV DEREF session_id=%s generation=%s action_id=%s",
                    req.session_id,
                    req.session_generation,
                    deref.action_id,
                )
                return

            start_time = time.perf_counter()
            result = self._session_refs.evict_radix_session(
                req.session_id, req.session_generation
            )
            self._record_deref_result(start_time, result)
            if result.status == "evicted" and result.generation is not None:
                if action_key is not None:
                    self._remember_deref_action(action_key)
                self._remember_deref_session(req.session_id, result.generation)
            self._log_deref_result(req, result)
            return

        if not has_reusable_leaf:
            return

        self._session_refs.register_session_ref(req)

    def on_request_match(self, req: Optional[Req]) -> None:
        """Track the first matching request after a successful DEREF."""
        if req is None or req.session_id is None or req.session_generation is None:
            return

        generation = self._deref_next_sessions.get(req.session_id)
        if generation != req.session_generation:
            return

        self._deref_next_sessions.move_to_end(req.session_id)
        self._remember_deref_request(req.rid, req.session_id)

    def on_request_prefill_ready(self, req: Req) -> None:
        """Record the next DEREF session request after its L3 prefetch resolves."""
        session_id = self._deref_next_requests.pop(req.rid, None)
        if session_id is None:
            return

        self._deref_next_sessions.pop(session_id, None)
        if self._metrics_collector is None:
            return

        self._metrics_collector.record_kv_hint_deref_next_request(
            input_tokens=len(req.full_untruncated_fill_ids),
            device_tokens=len(req.prefix_indices),
            host_tokens=req.host_hit_length,
            storage_tokens=req.storage_hit_length,
        )

    def _remember_deref_session(self, session_id: str, generation: int) -> None:
        self._deref_next_sessions[session_id] = generation
        self._deref_next_sessions.move_to_end(session_id)
        while len(self._deref_next_sessions) > _DEREF_NEXT_REQUEST_LIMIT:
            self._deref_next_sessions.popitem(last=False)

    def _remember_deref_request(self, request_id: str, session_id: str) -> None:
        self._deref_next_requests[request_id] = session_id
        self._deref_next_requests.move_to_end(request_id)
        while len(self._deref_next_requests) > _DEREF_NEXT_REQUEST_LIMIT:
            self._deref_next_requests.popitem(last=False)

    @staticmethod
    def _deref_action_key(req: Req, deref: DerefHint) -> Optional[tuple[str, str]]:
        if not deref.action_id:
            return None
        return (req.session_id, deref.action_id)

    def _remember_deref_action(self, action_key: tuple[str, str]) -> None:
        self._applied_deref_actions[action_key] = None
        self._applied_deref_actions.move_to_end(action_key)
        while len(self._applied_deref_actions) > _DEREF_ACTION_LIMIT:
            self._applied_deref_actions.popitem(last=False)

    def _record_deref_result(
        self, start_time: float, result: SessionCacheEvictResult
    ) -> None:
        if self._metrics_collector is None:
            return
        self._metrics_collector.record_kv_hint_deref(
            status=result.status,
            duration_seconds=time.perf_counter() - start_time,
            indexed_component_leaves=result.indexed_component_leaves,
        )

    def _record_duplicate_deref(self) -> None:
        if self._metrics_collector is None:
            return
        self._metrics_collector.record_kv_hint_deref(
            status="duplicate",
            duration_seconds=0.0,
            indexed_component_leaves=0,
        )

    @staticmethod
    def _log_deref_result(req: Req, result: SessionCacheEvictResult) -> None:
        logger.info(
            "Applied KV DEREF session_id=%s status=%s generation=%s "
            "indexed_component_leaves=%s",
            req.session_id,
            result.status,
            result.generation,
            result.indexed_component_leaves,
        )
