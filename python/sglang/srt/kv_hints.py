# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.utils.msgspec_utils import msgspec_struct_pydantic_core_schema

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
        SessionCacheEvictResult,
        UnifiedSessionRefTracker,
    )

logger = logging.getLogger(__name__)

KV_HINT_DEREF_V1 = "kv_hint.deref.v1"


class KvHintStruct(msgspec.Struct, kw_only=True):
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return msgspec_struct_pydantic_core_schema(cls, handler)


class DerefHint(KvHintStruct):
    pass


class KvHints(KvHintStruct):
    deref: Optional[DerefHint] = None


class KvHintManager:
    """Owns supported KV hint handlers and their request lifecycle hooks."""

    def __init__(self, session_refs: Optional[UnifiedSessionRefTracker] = None) -> None:
        self._session_refs = session_refs

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
            result = self._session_refs.evict_radix_session(
                req.session_id, req.session_generation
            )
            self._log_deref_result(req, result)
            return

        if not has_reusable_leaf:
            return

        self._session_refs.register_session_ref(req)

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
