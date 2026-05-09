# SPDX-License-Identifier: Apache-2.0
"""SRT tokenizer-manager transport for `/v1/omni/*` routes.

Requests enter through the SRT HTTP server and are forwarded to the scheduler
process. The scheduler lazily builds the omni orchestrator so the colocated U1
generation backend can access the local ModelRunner/session/KV state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sglang.omni.entrypoints.http_server import _serialize_response
from sglang.omni.protocol import OmniRequest

_SENSENOVA_U1_CACHE_KEY = "sensenova-u1"


@dataclass(slots=True)
class _OmniSessionRecord:
    context: Any
    turns: int
    created_at: float
    updated_at: float


def handle_omni_generate_from_scheduler(
    *,
    scheduler: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    action = str(payload.get("action", "")).lower()
    if action == "close_session":
        return _close_omni_session(
            scheduler,
            _resolve_required_session_id(payload),
        )

    request = OmniRequest.from_payload(payload)
    orchestrator = _get_sensenova_u1_orchestrator(scheduler, request)
    sessions = _get_omni_sessions(scheduler)
    session_id = _resolve_session_id(payload)
    keep_session = bool(payload.get("keep_session", False) or session_id)
    session_record = None
    if session_id is not None:
        session_record = sessions.get(session_id)
        if session_record is None:
            raise ValueError(f"Unknown omni session: {session_id}")

    response, context = orchestrator.generate_with_context(
        request,
        context=None if session_record is None else session_record.context,
        release_context=not keep_session,
        stop_after_generation_limit=keep_session,
    )
    response_payload = _serialize_response(response)

    if keep_session:
        session_id = _context_session_id(context)
        now = time.time()
        if session_record is None:
            session_record = _OmniSessionRecord(
                context=context,
                turns=0,
                created_at=now,
                updated_at=now,
            )
        session_record.context = context
        session_record.turns += 1
        session_record.updated_at = now
        sessions[session_id] = session_record
        response_payload["session"] = {
            "id": session_id,
            "turns": session_record.turns,
            "alive": True,
        }
    return response_payload


def _close_omni_session(scheduler: Any, session_id: str) -> dict[str, Any]:
    sessions = _get_omni_sessions(scheduler)
    session_record = sessions.pop(session_id, None)
    if session_record is None:
        raise ValueError(f"Unknown omni session: {session_id}")
    orchestrator = _get_sensenova_u1_orchestrator_by_model(
        scheduler,
        _SENSENOVA_U1_CACHE_KEY,
    )
    orchestrator.ar_backend.release(session_record.context)
    return {"session": {"id": session_id, "alive": False}}


def _get_sensenova_u1_orchestrator(scheduler: Any, request: OmniRequest) -> Any:
    return _get_sensenova_u1_orchestrator_by_model(
        scheduler,
        request.model or _SENSENOVA_U1_CACHE_KEY,
    )


def _get_sensenova_u1_orchestrator_by_model(scheduler: Any, model_name: str) -> Any:
    model = model_name.lower()
    if "sensenova" not in model and "u1" not in model:
        raise ValueError(f"Unsupported omni model {model_name!r}")

    cache = getattr(scheduler, "_omni_orchestrators", None)
    if cache is None:
        cache = {}
        setattr(scheduler, "_omni_orchestrators", cache)
    if _SENSENOVA_U1_CACHE_KEY not in cache:
        from sglang.omni.configs.sensenova_u1 import (
            build_sensenova_u1_orchestrator_from_scheduler,
        )

        cache[_SENSENOVA_U1_CACHE_KEY] = build_sensenova_u1_orchestrator_from_scheduler(
            scheduler=scheduler,
            server_args=getattr(scheduler, "server_args", None),
        )
    return cache[_SENSENOVA_U1_CACHE_KEY]


def _get_omni_sessions(scheduler: Any) -> dict[str, _OmniSessionRecord]:
    sessions = getattr(scheduler, "_omni_sessions", None)
    if sessions is None:
        sessions = {}
        setattr(scheduler, "_omni_sessions", sessions)
    return sessions


def _resolve_session_id(payload: dict[str, Any]) -> str | None:
    session_id = payload.get("session_id")
    if session_id is None and isinstance(payload.get("session"), dict):
        session_id = payload["session"].get("id")
    if session_id is None and isinstance(payload.get("metadata"), dict):
        session_id = payload["metadata"].get("session_id")
    if session_id is None:
        return None
    session_id = str(session_id).strip()
    return session_id or None


def _resolve_required_session_id(payload: dict[str, Any]) -> str:
    session_id = _resolve_session_id(payload)
    if session_id is None:
        raise ValueError("Omni session action requires session_id")
    return session_id


def _context_session_id(context: Any) -> str:
    session_id = getattr(getattr(context, "full", None), "session_id", None)
    if session_id is None:
        backend_context = getattr(context, "backend_context", None)
        full = getattr(backend_context, "full", None)
        session = getattr(full, "session", None)
        session_id = getattr(session, "session_id", None)
    if session_id is None:
        raise ValueError("Persistent omni request did not produce a session id")
    return str(session_id)
