# SPDX-License-Identifier: Apache-2.0
"""SRT tokenizer-manager transport for `/v1/omni/*` routes.

Requests enter through the SRT HTTP server and are forwarded to the scheduler
process. The scheduler lazily builds the omni orchestrator so the same-process
U1 generation backend can access local ModelRunner/session/KV state.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sglang.omni.configs.sensenova_u1 import (
    build_sensenova_u1_orchestrator_from_scheduler,
)
from sglang.omni.protocol import OmniContextBundle, OmniRequest
from sglang.omni.scheduler_state import OmniSessionRecord
from sglang.omni.serialization import serialize_response

if TYPE_CHECKING:
    from sglang.omni.coordinator import OmniCoordinator
    from sglang.srt.managers.scheduler import Scheduler

_SENSENOVA_U1_CACHE_KEY = "sensenova-u1"


def handle_omni_generate_with_omni_coordinator(
    *,
    scheduler: "Scheduler",
    payload: dict[str, Any],
) -> dict[str, Any]:
    action = str(payload.get("action", "")).lower()
    if action == "close_session":
        return _close_omni_session(
            scheduler,
            _resolve_required_session_id(payload),
        )

    request = OmniRequest.from_payload(payload)
    # get global orchestrator
    orchestrator = _get_sensenova_u1_orchestrator(scheduler, request)
    sessions = _scheduler_sessions(scheduler)
    session_id = _resolve_session_id(payload)
    keep_session = bool(payload.get("keep_session", False) or session_id)
    session_record = None
    if session_id is not None:
        session_record = sessions.get(session_id)
        if session_record is None:
            raise ValueError(f"Unknown omni session: {session_id}")

    # generate with request and session context
    response, context = orchestrator.generate_with_context(
        request,
        context=None if session_record is None else session_record.context,
        release_context=not keep_session,
        stop_after_generation_limit=keep_session,
    )
    response_payload = serialize_response(response)

    if keep_session:
        session_id = _context_session_id(context)
        now = time.time()
        if session_record is None:
            session_record = OmniSessionRecord(
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


def _close_omni_session(scheduler: "Scheduler", session_id: str) -> dict[str, Any]:
    sessions = _scheduler_sessions(scheduler)
    session_record = sessions.pop(session_id, None)
    if session_record is None:
        raise ValueError(f"Unknown omni session: {session_id}")
    orchestrator = _get_sensenova_u1_orchestrator_by_model(
        scheduler,
        _SENSENOVA_U1_CACHE_KEY,
    )
    orchestrator.ar_backend.release(session_record.context)
    return {"session": {"id": session_id, "alive": False}}


def _get_sensenova_u1_orchestrator(
    scheduler: "Scheduler",
    request: OmniRequest,
) -> "OmniCoordinator":
    return _get_sensenova_u1_orchestrator_by_model(
        scheduler,
        request.model or _SENSENOVA_U1_CACHE_KEY,
    )


def _get_sensenova_u1_orchestrator_by_model(
    scheduler: "Scheduler",
    model_name: str,
) -> "OmniCoordinator":
    model = model_name.lower()
    if "sensenova" not in model and "u1" not in model:
        raise ValueError(f"Unsupported omni model {model_name!r}")

    cache = scheduler.omni_scheduler_state.orchestrators
    with scheduler.omni_scheduler_state.orchestrator_lock:
        if _SENSENOVA_U1_CACHE_KEY not in cache:
            cache[_SENSENOVA_U1_CACHE_KEY] = (
                build_sensenova_u1_orchestrator_from_scheduler(
                    scheduler=scheduler,
                    server_args=scheduler.server_args,
                )
            )
    return cache[_SENSENOVA_U1_CACHE_KEY]


def _scheduler_sessions(scheduler: "Scheduler") -> dict[str, OmniSessionRecord]:
    return scheduler.omni_scheduler_state.sessions


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


def _context_session_id(context: OmniContextBundle) -> str:
    if context.full.session_id is None:
        raise ValueError("Persistent omni request did not produce a session id")
    return context.full.session_id
