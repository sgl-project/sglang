# SPDX-License-Identifier: Apache-2.0
"""SRT tokenizer-manager transport for `/v1/omni/*` routes.

Requests enter through the SRT HTTP server and are forwarded to the scheduler
process. The scheduler lazily builds the omni coordinator. Actual SRT
ModelRunner/session/KV execution remains in `sglang.srt.omni_session.runtime`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from sglang.omni.configs.registry import (
    get_or_create_omni_coordinator_from_scheduler,
    resolve_omni_model_key,
)
from sglang.omni.core.protocol import OmniContextBundle, OmniRequest
from sglang.omni.entrypoints.serialization import serialize_response
from sglang.omni.entrypoints.streaming import OmniStreamSink
from sglang.omni.runtime.srt_scheduler_state import OmniSessionRecord

if TYPE_CHECKING:
    from sglang.omni.core.coordinator import OmniCoordinator
    from sglang.srt.managers.scheduler import Scheduler


def handle_omni_generate_with_omni_coordinator(
    *,
    scheduler: "Scheduler",
    payload: dict[str, Any],
    stream_sink: OmniStreamSink | None = None,
) -> dict[str, Any]:
    action = str(payload.get("action", "")).lower()
    if action == "close_session":
        return _close_omni_session(
            scheduler,
            _resolve_required_session_id(payload),
        )

    request = OmniRequest.from_payload(payload)
    sessions = _scheduler_sessions(scheduler)
    session_id = _resolve_session_id(payload)
    keep_session = bool(payload.get("keep_session", False) or session_id)
    request.metadata["finish_turn_after_generation"] = keep_session
    session_record = None
    if session_id is not None:
        session_record = sessions.get(session_id)
        if session_record is None:
            raise ValueError(f"Unknown omni session: {session_id}")
    model_key = _resolve_model_key(
        request=request,
        session_record=session_record,
    )
    coordinator = _get_coordinator(
        scheduler,
        model_key,
    )

    # generate with request and session context
    response, context = coordinator.generate_with_context(
        request,
        context=None if session_record is None else session_record.context,
        release_context=not keep_session,
        stop_after_generation_limit=keep_session,
        stream_sink=stream_sink,
    )
    response_payload = serialize_response(response)

    if keep_session:
        session_id = _context_session_id(context)
        now = time.time()
        if session_record is None:
            session_record = OmniSessionRecord(
                context=context,
                model_key=model_key,
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
    coordinator = _get_coordinator(
        scheduler,
        session_record.model_key,
    )
    coordinator.ar_backend.release(session_record.context)
    return {"session": {"id": session_id, "alive": False}}


def _get_coordinator(
    scheduler: "Scheduler",
    model_key: str,
) -> "OmniCoordinator":
    return get_or_create_omni_coordinator_from_scheduler(
        scheduler=scheduler,
        model_name=model_key,
    )


def _resolve_model_key(
    *,
    request: OmniRequest,
    session_record: OmniSessionRecord | None,
) -> str:
    if session_record is None:
        return resolve_omni_model_key(request.model)
    if request.model is None:
        return session_record.model_key
    model_key = resolve_omni_model_key(request.model)
    if model_key != session_record.model_key:
        raise ValueError(
            "Omni session model mismatch: "
            f"expected {session_record.model_key!r}, got {model_key!r}"
        )
    return model_key


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
