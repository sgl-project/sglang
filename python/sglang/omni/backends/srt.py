# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sglang.omni.protocol import (
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniContextRef,
    OmniInputSegment,
    OmniRequest,
)


@dataclass(slots=True)
class SRTBackedContextOps:
    bridge: Any
    context: OmniContextBundle

    @property
    def metadata(self) -> dict[str, Any]:
        return _backend_context(self.context).full.metadata

    @property
    def g_kind(self) -> str | None:
        return getattr(self.bridge, "g_kind", None)

    @property
    def session_id(self) -> str:
        session = _backend_context(self.context).full.session
        if session is None:
            raise ValueError("SRT-backed omni context ops require a session handle")
        return session.session_id

    def get_role(self, name: str, default: str) -> str:
        attr_name = name if name.endswith("_role") else f"{name}_role"
        return str(getattr(self.bridge, attr_name, default))

    def get_model(self) -> Any:
        return self.bridge.runtime.srt_request_executor.get_srt_model()

    def get_position_count(
        self,
        *,
        sidecar_role: str | None = None,
    ) -> int | None:
        return self.bridge.runtime.srt_request_executor.get_latest_session_position_count(
            self.session_id,
            sidecar_role=sidecar_role,
        )

    def build_temporary_forward_batch(
        self,
        *,
        prepared: Any,
        g_query_embeds: Any,
        timestep: Any,
    ) -> Any:
        return self.bridge.runtime.srt_request_executor.build_temporary_context_forward_batch_for_session(
            prepared=self._to_srt_prepared(prepared),
            g_query_embeds=g_query_embeds,
            timestep=timestep,
        )

    def _to_srt_prepared(self, prepared: Any) -> Any:
        if getattr(prepared, "srt_session_id", None) is not None:
            return prepared
        data = dict(getattr(prepared, "__dict__", {}) or {})
        data["srt_session_id"] = data.get("session_id", self.session_id)
        data["srt_sidecar_role"] = data.get("sidecar_role")
        return SimpleNamespace(**data)


class SRTARBackend:
    def __init__(self, bridge: Any):
        self.bridge = bridge

    def prepare_context(self, request: OmniRequest) -> OmniContextBundle:
        if request.mode == "vlm":
            return self._prepare_vlm_context(request)

        context = self.bridge.prepare_u_context_from_messages(
            messages=[_to_legacy_message(message) for message in request.messages],
            think=request.think,
            think_max_new_tokens=request.think_max_new_tokens,
            sampling_params=request.sampling_params,
        )
        context = _coerce_context_bundle(context)
        context.metadata["pending_boundaries"] = _initial_boundaries(context)
        return context

    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
    ) -> OmniBoundary:
        del request
        pending_boundaries = context.metadata.get("pending_boundaries")
        if pending_boundaries:
            return pending_boundaries.pop(0)
        return _coerce_boundary(
            self.bridge.continue_u_decode(contexts=_backend_context(context))
        )

    def append_generated_segment(
        self,
        context: OmniContextBundle,
        segment: GeneratedSegment,
        *,
        request: OmniRequest,
    ) -> OmniContextBundle:
        del request
        self.bridge.commit_generated_segment(
            contexts=_backend_context(context),
            segment=segment,
        )
        return context

    def get_context_ops(self, context: OmniContextBundle) -> SRTBackedContextOps:
        return SRTBackedContextOps(self.bridge, context)

    def release(self, context: OmniContextBundle) -> None:
        if _is_vlm_backend_context(context):
            _release_vlm_context(self.bridge, context)
            return
        self.bridge.release(_backend_context(context))

    def _prepare_vlm_context(self, request: OmniRequest) -> OmniContextBundle:
        result = self.bridge.generate_vlm_text(
            messages=[_to_legacy_message(message) for message in request.messages],
            max_new_tokens=_resolve_vlm_max_new_tokens(request),
        )
        session = getattr(result, "session", None)
        metadata = _vlm_text_metadata(result)
        context = OmniContextBundle(
            full=OmniContextRef(
                context_id=str(
                    getattr(session, "anchor_request_id", None)
                    or getattr(session, "session_id", None)
                    or "vlm"
                ),
                token_count=int(getattr(session, "context_length", 0) or 0),
                session_id=getattr(session, "session_id", None),
                metadata={"mode": "vlm"},
            ),
            metadata={
                "pending_boundaries": [
                    OmniBoundary(
                        type="text",
                        text=getattr(result, "text", "") or "",
                        token_ids=tuple(
                            int(token_id)
                            for token_id in metadata.pop("token_ids", ())
                        ),
                        metadata=metadata,
                    ),
                    OmniBoundary(type="done"),
                ],
            },
            backend_context=SimpleNamespace(vlm_result=result),
        )
        return context


def _to_legacy_message(message: OmniInputSegment) -> dict[str, Any]:
    if message.type == "text":
        return {"type": "text", "text": message.text or ""}
    return {"type": message.type, message.type: getattr(message, message.type)}


def _coerce_boundary(boundary: Any) -> OmniBoundary:
    if isinstance(boundary, OmniBoundary):
        return boundary
    boundary_type = getattr(boundary, "type")
    if boundary_type == "image_marker":
        boundary_type = "image"
    return OmniBoundary(
        type=boundary_type,
        text=getattr(boundary, "text", None),
        token_ids=tuple(getattr(boundary, "token_ids", ()) or ()),
        metadata=dict(getattr(boundary, "metadata", {}) or {}),
    )


def _coerce_context_bundle(context: Any) -> OmniContextBundle:
    if isinstance(context, OmniContextBundle):
        return context

    full = getattr(context, "full", context)
    text_cfg = getattr(context, "text_cfg", None)
    image_cfg = getattr(context, "image_cfg", None)
    return OmniContextBundle(
        full=_coerce_context_ref(full),
        text_cfg=_coerce_context_ref(text_cfg) if text_cfg is not None else None,
        image_cfg=_coerce_context_ref(image_cfg) if image_cfg is not None else None,
        metadata=dict(getattr(context, "metadata", {}) or {}),
        backend_context=context,
    )


def _coerce_context_ref(ref: Any) -> OmniContextRef:
    if isinstance(ref, OmniContextRef):
        return ref
    session = getattr(ref, "session", None)
    return OmniContextRef(
        context_id=str(
            getattr(ref, "context_id", None)
            or getattr(ref, "request_id", None)
            or getattr(session, "anchor_request_id", "")
        ),
        token_count=int(
            getattr(ref, "token_count", None)
            or getattr(ref, "context_length", None)
            or getattr(session, "context_length", 0)
            or 0
        ),
        session_id=getattr(ref, "session_id", None)
        or getattr(session, "session_id", None),
        version=int(
            getattr(ref, "version", None)
            or getattr(ref, "context_version", None)
            or getattr(session, "context_version", 0)
            or 0
        ),
        metadata=dict(getattr(ref, "metadata", {}) or {}),
    )


def _backend_context(context: OmniContextBundle) -> Any:
    return context.backend_context if context.backend_context is not None else context


def _is_vlm_backend_context(context: OmniContextBundle) -> bool:
    return getattr(_backend_context(context), "vlm_result", None) is not None


def _release_vlm_context(bridge: Any, context: OmniContextBundle) -> None:
    result = getattr(_backend_context(context), "vlm_result", None)
    session = getattr(result, "session", None)
    if session is None:
        return
    runtime = getattr(bridge, "runtime", None)
    close_session = getattr(runtime, "close_session", None)
    if callable(close_session):
        close_session(session)


def _resolve_vlm_max_new_tokens(request: OmniRequest) -> int:
    value = request.metadata.get("max_new_tokens", request.metadata.get("max_length"))
    if value is None:
        value = getattr(request.sampling_params, "max_new_tokens", None)
    if value is None:
        value = getattr(request.sampling_params, "max_length", None)
    if value is None:
        value = request.think_max_new_tokens
    if value is None:
        value = 1024
    value = int(value)
    if value <= 0:
        raise ValueError(f"VLM max_new_tokens must be positive, got {value}")
    return value


def _vlm_text_metadata(result: Any) -> dict[str, Any]:
    metadata = {}
    output_ids = getattr(result, "next_token_ids", ())
    if output_ids:
        metadata["token_ids"] = [int(token_id) for token_id in output_ids]
    input_ids = getattr(result, "token_ids", ())
    if input_ids:
        metadata["input_token_ids"] = [int(token_id) for token_id in input_ids]
    position_ids = getattr(result, "position_ids", ())
    if position_ids:
        metadata["position_ids"] = [int(position_id) for position_id in position_ids]
    return metadata


def _initial_boundaries(context: OmniContextBundle) -> list[OmniBoundary]:
    boundaries = [
        _pre_image_segment_to_boundary(segment)
        for segment in context.full.metadata.get("pre_image_segments", [])
    ]
    boundaries.append(OmniBoundary(type="image"))
    return boundaries


def _pre_image_segment_to_boundary(segment: dict[str, Any]) -> OmniBoundary:
    metadata = dict(segment.get("metadata") or {})
    token_ids = tuple(metadata.pop("token_ids", ()) or ())
    return OmniBoundary(
        type="text",
        text=segment.get("text") or "",
        token_ids=token_ids,
        metadata=metadata,
    )
