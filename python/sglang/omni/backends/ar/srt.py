# SPDX-License-Identifier: Apache-2.0
"""SRT-backed AR backend for omni orchestration.

The backend translates generic omni requests into the lower-level
`srt.omni_session` session adapter. SRT remains the owner of sessions, tokenizer state,
and KV cache; omni only receives context references and a narrow ContextOps
view for generation-side execution.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Literal

from sglang.omni.core.interleaved import (
    INTERLEAVED_BOUNDARY_MODALITY_KEY,
    INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY,
    STREAMED_TEXT_METADATA_KEY,
    TEXT_ROLE_METADATA_KEY,
)
from sglang.omni.core.protocol import (
    ARBackend,
    ContextOps,
    GeneratedSegment,
    OmniBoundary,
    OmniContextBundle,
    OmniContextRef,
    OmniInputSegment,
    OmniRequest,
    TemporaryForwardPrepared,
)
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniVLMTextGenerationResult,
)
from sglang.srt.omni_session.runtime_types import (
    OmniContextBundle as SRTOmniContextBundle,
)
from sglang.srt.omni_session.runtime_types import (
    OmniContextHandle as SRTOmniContextHandle,
)
from sglang.srt.omni_session.session_adapter import SRTBackedOmniSessionAdapter

if TYPE_CHECKING:
    from sglang.omni.entrypoints.streaming import OmniStreamSink


@dataclass(slots=True)
class _VLMBackendContext:
    result: OmniVLMTextGenerationResult


@dataclass(slots=True)
class SRTBackedContextOps(ContextOps):
    """Expose live SRT context operations to a colocated generation backend."""

    session_adapter: SRTBackedOmniSessionAdapter
    context: OmniContextBundle

    @property
    def metadata(self) -> dict[str, Any]:
        return _srt_backend_context(self.context).full.metadata

    @property
    def generation_kind(self) -> str | None:
        return self.session_adapter.generation_kind

    @property
    def session_id(self) -> str:
        session = _srt_backend_context(self.context).full.session
        if session is None:
            raise ValueError("SRT-backed omni context ops require a session handle")
        return session.session_id

    def get_role(self, name: str, default: str) -> str:
        return self.session_adapter.get_condition_path_role(name, default)

    def get_model(self) -> Any:
        return self.session_adapter.runtime.srt_request_executor.get_srt_model()

    def get_position_count(
        self,
        *,
        condition_path_role: str | None = None,
    ) -> int | None:
        return self.session_adapter.runtime.srt_request_executor.get_latest_session_position_count(
            self.session_id,
            condition_path_role=condition_path_role,
        )

    def run_temporary_forward(
        self,
        *,
        prepared: TemporaryForwardPrepared,
        forward: Callable[[Any], Any],
    ) -> Any:
        return self.session_adapter.runtime.srt_request_executor.run_temporary_context_forward(
            prepared=prepared,
            forward=forward,
        )


class SRTARBackend(ARBackend):
    """AR backend that delegates AR-side work to an SRT-owned session adapter."""

    def __init__(self, session_adapter: SRTBackedOmniSessionAdapter):
        self.session_adapter = session_adapter

    def begin_request_context(
        self,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        if request.mode == "vlm":
            return self._prefill_and_decode_vlm_answer(request)

        context = self._prefill_and_decode_to_image_boundary(
            request,
            stream_sink=stream_sink,
        )
        context.metadata["pending_boundaries"] = _initial_boundaries(
            context, mode=request.mode
        )
        return context

    def append_input_segments(
        self,
        context: OmniContextBundle,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        if request.mode == "vlm":
            raise ValueError("Persistent omni sessions do not support vlm mode")
        if _is_vlm_backend_context(context):
            raise ValueError("Cannot continue a vlm context as an omni session")
        session_id = _session_id_for_context(context)
        context = self._prefill_and_decode_to_image_boundary(
            request,
            session_id=session_id,
            stream_sink=stream_sink,
        )
        context.metadata["pending_boundaries"] = _initial_boundaries(
            context, mode=request.mode
        )
        return context

    def _prefill_and_decode_to_image_boundary(
        self,
        request: OmniRequest,
        *,
        session_id: str | None = None,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        context = _srt_context_bundle_to_omni(
            self.session_adapter.prefill_and_decode_to_image_boundary(
                messages=[_to_legacy_message(message) for message in request.messages],
                think=request.think,
                think_max_new_tokens=request.think_max_new_tokens,
                sampling_params=request.sampling_params,
                session_id=session_id,
                stream_sink=stream_sink,
            )
        )
        return context

    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniBoundary:
        pending_boundaries = context.metadata.get("pending_boundaries")
        if pending_boundaries:
            return _record_generation_boundary(context, pending_boundaries.pop(0))
        return _record_generation_boundary(
            context,
            _decode_result_to_boundary(
                self.session_adapter.continue_ar_decode(
                    contexts=_srt_backend_context(context),
                    stream_sink=stream_sink,
                )
            ),
        )

    def append_generated_segment(
        self,
        context: OmniContextBundle,
        segment: GeneratedSegment,
        *,
        request: OmniRequest,
    ) -> OmniContextBundle:
        if request.mode != "interleave":
            return context
        self.session_adapter.commit_generated_segment(
            contexts=_srt_backend_context(context),
            segment=segment,
        )
        if (
            request.metadata.get("finish_turn_after_generation")
            and request.max_text_segments == 0
        ):
            # 1. only force-close when the caller explicitly disables post-image AR
            self.session_adapter.finish_generated_segment_turn(
                contexts=_srt_backend_context(context)
            )
        return context

    def get_context_ops(self, context: OmniContextBundle) -> SRTBackedContextOps:
        return SRTBackedContextOps(self.session_adapter, context)

    def release(self, context: OmniContextBundle) -> None:
        if _is_vlm_backend_context(context):
            _release_vlm_context(self.session_adapter, context)
            return
        self.session_adapter.release(_srt_backend_context(context))

    def _prefill_and_decode_vlm_answer(self, request: OmniRequest) -> OmniContextBundle:
        result = self.session_adapter.generate_vlm_answer(
            messages=[_to_legacy_message(message) for message in request.messages],
            max_new_tokens=_resolve_vlm_max_new_tokens(request),
        )
        session = result.session
        metadata = _vlm_text_metadata(result)
        context = OmniContextBundle(
            full=OmniContextRef(
                context_id=session.anchor_request_id,
                token_count=session.context_length,
                session_id=session.session_id,
                version=session.context_version,
                metadata={"mode": "vlm"},
            ),
            metadata={
                "pending_boundaries": [
                    OmniBoundary(
                        type="text",
                        text=result.text,
                        token_ids=tuple(
                            int(token_id) for token_id in metadata.pop("token_ids", ())
                        ),
                        metadata=metadata,
                    ),
                    OmniBoundary(type="done"),
                ],
            },
            backend_context=_VLMBackendContext(result=result),
        )
        return context


def _to_legacy_message(message: OmniInputSegment) -> dict[str, Any]:
    if message.type == "text":
        return {"type": "text", "text": message.text or ""}
    if message.type == "image":
        return {"type": "image", "image": _decode_image_payload(message.image)}
    if message.type == "audio":
        return {"type": "audio", "audio": message.audio}
    if message.type == "video":
        return {"type": "video", "video": message.video}
    raise ValueError(f"Unsupported omni input segment type: {message.type!r}")


def _decode_image_payload(image: Any) -> Any:
    if not isinstance(image, dict):
        return image
    b64_json = image.get("b64_json") or image.get("base64")
    if b64_json is None:
        return image
    if isinstance(b64_json, str) and "," in b64_json:
        b64_json = b64_json.split(",", 1)[1]
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(b64_json))).convert("RGB")


def _decode_result_to_boundary(boundary: OmniDecodeResult) -> OmniBoundary:
    boundary_type: Literal["text", "image", "done"]
    boundary_type = "image" if boundary.type == "image_marker" else boundary.type
    return OmniBoundary(
        type=boundary_type,
        text=boundary.text,
        token_ids=boundary.token_ids,
        metadata=dict(boundary.metadata),
    )


def _record_generation_boundary(
    context: OmniContextBundle,
    boundary: OmniBoundary,
) -> OmniBoundary:
    if boundary.type not in {"image", "audio", "video"}:
        return boundary
    metadata = dict(boundary.metadata)
    metadata.setdefault(INTERLEAVED_BOUNDARY_MODALITY_KEY, boundary.type)
    context.metadata[INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY] = metadata
    backend_context = context.backend_context
    if isinstance(backend_context, SRTOmniContextBundle):
        backend_context.full.metadata[INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY] = (
            metadata
        )
    return OmniBoundary(
        type=boundary.type,
        text=boundary.text,
        token_ids=boundary.token_ids,
        metadata=metadata,
    )


def _srt_context_bundle_to_omni(context: SRTOmniContextBundle) -> OmniContextBundle:
    return OmniContextBundle(
        full=_srt_context_handle_to_omni(context.full),
        text_cfg=_srt_context_handle_to_omni(context.text_cfg),
        image_cfg=_srt_context_handle_to_omni(context.image_cfg),
        metadata={},
        backend_context=context,
    )


def _srt_context_handle_to_omni(ref: SRTOmniContextHandle) -> OmniContextRef:
    session = ref.session
    return OmniContextRef(
        context_id=ref.request_id,
        token_count=ref.token_count,
        session_id=None if session is None else session.session_id,
        version=0 if session is None else session.context_version,
        metadata=dict(ref.metadata),
    )


def _srt_backend_context(context: OmniContextBundle) -> SRTOmniContextBundle:
    backend_context = context.backend_context
    if not isinstance(backend_context, SRTOmniContextBundle):
        raise ValueError("SRT omni context requires an SRT backend context")
    return backend_context


def _session_id_for_context(context: OmniContextBundle) -> str:
    if context.full.session_id is not None:
        return context.full.session_id
    raise ValueError("Persistent omni session requires a live SRT session")


def _is_vlm_backend_context(context: OmniContextBundle) -> bool:
    return isinstance(context.backend_context, _VLMBackendContext)


def _release_vlm_context(
    session_adapter: SRTBackedOmniSessionAdapter,
    context: OmniContextBundle,
) -> None:
    backend_context = context.backend_context
    if not isinstance(backend_context, _VLMBackendContext):
        return
    session_adapter.runtime.close_session(backend_context.result.session)


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


def _vlm_text_metadata(result: OmniVLMTextGenerationResult) -> dict[str, Any]:
    metadata = {}
    if result.next_token_ids:
        metadata["token_ids"] = [int(token_id) for token_id in result.next_token_ids]
    if result.token_ids:
        metadata["input_token_ids"] = [int(token_id) for token_id in result.token_ids]
    if result.position_ids:
        metadata["position_ids"] = [
            int(position_id) for position_id in result.position_ids
        ]
    return metadata


def _initial_boundaries(
    context: OmniContextBundle,
    *,
    mode: str,
) -> list[OmniBoundary]:
    boundaries = _pre_image_segments_to_boundaries(
        context.full.metadata.get("pre_image_segments", [])
    )
    if not context.full.metadata.get("pre_image_reached_image_marker", True):
        boundaries.append(OmniBoundary(type="done"))
        return boundaries
    boundaries.append(
        OmniBoundary(
            type="image",
            metadata=dict(
                context.full.metadata.get("pre_image_boundary_metadata") or {}
            ),
        )
    )
    if mode != "interleave":
        boundaries.append(OmniBoundary(type="done"))
    return boundaries


def _pre_image_segments_to_boundaries(
    segments: list[dict[str, Any]],
) -> list[OmniBoundary]:
    boundaries: list[OmniBoundary] = []
    text_parts: list[str] = []
    text_token_ids: list[int] = []
    text_metadata: dict[str, Any] = {}

    def flush_text() -> None:
        if not text_parts and not text_token_ids:
            return
        boundaries.append(
            OmniBoundary(
                type="text",
                text="".join(text_parts),
                token_ids=tuple(text_token_ids),
                metadata=dict(text_metadata),
            )
        )
        text_parts.clear()
        text_token_ids.clear()
        text_metadata.clear()

    for segment in segments:
        if segment.get("type") != "text":
            flush_text()
            boundaries.append(_pre_image_segment_to_boundary(segment))
            continue
        boundary = _pre_image_segment_to_boundary(segment)
        if text_parts and (
            text_metadata.get(STREAMED_TEXT_METADATA_KEY)
            != boundary.metadata.get(STREAMED_TEXT_METADATA_KEY)
            or text_metadata.get(TEXT_ROLE_METADATA_KEY)
            != boundary.metadata.get(TEXT_ROLE_METADATA_KEY)
        ):
            flush_text()
        text_parts.append(boundary.text or "")
        text_token_ids.extend(int(token_id) for token_id in boundary.token_ids)
        if boundary.metadata:
            text_metadata.update(boundary.metadata)

    flush_text()
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
