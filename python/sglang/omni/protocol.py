# SPDX-License-Identifier: Apache-2.0
"""Shared omni contracts between orchestration and execution backends.

The protocol deliberately carries modality segments and context references,
not model-specific session objects. AR backends own token/session state, while
generation backends own media synthesis and may choose colocated or standalone
serving runtimes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

InputSegmentType = Literal["text", "image", "audio", "video"]
OutputSegmentType = Literal["text", "image", "audio", "video"]
BoundaryType = Literal["text", "image", "audio", "video", "done"]


@dataclass(frozen=True, slots=True)
class OmniInputSegment:
    """One user-provided segment in a multimodal request."""

    type: InputSegmentType
    text: str | None = None
    image: Any | None = None
    audio: Any | None = None
    video: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OmniInputSegment":
        segment_type = payload.get("type")
        if segment_type not in {"text", "image", "audio", "video"}:
            raise ValueError(f"Unsupported omni input segment type: {segment_type!r}")
        return cls(
            type=segment_type,
            text=payload.get("text", payload.get("content")),
            image=payload.get("image"),
            audio=payload.get("audio"),
            video=payload.get("video"),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        value = getattr(self, self.type)
        if self.type == "text":
            payload["text"] = value or ""
        else:
            payload[self.type] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class OmniRequest:
    """Normalized request consumed by the omni coordinator."""

    messages: tuple[OmniInputSegment, ...]
    model: str | None = None
    mode: str = "interleave"
    sampling_params: Any | None = None
    max_images: int = 1
    max_text_segments: int = 8
    think: bool = False
    think_max_new_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OmniRequest":
        messages = tuple(
            (
                segment
                if isinstance(segment, OmniInputSegment)
                else OmniInputSegment.from_dict(segment)
            )
            for segment in payload.get("messages", ())
        )
        if not messages:
            raise ValueError("Omni request requires at least one message")
        return cls(
            messages=messages,
            model=payload.get("model"),
            mode=str(payload.get("task", payload.get("mode", "interleave"))),
            sampling_params=payload.get("sampling_params") or {},
            max_images=int(payload.get("max_images", 1)),
            max_text_segments=int(payload.get("max_text_segments", 8)),
            think=bool(payload.get("think", False)),
            think_max_new_tokens=payload.get("think_max_new_tokens"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class OmniContextRef:
    context_id: str
    token_count: int = 0
    session_id: str | None = None
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OmniContextBundle:
    """Coordinator-visible context plus backend-private live state.

    `backend_context` is intentionally opaque: colocated SRT backends keep live
    session handles there, while future standalone generation backends can keep
    only serializable references.
    """

    full: OmniContextRef
    text_cfg: OmniContextRef | None = None
    image_cfg: OmniContextRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # backend-owned object kept out of the public response
    backend_context: Any | None = None


@dataclass(frozen=True, slots=True)
class OmniBoundary:
    """AR-side handoff point that tells the coordinator what to do next."""

    type: BoundaryType
    text: str | None = None
    token_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeneratedSegment:
    """Media/text segment returned by a generation backend.

    `commit_payload` may differ from the user-visible segment. U1 uses this to
    commit native pixel tensors back into SRT without re-encoding the PNG output.
    """

    type: OutputSegmentType
    text: str | None = None
    image: Any | None = None
    audio: Any | None = None
    video: Any | None = None
    commit_payload: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def commit_image(self) -> Any | None:
        if self.commit_payload is not None:
            return self.commit_payload
        return self.image


@dataclass(frozen=True, slots=True)
class OmniOutputSegment:
    type: OutputSegmentType
    text: str | None = None
    image: Any | None = None
    audio: Any | None = None
    video: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_boundary(cls, boundary: OmniBoundary) -> "OmniOutputSegment":
        if boundary.type != "text":
            raise ValueError(f"Cannot convert {boundary.type!r} boundary to text")
        metadata = dict(boundary.metadata)
        if boundary.token_ids:
            metadata["token_ids"] = [int(token_id) for token_id in boundary.token_ids]
        return cls(type="text", text=boundary.text or "", metadata=metadata)

    @classmethod
    def from_generated(cls, segment: GeneratedSegment) -> "OmniOutputSegment":
        return cls(
            type=segment.type,
            text=segment.text,
            image=segment.image,
            audio=segment.audio,
            video=segment.video,
            metadata=dict(segment.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.type == "text":
            payload["text"] = self.text or ""
        else:
            payload[self.type] = getattr(self, self.type)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class OmniResponse:
    segments: tuple[OmniOutputSegment, ...]
    context: OmniContextRef | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "segments": [segment.to_dict() for segment in self.segments],
            "stats": dict(self.stats),
            "metadata": dict(self.metadata),
        }
        if self.context is not None:
            payload["context"] = asdict(self.context)
        return payload


class ContextOps(Protocol):
    """Narrow live-context capability exposed to generation backends."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def generation_kind(self) -> str | None: ...

    @property
    def session_id(self) -> str | None: ...

    def get_role(self, name: str, default: str) -> str: ...

    def get_model(self) -> Any: ...

    def get_position_count(
        self,
        *,
        sidecar_role: str | None = None,
    ) -> int | None: ...

    def build_temporary_forward_batch(
        self,
        *,
        prepared: Any,
        generation_query_embeds: Any,
        timestep: Any,
    ) -> Any: ...


class ARBackend(Protocol):
    """Autoregressive text/session backend used by omni orchestration."""

    def prepare_context(self, request: OmniRequest) -> OmniContextBundle: ...

    def append_input_segments(
        self,
        context: OmniContextBundle,
        request: OmniRequest,
    ) -> OmniContextBundle: ...

    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
    ) -> OmniBoundary: ...

    def append_generated_segment(
        self,
        context: OmniContextBundle,
        segment: GeneratedSegment,
        *,
        request: OmniRequest,
    ) -> OmniContextBundle: ...

    def get_context_ops(self, context: OmniContextBundle) -> ContextOps: ...

    def release(self, context: OmniContextBundle) -> None: ...


class GenerationBackend(Protocol):
    """Media generation backend used by omni orchestration."""

    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment: ...
