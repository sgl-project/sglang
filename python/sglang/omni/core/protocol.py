# SPDX-License-Identifier: Apache-2.0
"""Shared omni contracts between orchestration and execution backends.

The contracts deliberately carry modality segments and context references,
not model-specific session objects. AR backends own token/session state, while
generation backends own media synthesis and may choose colocated or standalone
serving runtimes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from sglang.omni.entrypoints.streaming import OmniStreamSink

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
    # vlm: image-understanding
    # interleave: image-text -> image-text

    model: str | None = None
    mode: str = "interleave"
    sampling_params: Any | None = None
    # max number of images allowed in a req
    max_images: int = 1
    max_text_segments: int = 8
    max_text_segments_after_media: int | None = None
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
            max_text_segments_after_media=(
                None
                if payload.get("max_text_segments_after_media") is None
                else int(payload.get("max_text_segments_after_media"))
            ),
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
    """Coordinator-visible context (for omni-gen) plus backend-private live state.

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
    """AR-side handoff point that tells the coordinator what to do next.

    'Boundary' is a common and crucial concept in interleaved (ar + multimodal_gen) generation.
     Conceptually, it marks the boundary of continuous token of different modalities, usually decided by the AR backend

    Media boundary metadata may carry the AR-emitted marker token and optional
    generation token budget for models where AR decides image/audio/video length.
    """

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
class TemporaryForwardPrepared:
    """Temporary query metadata for generation backends that read live SRT KV."""

    generation_input: dict[str, Any]
    srt_session_id: str
    condition_path_role: str | None = None


@dataclass(frozen=True, slots=True)
class OmniOutputSegment:
    """OmniOutputSegment is the basic element of the result of an omni request"""

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


class ContextOps(ABC):
    """Narrow live-context capability exposed to generation backends."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def generation_kind(self) -> str | None: ...

    @property
    @abstractmethod
    def session_id(self) -> str | None: ...

    @abstractmethod
    def get_role(self, name: str, default: str) -> str: ...

    @abstractmethod
    def get_model(self) -> Any: ...

    @abstractmethod
    def get_position_count(
        self,
        *,
        condition_path_role: str | None = None,
    ) -> int | None: ...

    @abstractmethod
    def run_temporary_forward(
        self,
        *,
        prepared: TemporaryForwardPrepared,
        forward: Callable[[Any], Any],
    ) -> Any:
        """run a short-lived generation forward while the backend owns KV lifetime"""

        ...


class ARBackend(ABC):
    """Autoregressive text/session backend used by omni coordinator"""

    @abstractmethod
    def begin_request_context(
        self,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle: ...

    @abstractmethod
    def append_input_segments(
        self,
        context: OmniContextBundle,
        request: OmniRequest,
        *,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle: ...

    @abstractmethod
    def decode_until_boundary(
        self,
        context: OmniContextBundle,
        *,
        request: OmniRequest,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniBoundary: ...

    @abstractmethod
    def append_generated_segment(
        self,
        context: OmniContextBundle,
        segment: GeneratedSegment,
        *,
        request: OmniRequest,
    ) -> OmniContextBundle: ...

    @abstractmethod
    def get_context_ops(self, context: OmniContextBundle) -> ContextOps: ...

    @abstractmethod
    def release(self, context: OmniContextBundle) -> None: ...


class MultimodalGenerationBackend(ABC):
    """Media generation backend used by omni orchestration."""

    @abstractmethod
    def generate_segment(
        self,
        request: OmniRequest,
        context_ops: ContextOps,
    ) -> GeneratedSegment: ...
