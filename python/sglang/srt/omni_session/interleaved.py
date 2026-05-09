# SPDX-License-Identifier: Apache-2.0
"""Legacy omni request/response shapes used below the omni protocol layer."""

from dataclasses import dataclass, field
from typing import Any, Literal, cast

OmniGenerationMode = Literal["t2i", "edit", "interleave", "vlm"]
GenerationKind = str
DEFAULT_OMNI_TEXT_MAX_NEW_TOKENS = 128

_OMNI_MODE_ALIASES = {
    "text_to_image": "t2i",
    "txt2img": "t2i",
    "image_edit": "edit",
    "i2i": "edit",
    "img2img": "edit",
    "interleaved": "interleave",
    "vlm_chat": "vlm",
    "chat": "vlm",
}
_OMNI_GENERATION_MODES = {"t2i", "edit", "interleave", "vlm"}


def normalize_omni_generation_mode(
    mode: Any | None,
    *,
    default: OmniGenerationMode = "interleave",
) -> OmniGenerationMode:
    if mode is None:
        return default
    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = _OMNI_MODE_ALIASES.get(normalized, normalized)
    if normalized not in _OMNI_GENERATION_MODES:
        raise ValueError(
            "Unsupported omni generation mode "
            f"{mode!r}; expected one of {sorted(_OMNI_GENERATION_MODES)}"
        )
    return cast(OmniGenerationMode, normalized)


@dataclass(frozen=True, slots=True)
class OmniInputSegment:
    """Legacy text/image input segment for SRT omni_session adapters."""

    type: Literal["text", "image"]
    text: str | None = None
    image: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, *, metadata: dict[str, Any] | None = None
    ) -> "OmniInputSegment":
        return cls(type="text", text=text, metadata=metadata or {})

    @classmethod
    def from_image(
        cls, image: Any, *, metadata: dict[str, Any] | None = None
    ) -> "OmniInputSegment":
        return cls(type="image", image=image, metadata=metadata or {})

    @classmethod
    def from_legacy_segment(cls, segment: dict[str, Any]) -> "OmniInputSegment":
        segment_type = segment.get("type")
        metadata = dict(segment.get("metadata") or {})
        if segment_type == "text":
            text = segment.get("text", segment.get("content"))
            if text is None:
                raise ValueError("omni text input segment is missing text")
            return cls.from_text(str(text), metadata=metadata)
        if segment_type == "image":
            image = segment.get("image", segment.get("content"))
            if image is None:
                raise ValueError("omni image input segment is missing image")
            return cls.from_image(image, metadata=metadata)
        raise ValueError(f"Unsupported omni input segment type: {segment_type!r}")

    def to_legacy_segment(self) -> dict[str, Any]:
        if self.type == "text":
            if self.text is None:
                raise ValueError("omni text input segment is missing text")
            segment = {"type": "text", "text": self.text}
        elif self.type == "image":
            if self.image is None:
                raise ValueError("omni image input segment is missing image")
            segment = {"type": "image", "image": self.image}
        else:
            raise ValueError(f"Unsupported omni input segment type: {self.type!r}")
        if self.metadata:
            segment["metadata"] = dict(self.metadata)
        return segment


@dataclass(frozen=True, slots=True)
class OmniInterleavedRequest:
    messages: tuple[OmniInputSegment, ...]
    sampling_params: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_segments(
        cls,
        messages: list[Any],
        *,
        sampling_params: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "OmniInterleavedRequest":
        normalized = []
        for message in messages:
            if isinstance(message, OmniInputSegment):
                normalized.append(message)
            elif isinstance(message, dict):
                normalized.append(OmniInputSegment.from_legacy_segment(message))
            elif hasattr(message, "type") and hasattr(message, "content"):
                if message.type == "text":
                    normalized.append(OmniInputSegment.from_text(str(message.content)))
                elif message.type == "image":
                    normalized.append(OmniInputSegment.from_image(message.content))
                else:
                    raise ValueError(
                        f"Unsupported omni input segment type: {message.type!r}"
                    )
            else:
                raise TypeError(f"omni input segment must be a dict: {message!r}")
        if not normalized:
            raise ValueError("omni interleaved request messages must not be empty")
        return cls(
            messages=tuple(normalized),
            sampling_params=sampling_params,
            metadata=metadata or {},
        )

    def to_legacy_segments(self) -> list[dict[str, Any]]:
        return [message.to_legacy_segment() for message in self.messages]


@dataclass(frozen=True, slots=True)
class OmniGeneratedImage:
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeneratedSegmentResult:
    type: Literal["image"]
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    commit_image: Any | None = None


@dataclass(frozen=True, slots=True)
class OmniOutputSegment:
    type: Literal["text", "image"]
    text: str | None = None
    image: OmniGeneratedImage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, *, metadata: dict[str, Any] | None = None
    ) -> "OmniOutputSegment":
        return cls(type="text", text=text, metadata=metadata or {})

    @classmethod
    def from_image(
        cls,
        image: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "OmniOutputSegment":
        return cls(
            type="image",
            image=OmniGeneratedImage(image=image, metadata=metadata or {}),
        )

    @classmethod
    def from_legacy_segment(cls, segment: dict[str, Any]) -> "OmniOutputSegment":
        segment_type = segment.get("type")
        metadata = dict(segment.get("metadata") or {})
        if segment_type == "text":
            text = segment.get("text", segment.get("content"))
            if text is None:
                raise ValueError("omni text output segment is missing text")
            return cls.from_text(str(text), metadata=metadata)
        if segment_type == "image":
            image = segment.get("image", segment.get("content"))
            if image is None:
                raise ValueError("omni image output segment is missing image")
            return cls.from_image(image, metadata=metadata)
        raise ValueError(f"Unsupported omni output segment type: {segment_type!r}")

    def to_legacy_segment(self) -> dict[str, Any]:
        if self.type == "text":
            if self.text is None:
                raise ValueError("omni text output segment is missing text")
            segment = {"type": "text", "text": self.text}
            if self.metadata:
                segment["metadata"] = dict(self.metadata)
            return segment
        if self.type == "image":
            if self.image is None:
                raise ValueError("omni image output segment is missing image")
            segment = {"type": "image", "image": self.image.image}
            metadata = dict(self.image.metadata)
            metadata.update(self.metadata)
            if metadata:
                segment["metadata"] = metadata
            return segment
        raise ValueError(f"Unsupported omni output segment type: {self.type!r}")


@dataclass(frozen=True, slots=True)
class OmniRuntimeStats:
    session_id: str
    state: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    srt_request_count: int = 0
    srt_executed_request_count: int = 0
    srt_sidecar_request_count: int = 0
    srt_ar_decode_request_count: int = 0

    @classmethod
    def from_debug_counters(cls, counters: dict[str, Any]) -> "OmniRuntimeStats":
        return cls(
            session_id=str(counters["session_id"]),
            state=str(counters["state"]),
            context_length=int(counters.get("context_length", 0)),
            context_version=int(counters.get("context_version", 0)),
            prefill_count=int(counters.get("prefill_count", 0)),
            append_image_count=int(counters.get("append_image_count", 0)),
            decode_count=int(counters.get("decode_count", 0)),
            srt_request_count=int(counters.get("srt_request_count", 0)),
            srt_executed_request_count=int(
                counters.get("srt_executed_request_count", 0)
            ),
            srt_sidecar_request_count=int(counters.get("srt_sidecar_request_count", 0)),
            srt_ar_decode_request_count=int(
                counters.get("srt_ar_decode_request_count", 0)
            ),
        )


@dataclass(frozen=True, slots=True)
class OmniInterleavedResponse:
    segments: tuple[OmniOutputSegment, ...]
    stats: OmniRuntimeStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_segments(
        cls,
        segments: list[dict[str, Any]],
        *,
        stats: OmniRuntimeStats | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "OmniInterleavedResponse":
        return cls(
            segments=tuple(
                OmniOutputSegment.from_legacy_segment(segment) for segment in segments
            ),
            stats=stats,
            metadata=metadata or {},
        )

    def to_legacy_segments(self) -> list[dict[str, Any]]:
        return [segment.to_legacy_segment() for segment in self.segments]

    def to_segments(self) -> list[dict[str, Any]]:
        return self.to_legacy_segments()

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.to_legacy_segments())

    def __getitem__(self, index):
        return self.to_legacy_segments()[index]
