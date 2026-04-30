# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

UGGenerationMode = Literal["t2i", "edit", "interleave", "vlm"]

_UG_MODE_ALIASES = {
    "text_to_image": "t2i",
    "txt2img": "t2i",
    "image_edit": "edit",
    "i2i": "edit",
    "img2img": "edit",
    "interleaved": "interleave",
    "vlm_chat": "vlm",
    "chat": "vlm",
}
_UG_GENERATION_MODES = {"t2i", "edit", "interleave", "vlm"}


def normalize_ug_generation_mode(
    mode: Any | None,
    *,
    default: UGGenerationMode = "interleave",
) -> UGGenerationMode:
    if mode is None:
        return default
    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = _UG_MODE_ALIASES.get(normalized, normalized)
    if normalized not in _UG_GENERATION_MODES:
        raise ValueError(
            "Unsupported UG generation mode "
            f"{mode!r}; expected one of {sorted(_UG_GENERATION_MODES)}"
        )
    return cast(UGGenerationMode, normalized)


@dataclass(frozen=True, slots=True)
class UGInputSegment:
    type: Literal["text", "image"]
    text: str | None = None
    image: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, *, metadata: dict[str, Any] | None = None
    ) -> "UGInputSegment":
        return cls(type="text", text=text, metadata=metadata or {})

    @classmethod
    def from_image(
        cls, image: Any, *, metadata: dict[str, Any] | None = None
    ) -> "UGInputSegment":
        return cls(type="image", image=image, metadata=metadata or {})

    @classmethod
    def from_legacy_segment(cls, segment: dict[str, Any]) -> "UGInputSegment":
        segment_type = segment.get("type")
        metadata = dict(segment.get("metadata") or {})
        if segment_type == "text":
            text = segment.get("text", segment.get("content"))
            if text is None:
                raise ValueError("UG text input segment is missing text")
            return cls.from_text(str(text), metadata=metadata)
        if segment_type == "image":
            image = segment.get("image", segment.get("content"))
            if image is None:
                raise ValueError("UG image input segment is missing image")
            return cls.from_image(image, metadata=metadata)
        raise ValueError(f"Unsupported UG input segment type: {segment_type!r}")

    def to_legacy_segment(self) -> dict[str, Any]:
        if self.type == "text":
            if self.text is None:
                raise ValueError("UG text input segment is missing text")
            segment = {"type": "text", "text": self.text}
        elif self.type == "image":
            if self.image is None:
                raise ValueError("UG image input segment is missing image")
            segment = {"type": "image", "image": self.image}
        else:
            raise ValueError(f"Unsupported UG input segment type: {self.type!r}")
        if self.metadata:
            segment["metadata"] = dict(self.metadata)
        return segment


@dataclass(frozen=True, slots=True)
class UGInterleavedRequest:
    messages: tuple[UGInputSegment, ...]
    sampling_params: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_segments(
        cls,
        messages: list[Any],
        *,
        sampling_params: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UGInterleavedRequest":
        normalized = []
        for message in messages:
            if isinstance(message, UGInputSegment):
                normalized.append(message)
            elif isinstance(message, dict):
                normalized.append(UGInputSegment.from_legacy_segment(message))
            elif hasattr(message, "type") and hasattr(message, "content"):
                if message.type == "text":
                    normalized.append(UGInputSegment.from_text(str(message.content)))
                elif message.type == "image":
                    normalized.append(UGInputSegment.from_image(message.content))
                else:
                    raise ValueError(
                        f"Unsupported UG input segment type: {message.type!r}"
                    )
            else:
                raise TypeError(f"UG input segment must be a dict: {message!r}")
        if not normalized:
            raise ValueError("UG interleaved request messages must not be empty")
        return cls(
            messages=tuple(normalized),
            sampling_params=sampling_params,
            metadata=metadata or {},
        )

    def to_legacy_segments(self) -> list[dict[str, Any]]:
        return [message.to_legacy_segment() for message in self.messages]


@dataclass(frozen=True, slots=True)
class UGGeneratedImage:
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGOutputSegment:
    type: Literal["text", "image"]
    text: str | None = None
    image: UGGeneratedImage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls, text: str, *, metadata: dict[str, Any] | None = None
    ) -> "UGOutputSegment":
        return cls(type="text", text=text, metadata=metadata or {})

    @classmethod
    def from_image(
        cls,
        image: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "UGOutputSegment":
        return cls(
            type="image",
            image=UGGeneratedImage(image=image, metadata=metadata or {}),
        )

    @classmethod
    def from_legacy_segment(cls, segment: dict[str, Any]) -> "UGOutputSegment":
        segment_type = segment.get("type")
        metadata = dict(segment.get("metadata") or {})
        if segment_type == "text":
            text = segment.get("text", segment.get("content"))
            if text is None:
                raise ValueError("UG text output segment is missing text")
            return cls.from_text(str(text), metadata=metadata)
        if segment_type == "image":
            image = segment.get("image", segment.get("content"))
            if image is None:
                raise ValueError("UG image output segment is missing image")
            return cls.from_image(image, metadata=metadata)
        raise ValueError(f"Unsupported UG output segment type: {segment_type!r}")

    def to_legacy_segment(self) -> dict[str, Any]:
        if self.type == "text":
            if self.text is None:
                raise ValueError("UG text output segment is missing text")
            segment = {"type": "text", "text": self.text}
            if self.metadata:
                segment["metadata"] = dict(self.metadata)
            return segment
        if self.type == "image":
            if self.image is None:
                raise ValueError("UG image output segment is missing image")
            segment = {"type": "image", "image": self.image.image}
            metadata = dict(self.image.metadata)
            metadata.update(self.metadata)
            if metadata:
                segment["metadata"] = metadata
            return segment
        raise ValueError(f"Unsupported UG output segment type: {self.type!r}")


@dataclass(frozen=True, slots=True)
class UGRuntimeStats:
    session_id: str
    state: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    velocity_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    srt_request_count: int = 0
    srt_executed_request_count: int = 0
    srt_sidecar_request_count: int = 0
    srt_u_decode_request_count: int = 0

    @classmethod
    def from_debug_counters(cls, counters: dict[str, Any]) -> "UGRuntimeStats":
        return cls(
            session_id=str(counters["session_id"]),
            state=str(counters["state"]),
            context_length=int(counters.get("context_length", 0)),
            context_version=int(counters.get("context_version", 0)),
            prefill_count=int(counters.get("prefill_count", 0)),
            velocity_count=int(counters.get("velocity_count", 0)),
            append_image_count=int(counters.get("append_image_count", 0)),
            decode_count=int(counters.get("decode_count", 0)),
            srt_request_count=int(counters.get("srt_request_count", 0)),
            srt_executed_request_count=int(
                counters.get("srt_executed_request_count", 0)
            ),
            srt_sidecar_request_count=int(counters.get("srt_sidecar_request_count", 0)),
            srt_u_decode_request_count=int(
                counters.get("srt_u_decode_request_count", 0)
            ),
        )


@dataclass(frozen=True, slots=True)
class UGInterleavedResponse:
    segments: tuple[UGOutputSegment, ...]
    stats: UGRuntimeStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_segments(
        cls,
        segments: list[dict[str, Any]],
        *,
        stats: UGRuntimeStats | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UGInterleavedResponse":
        return cls(
            segments=tuple(
                UGOutputSegment.from_legacy_segment(segment) for segment in segments
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
