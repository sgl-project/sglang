# SPDX-License-Identifier: Apache-2.0
"""Streaming event sink used by omni HTTP adapters.

The sink owns only wire-level event ordering. Coordinator and model policies
still own generation semantics; they call this object when a text/image segment
crosses an observable boundary.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.omni.entrypoints.serialization import serialize_image


class OmniStreamSink:
    """Build ordered SSE-compatible events for one omni request."""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit
        self._next_segment_index = 0
        self._active_text_segment_id: str | None = None
        self._active_text_metadata: dict[str, Any] | None = None

    def text_delta(
        self,
        delta: str,
        *,
        token_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not delta:
            return
        if self._active_text_segment_id is None:
            self._active_text_segment_id = self._new_segment_id()
            self._active_text_metadata = dict(metadata or {})
        elif metadata:
            self._active_text_metadata = dict(metadata)
        event: dict[str, Any] = {
            "type": "text_delta",
            "segment_id": self._active_text_segment_id,
            "delta": delta,
        }
        if token_id is not None:
            event["token_id"] = int(token_id)
        if self._active_text_metadata:
            event["metadata"] = dict(self._active_text_metadata)
        self._emit(event)

    def text_segment(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not text:
            return
        self.text_delta(text, metadata=metadata)
        self.finish_text()

    def finish_text(self) -> None:
        if self._active_text_segment_id is None:
            return
        event: dict[str, Any] = {
            "type": "text_end",
            "segment_id": self._active_text_segment_id,
        }
        if self._active_text_metadata:
            event["metadata"] = dict(self._active_text_metadata)
        self._emit(event)
        self._active_text_segment_id = None
        self._active_text_metadata = None

    def begin_image(self) -> str:
        self.finish_text()
        segment_id = self._new_segment_id()
        self._emit(
            {
                "type": "image_start",
                "segment_id": segment_id,
            }
        )
        return segment_id

    def image(
        self,
        *,
        segment_id: str,
        image: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "type": "image",
            "segment_id": segment_id,
            "image": serialize_image(image),
        }
        if metadata:
            event["metadata"] = dict(metadata)
        self._emit(event)

    def done(self, payload: dict[str, Any]) -> None:
        self.finish_text()
        event: dict[str, Any] = {"type": "done"}
        for key in ("session", "context", "stats", "metadata"):
            if key in payload:
                event[key] = payload[key]
        self._emit(event)

    def error(self, *, message: str, status_code: int = 500) -> None:
        self.finish_text()
        self._emit(
            {
                "type": "error",
                "error": {"message": message},
                "status_code": int(status_code),
            }
        )

    def _new_segment_id(self) -> str:
        segment_id = f"seg-{self._next_segment_index}"
        self._next_segment_index += 1
        return segment_id
