# SPDX-License-Identifier: Apache-2.0
"""Adapter layer between model-specific AR/generation code and omni session runtime."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from sglang.srt.omni_session.context import OmniSessionHandle, OmniSRTRequestView
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniSegmentState,
    OmniSessionRecord,
    OmniSRTPreparedInput,
    OmniVLMTextGenerationResult,
)


@dataclass(frozen=True, slots=True)
class OmniModelSessionView:
    """Narrow SRT-owned session view exposed to omni model adapters."""

    handle: OmniSessionHandle
    state: OmniSegmentState
    srt_request_count: int
    srt_last_request_id: str | None
    srt_last_origin_input_len: int
    srt_mm_offsets: tuple[tuple[int, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OmniModelPrefillResult:
    added_tokens: int


@dataclass(frozen=True, slots=True)
class OmniModelAppendImageResult:
    added_tokens: int


class OmniModelAdapterProtocol(Protocol):
    """Model-side omni entrypoints implemented by each unified-generation backend."""

    def prepare_srt_ar_message_inputs(
        self,
        *,
        session: OmniModelSessionView,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None: ...

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None: ...

    def observe_srt_ar_forward(
        self,
        *,
        session: OmniModelSessionView,
        request: OmniSRTRequestView,
        messages: list[OmniInterleavedMessage],
    ) -> None: ...

    def prefill_interleaved(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
    ) -> OmniModelPrefillResult: ...

    def decode_next_segment(self, *, session: OmniModelSessionView) -> OmniDecodeResult: ...

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult: ...

    def append_generated_image(
        self,
        *,
        session: OmniModelSessionView,
        image: Any | None,
    ) -> OmniModelAppendImageResult: ...

    def close_session(self, *, session_id: str) -> None: ...


class OmniModelRunnerAdapter:
    """Adapts a model-side omni adapter to OmniSessionRuntime's runner protocol."""

    def __init__(self, adapter: OmniModelAdapterProtocol) -> None:
        self.adapter = adapter

    def prepare_srt_ar_message_inputs(
        self,
        *,
        record: OmniSessionRecord,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        prepare = getattr(self.adapter, "prepare_srt_ar_message_inputs", None)
        if not callable(prepare):
            return None
        return prepare(
            session=self._session_view(record),
            message=message,
            state=state,
        )

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        record: OmniSessionRecord,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        prepare = getattr(self.adapter, "prepare_srt_ar_interleaved_inputs", None)
        if not callable(prepare):
            return None
        return prepare(
            session=self._session_view(record),
            messages=messages,
            state=state,
        )

    def prefill_interleaved(
        self, *, record: OmniSessionRecord, messages: list[OmniInterleavedMessage]
    ) -> int:
        result = self.adapter.prefill_interleaved(
            session=self._session_view(record),
            messages=messages,
        )
        return result.added_tokens

    def observe_srt_ar_forward(
        self,
        *,
        record: OmniSessionRecord,
        request: OmniSRTRequestView,
        messages: list[OmniInterleavedMessage],
    ) -> None:
        observe = getattr(self.adapter, "observe_srt_ar_forward", None)
        if not callable(observe):
            return
        observe(
            session=self._session_view(record),
            request=request,
            messages=messages,
        )

    def decode_next_segment(self, *, record: OmniSessionRecord) -> OmniDecodeResult:
        return self.adapter.decode_next_segment(session=self._session_view(record))

    def decode_next_segment_from_runtime(
        self, *, runtime: Any, record: OmniSessionRecord
    ) -> OmniDecodeResult:
        decode = getattr(self.adapter, "decode_next_segment_from_runtime", None)
        if not callable(decode):
            if runtime.srt_ar_decode_max_new_tokens > 0:
                runtime._append_srt_ar_decode_request(record, greedy=True)
            return self.decode_next_segment(record=record)
        return decode(runtime=runtime, session=record.handle())

    def decode_vlm_text(
        self,
        *,
        runtime: Any,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        decode = getattr(self.adapter, "decode_vlm_text", None)
        if not callable(decode):
            raise NotImplementedError(
                f"{self.adapter.__class__.__name__} does not support VLM text decode"
            )
        return decode(
            runtime=runtime,
            session=session,
            max_new_tokens=max_new_tokens,
        )

    def append_generated_image(
        self, *, record: OmniSessionRecord, image: Any | None
    ) -> int:
        result = self.adapter.append_generated_image(
            session=self._session_view(record),
            image=image,
        )
        return result.added_tokens

    def close_session(self, *, session_id: str) -> None:
        self.adapter.close_session(session_id=session_id)

    @staticmethod
    def _session_view(record: OmniSessionRecord) -> OmniModelSessionView:
        return OmniModelSessionView(
            handle=record.handle(),
            state=record.state,
            srt_request_count=record.srt_request_count,
            srt_last_request_id=record.srt_last_request_id,
            srt_last_origin_input_len=record.srt_last_origin_input_len,
            srt_mm_offsets=tuple(record.srt_mm_offsets),
            metadata={
                "srt_ar_decode_request_count": record.srt_ar_decode_request_count,
                "srt_last_ar_decode_request_id": (record.srt_last_ar_decode_request_id),
                "srt_last_ar_decode_origin_input_len": (
                    record.srt_last_ar_decode_origin_input_len
                ),
                "srt_last_ar_decode_output_ids": tuple(
                    record.srt_last_ar_decode_output_ids
                ),
                "srt_last_ar_decode_text": record.srt_last_ar_decode_text,
                "omni_model_state": deepcopy(record.omni_model_state),
            },
        )
