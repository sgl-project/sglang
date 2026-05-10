# SPDX-License-Identifier: Apache-2.0
"""Adapter layer between model-specific AR/generation code and omni session runtime."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.omni_session.runtime_protocol import OmniSessionHandle, OmniSRTRequestView
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniModelRunner,
    OmniSegmentState,
    OmniSessionRecord,
    OmniSessionRuntime,
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


class OmniModelAdapter:
    """Base class for model-specific omni session adapters.

    Subclasses implement the model-native prompt formatting and decode policy.
    The runner adapter below turns these narrow session-view hooks into the
    record-oriented surface used by `OmniSessionRuntime`.
    """

    def prepare_srt_ar_message_inputs(
        self,
        *,
        session: OmniModelSessionView,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        del session, message, state
        return None

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        del session, messages, state
        return None

    def observe_srt_ar_forward(
        self,
        *,
        session: OmniModelSessionView,
        request: OmniSRTRequestView,
        messages: list[OmniInterleavedMessage],
    ) -> None:
        del session, request, messages

    def prefill_interleaved(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
    ) -> OmniModelPrefillResult:
        del messages
        return OmniModelPrefillResult(
            added_tokens=max(
                0,
                int(session.srt_last_origin_input_len)
                - int(session.handle.context_length),
            )
        )

    def decode_next_segment(
        self, *, session: OmniModelSessionView
    ) -> OmniDecodeResult:
        del session
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support segment decode"
        )

    def decode_next_segment_from_runtime(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniModelSessionView,
    ) -> OmniDecodeResult | None:
        del runtime, session
        return None

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        del runtime, session, max_new_tokens
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support VLM text decode"
        )

    def append_generated_image(
        self,
        *,
        session: OmniModelSessionView,
        image: Any | None,
    ) -> OmniModelAppendImageResult:
        del image
        return OmniModelAppendImageResult(
            added_tokens=max(
                0,
                int(session.srt_last_origin_input_len)
                - int(session.handle.context_length),
            )
        )

    def close_session(self, *, session_id: str) -> None:
        del session_id


class OmniModelRunnerAdapter(OmniModelRunner):
    """Adapts a model-side omni adapter to OmniSessionRuntime's runner hooks."""

    def __init__(self, adapter: OmniModelAdapter) -> None:
        self.adapter = adapter

    def prepare_srt_ar_message_inputs(
        self,
        *,
        record: OmniSessionRecord,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        return self.adapter.prepare_srt_ar_message_inputs(
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
        return self.adapter.prepare_srt_ar_interleaved_inputs(
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
        self.adapter.observe_srt_ar_forward(
            session=self._session_view(record),
            request=request,
            messages=messages,
        )

    def decode_next_segment(self, *, record: OmniSessionRecord) -> OmniDecodeResult:
        return self.adapter.decode_next_segment(session=self._session_view(record))

    def decode_next_segment_from_runtime(
        self, *, runtime: OmniSessionRuntime, record: OmniSessionRecord
    ) -> OmniDecodeResult:
        result = self.adapter.decode_next_segment_from_runtime(
            runtime=runtime,
            session=self._session_view(record),
        )
        if result is not None:
            return result
        return super().decode_next_segment_from_runtime(runtime=runtime, record=record)

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        return self.adapter.decode_vlm_text(
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
