# SPDX-License-Identifier: Apache-2.0
"""Policy layer between model-specific AR/generation code and omni session runtime."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.omni_session.runtime_protocol import OmniSessionHandle
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniModelPolicy,
    OmniSegmentState,
    OmniSessionRecord,
    OmniSessionRuntime,
    OmniSRTPreparedInput,
    OmniVLMTextGenerationResult,
)


@dataclass(frozen=True, slots=True)
class OmniModelSessionView:
    """Narrow SRT-owned session view exposed to omni model policies."""

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


class OmniSessionModelPolicy:
    """Base class for model-specific omni session policies.

    Subclasses implement the model-native prompt formatting and decode rules.
    The policy runner below turns these narrow session-view hooks into the
    record-oriented surface used by `OmniSessionRuntime`.
    """

    def prepare_srt_ar_message_inputs(
        self,
        *,
        session: OmniModelSessionView,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        return None

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        return None

    def on_prefill_finished(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
    ) -> OmniModelPrefillResult:
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
        """decode the next interleaved boundary: text, image marker, or done"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support segment decode"
        )

    def decode_next_segment_with_runtime(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniModelSessionView,
    ) -> OmniDecodeResult | None:
        """override boundary decode when the model needs live SRT runtime access"""
        return None

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        """decode a plain VLM text answer without entering generation boundaries"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support VLM text decode"
        )

    def append_generated_image(
        self,
        *,
        session: OmniModelSessionView,
        image: Any | None,
    ) -> OmniModelAppendImageResult:
        return OmniModelAppendImageResult(
            added_tokens=max(
                0,
                int(session.srt_last_origin_input_len)
                - int(session.handle.context_length),
            )
        )

    def close_session(self, *, session_id: str) -> None:
        pass


class OmniModelPolicyRunner(OmniModelPolicy):
    """Wrap a model-side omni policy for OmniSessionRuntime's record hooks."""

    def __init__(self, policy: OmniSessionModelPolicy) -> None:
        self.policy = policy

    def prepare_srt_ar_message_inputs(
        self,
        *,
        record: OmniSessionRecord,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        return self.policy.prepare_srt_ar_message_inputs(
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
        return self.policy.prepare_srt_ar_interleaved_inputs(
            session=self._session_view(record),
            messages=messages,
            state=state,
        )

    def on_prefill_finished(
        self, *, record: OmniSessionRecord, messages: list[OmniInterleavedMessage]
    ) -> int:
        result = self.policy.on_prefill_finished(
            session=self._session_view(record),
            messages=messages,
        )
        return result.added_tokens

    def decode_next_segment(self, *, record: OmniSessionRecord) -> OmniDecodeResult:
        return self.policy.decode_next_segment(session=self._session_view(record))

    def decode_next_segment_with_runtime(
        self, *, runtime: OmniSessionRuntime, record: OmniSessionRecord
    ) -> OmniDecodeResult:
        """forward runtime-aware boundary decode to the model policy"""
        result = self.policy.decode_next_segment_with_runtime(
            runtime=runtime,
            session=self._session_view(record),
        )
        if result is not None:
            return result
        return super().decode_next_segment_with_runtime(runtime=runtime, record=record)

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        """forward plain VLM answer decode to the model policy"""
        return self.policy.decode_vlm_text(
            runtime=runtime,
            session=session,
            max_new_tokens=max_new_tokens,
        )

    def append_generated_image(
        self, *, record: OmniSessionRecord, image: Any | None
    ) -> int:
        result = self.policy.append_generated_image(
            session=self._session_view(record),
            image=image,
        )
        return result.added_tokens

    def close_session(self, *, session_id: str) -> None:
        self.policy.close_session(session_id=session_id)

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
