# SPDX-License-Identifier: Apache-2.0
"""Policy layer between model-specific AR/generation code and omni session runtime."""

from dataclasses import dataclass, field
from typing import Any

from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniSegmentState,
    OmniSessionRuntime,
    OmniSRTPreparedInput,
    OmniVLMTextGenerationResult,
)
from sglang.srt.omni_session.runtime_types import OmniSessionHandle


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


class OmniModelPolicy:
    """Base class for model-specific omni session policies.

    Subclasses implement the model-native prompt formatting and decode rules.
    Most hooks receive only a narrow session view; runtime-aware hooks are
    reserved for model rules that must run SRT decode or update condition paths.
    `OmniSessionRuntime` owns mutable records and passes this view to keep model
    code away from SRT session internals.
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

    def decode_next_segment_with_runtime(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniModelSessionView,
        stream_sink: Any | None = None,
    ) -> OmniDecodeResult:
        """decode with live runtime access until the next text/media boundary"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support segment decode"
        )

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
        stream_sink: Any | None = None,
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
        """call back for committing a generated media after SRT has materialized it."""

        return OmniModelAppendImageResult(
            added_tokens=max(
                0,
                int(session.srt_last_origin_input_len)
                - int(session.handle.context_length),
            )
        )

    def close_session(self, *, session_id: str) -> None:
        pass
