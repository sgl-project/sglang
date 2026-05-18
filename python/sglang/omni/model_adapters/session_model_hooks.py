# SPDX-License-Identifier: Apache-2.0
"""Model hooks between model-specific code and the omni session runtime."""

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
class OmniSessionModelView:
    """Narrow SRT-owned session view exposed to omni session model hooks."""

    handle: OmniSessionHandle
    state: OmniSegmentState
    srt_request_count: int
    srt_last_request_id: str | None
    srt_last_origin_input_len: int
    srt_mm_offsets: tuple[tuple[int, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OmniSessionPrefillResult:
    added_tokens: int


@dataclass(frozen=True, slots=True)
class OmniSessionAppendImageResult:
    added_tokens: int


class OmniSessionModelHooks:
    """Base class for model-specific hooks called by `OmniSessionRuntime`.

    Hooks define model-specific AR session behavior:
      - message-to-SRT prepared inputs
      - prefill accounting
      - next-segment decode
      - VLM text decode
      - generated media commit accounting
      - ession close cleanup.

    Different from the session adapter which owns request-level mode selection, these hooks only own token grammar and state patches for one model.
    """

    def prepare_srt_ar_message_inputs(
        self,
        *,
        session: OmniSessionModelView,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        """prepare one SRT input chunk for an appended non-text message"""
        return None

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        session: OmniSessionModelView,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        """prepare SRT input chunks for a new user turn"""
        return None

    def on_prefill_finished(
        self,
        *,
        session: OmniSessionModelView,
        messages: list[OmniInterleavedMessage],
    ) -> OmniSessionPrefillResult:
        """account for model-specific token growth after SRT prefill"""
        return OmniSessionPrefillResult(
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
        session: OmniSessionModelView,
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
        session: OmniSessionModelView,
        image: Any | None,
    ) -> OmniSessionAppendImageResult:
        """account for model-specific token growth after generated media commit"""

        return OmniSessionAppendImageResult(
            added_tokens=max(
                0,
                int(session.srt_last_origin_input_len)
                - int(session.handle.context_length),
            )
        )

    def close_session(self, *, session_id: str) -> None:
        """release model-specific side state associated with a session id"""
        pass
