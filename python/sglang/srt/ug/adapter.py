# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from sglang.srt.ug.context import UGSRTRequestView, UGSessionHandle
from sglang.srt.ug.runtime import (
    UGDecodeResult,
    UGInterleavedMessage,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGLatentPrepareResult,
    UGSegmentState,
    UGSessionRecord,
    UGVelocityRequest,
)


@dataclass(frozen=True, slots=True)
class UGModelSessionView:
    """Narrow SRT-owned session view exposed to UG model adapters."""

    handle: UGSessionHandle
    state: UGSegmentState
    srt_request_count: int
    srt_last_request_id: str | None
    srt_last_origin_input_len: int
    srt_mm_offsets: tuple[tuple[int, int], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGModelPrefillResult:
    added_tokens: int


@dataclass(frozen=True, slots=True)
class UGModelAppendImageResult:
    added_tokens: int


class UGModelAdapterProtocol(Protocol):
    """Model-side UG entrypoints that BAGEL-like models need to implement."""

    def observe_srt_u_forward(
        self,
        *,
        session: UGModelSessionView,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None: ...

    def prefill_interleaved(
        self,
        *,
        session: UGModelSessionView,
        messages: list[UGInterleavedMessage],
    ) -> UGModelPrefillResult: ...

    def decode_next_segment(self, *, session: UGModelSessionView) -> UGDecodeResult: ...

    def predict_velocity_from_session(
        self,
        *,
        session: UGModelSessionView,
        request: UGVelocityRequest,
    ) -> torch.Tensor: ...

    def prepare_latents_from_session(
        self,
        *,
        session: UGModelSessionView,
        request: UGLatentPrepareRequest,
    ) -> UGLatentPrepareResult | None: ...

    def append_generated_image(
        self,
        *,
        session: UGModelSessionView,
        image: Any | None,
    ) -> UGModelAppendImageResult: ...

    def decode_latents_to_image(
        self,
        *,
        session: UGModelSessionView,
        request: UGLatentDecodeRequest,
    ) -> Any | None: ...

    def close_session(self, *, session_id: str) -> None: ...


class UGModelRunnerAdapter:
    """Adapts a model-side UG adapter to UGSessionRuntime's runner protocol."""

    def __init__(self, adapter: UGModelAdapterProtocol) -> None:
        self.adapter = adapter

    def prefill_interleaved(
        self, *, record: UGSessionRecord, messages: list[UGInterleavedMessage]
    ) -> int:
        result = self.adapter.prefill_interleaved(
            session=self._session_view(record),
            messages=messages,
        )
        return result.added_tokens

    def observe_srt_u_forward(
        self,
        *,
        record: UGSessionRecord,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        observe = getattr(self.adapter, "observe_srt_u_forward", None)
        if not callable(observe):
            return
        observe(
            session=self._session_view(record),
            request=request,
            messages=messages,
        )

    def decode_next_segment(self, *, record: UGSessionRecord) -> UGDecodeResult:
        return self.adapter.decode_next_segment(session=self._session_view(record))

    def predict_velocity_from_session(
        self, *, request: UGVelocityRequest, record: UGSessionRecord
    ) -> torch.Tensor:
        return self.adapter.predict_velocity_from_session(
            session=self._session_view(record),
            request=request,
        )

    def prepare_latents_from_session(
        self, *, request: UGLatentPrepareRequest, record: UGSessionRecord
    ) -> UGLatentPrepareResult | None:
        return self.adapter.prepare_latents_from_session(
            session=self._session_view(record),
            request=request,
        )

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int:
        result = self.adapter.append_generated_image(
            session=self._session_view(record),
            image=image,
        )
        return result.added_tokens

    def decode_latents_to_image(
        self, *, request: UGLatentDecodeRequest, record: UGSessionRecord
    ) -> Any | None:
        return self.adapter.decode_latents_to_image(
            session=self._session_view(record),
            request=request,
        )

    def close_session(self, *, session_id: str) -> None:
        self.adapter.close_session(session_id=session_id)

    @staticmethod
    def _session_view(record: UGSessionRecord) -> UGModelSessionView:
        return UGModelSessionView(
            handle=record.handle(),
            state=record.state,
            srt_request_count=record.srt_request_count,
            srt_last_request_id=record.srt_last_request_id,
            srt_last_origin_input_len=record.srt_last_origin_input_len,
            srt_mm_offsets=tuple(record.srt_mm_offsets),
            metadata={
                "srt_u_decode_request_count": record.srt_u_decode_request_count,
                "srt_last_u_decode_request_id": (
                    record.srt_last_u_decode_request_id
                ),
                "srt_last_u_decode_origin_input_len": (
                    record.srt_last_u_decode_origin_input_len
                ),
                "srt_last_u_decode_output_ids": tuple(
                    record.srt_last_u_decode_output_ids
                ),
            },
        )
