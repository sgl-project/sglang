# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

import torch

from sglang.srt.ug.context import (
    UGSRTKVTokenBinding,
    UGSRTRequestView,
    UGSessionHandle,
)


class _UGSimpleTokenizer:
    bos_token_id = 1


@dataclass(slots=True)
class _UGSessionMMItem:
    offsets: list[tuple[int, int]]
    feature: Any | None = field(default_factory=object)


@dataclass(slots=True)
class _UGSessionMMInputs:
    mm_items: list[_UGSessionMMItem]
    release_count: int = 0

    def merge(self, other: "_UGSessionMMInputs") -> None:
        self.mm_items.extend(other.mm_items)

    def release_features(self) -> None:
        self.release_count += 1
        for item in self.mm_items:
            item.feature = None


class UGSegmentState(str, Enum):
    U_PREFILL = "u_prefill"
    U_DECODE = "u_decode"
    G_DENOISE = "g_denoise"
    APPEND_IMAGE = "append_image"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class UGInterleavedMessage:
    type: Literal["text", "image"]
    content: Any


@dataclass(frozen=True, slots=True)
class UGVelocityRequest:
    session: UGSessionHandle
    latent_tokens: torch.Tensor
    timestep: torch.Tensor
    latent_position_ids: torch.Tensor
    sampling_params: Any


@dataclass(frozen=True, slots=True)
class UGVelocityResponse:
    session: UGSessionHandle
    velocity: torch.Tensor


@dataclass(frozen=True, slots=True)
class UGLatentPrepareRequest:
    session: UGSessionHandle
    sampling_params: Any
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class UGLatentPrepareResult:
    latent_tokens: torch.Tensor
    latent_position_ids: torch.Tensor
    latent_shape: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class UGLatentDecodeRequest:
    session: UGSessionHandle
    latent_tokens: torch.Tensor
    sampling_params: Any


@dataclass(frozen=True, slots=True)
class UGDecodeResult:
    type: Literal["text", "image_marker", "done"]
    text: str | None = None


@dataclass(slots=True)
class UGSessionRecord:
    session_id: str
    state: UGSegmentState
    anchor_request_id: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    velocity_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    srt_request_count: int = 0
    srt_last_request_id: str | None = None
    srt_last_origin_input_len: int = 0
    srt_last_origin_input_ids: list[int] = field(default_factory=list)
    srt_u_decode_request_count: int = 0
    srt_last_u_decode_request_id: str | None = None
    srt_last_u_decode_origin_input_len: int = 0
    srt_last_u_decode_output_ids: list[int] = field(default_factory=list)
    srt_mm_offsets: list[tuple[int, int]] = field(default_factory=list)
    srt_mm_inputs: _UGSessionMMInputs | None = None
    srt_executed_request_count: int = 0
    srt_last_executed_request_id: str | None = None
    srt_last_executed_state: str | None = None
    srt_model_runner_forward_request_ids: set[str] = field(default_factory=set)
    closed: bool = False

    def handle(self) -> UGSessionHandle:
        return UGSessionHandle(
            session_id=self.session_id,
            anchor_request_id=self.anchor_request_id,
            context_length=self.context_length,
            context_version=self.context_version,
        )


class UGModelRunnerProtocol(Protocol):
    def observe_srt_u_forward(
        self,
        *,
        record: UGSessionRecord,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None: ...

    def prefill_interleaved(
        self, *, record: UGSessionRecord, messages: list[UGInterleavedMessage]
    ) -> int: ...

    def decode_next_segment(self, *, record: UGSessionRecord) -> UGDecodeResult: ...

    def predict_velocity_from_session(
        self, *, request: UGVelocityRequest, record: UGSessionRecord
    ) -> torch.Tensor: ...

    def prepare_latents_from_session(
        self, *, request: UGLatentPrepareRequest, record: UGSessionRecord
    ) -> UGLatentPrepareResult | None: ...

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int: ...

    def decode_latents_to_image(
        self, *, request: UGLatentDecodeRequest, record: UGSessionRecord
    ) -> Any | None: ...

    def close_session(self, *, session_id: str) -> None: ...


class UGSRTRequestExecutorProtocol(Protocol):
    """Executes a UG SRT request after SessionController has materialized it."""

    def execute_ug_request(
        self,
        *,
        record: UGSessionRecord,
        req: Any,
        state: UGSegmentState,
    ) -> None: ...


class FakeUGModelRunner:
    """Deterministic UG model shell used to prove session/KV ownership plumbing."""

    def prefill_interleaved(
        self, *, record: UGSessionRecord, messages: list[UGInterleavedMessage]
    ) -> int:
        del record
        token_count = 0
        for message in messages:
            if message.type == "text":
                token_count += len(str(message.content).split())
            elif message.type == "image":
                token_count += 2
            else:
                raise ValueError(f"Unsupported UG message type: {message.type}")
        return token_count

    def decode_next_segment(self, *, record: UGSessionRecord) -> UGDecodeResult:
        if record.append_image_count == 0 and record.decode_count == 0:
            return UGDecodeResult(type="image_marker")
        if record.append_image_count > 0 and record.decode_count == 1:
            return UGDecodeResult(type="text", text="generated_text_after_image")
        return UGDecodeResult(type="done")

    def predict_velocity_from_session(
        self, *, request: UGVelocityRequest, record: UGSessionRecord
    ) -> torch.Tensor:
        scale = 1.0 + record.context_length * 0.01 + record.context_version * 0.001
        return request.latent_tokens + scale * request.timestep.reshape(-1, 1, 1).to(
            request.latent_tokens
        )

    def prepare_latents_from_session(
        self, *, request: UGLatentPrepareRequest, record: UGSessionRecord
    ) -> UGLatentPrepareResult | None:
        del request, record
        return None

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int:
        del record, image
        return 2

    def decode_latents_to_image(
        self, *, request: UGLatentDecodeRequest, record: UGSessionRecord
    ) -> Any | None:
        del request, record
        return None

    def close_session(self, *, session_id: str) -> None:
        del session_id

    def observe_srt_u_forward(
        self,
        *,
        record: UGSessionRecord,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        del record, request, messages


class UGSessionRuntime:
    """Lightweight UG state machine layered on top of SRT sessions."""

    def __init__(
        self,
        *,
        model_runner: UGModelRunnerProtocol | None = None,
        session_controller: Any | None = None,
        srt_request_executor: UGSRTRequestExecutorProtocol | None = None,
        capacity_of_str_len: int = 4096,
        tokenizer: Any | None = None,
        vocab_size: int = 32000,
        srt_u_decode_max_new_tokens: int = 0,
    ) -> None:
        self.model_runner = model_runner or FakeUGModelRunner()
        self.session_controller = session_controller
        self.srt_request_executor = srt_request_executor
        self.capacity_of_str_len = capacity_of_str_len
        self.tokenizer = tokenizer or _UGSimpleTokenizer()
        self.vocab_size = vocab_size
        self.srt_u_decode_max_new_tokens = srt_u_decode_max_new_tokens
        self._records: dict[str, UGSessionRecord] = {}
        register_observer = getattr(
            self.srt_request_executor,
            "set_ug_u_forward_observer",
            None,
        )
        if callable(register_observer):
            register_observer(self._observe_srt_u_forward_from_model_runner)

    @staticmethod
    def normalize_messages(
        *, prompt: str | list[str] | None = None, image: Any | None = None
    ) -> list[UGInterleavedMessage]:
        messages: list[UGInterleavedMessage] = []
        if image is not None:
            messages.append(UGInterleavedMessage(type="image", content=image))
        if prompt is not None:
            prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt
            messages.append(UGInterleavedMessage(type="text", content=prompt_text))
        return messages

    def prefill_interleaved(
        self,
        messages: list[UGInterleavedMessage],
        *,
        session_id: str | None = None,
    ) -> UGSessionHandle:
        if not messages:
            raise ValueError("UG prefill requires at least one text or image message")
        session_id = session_id or uuid.uuid4().hex
        record = self._records.get(session_id)
        if record is None:
            self._ensure_srt_session(session_id)
            record = UGSessionRecord(
                session_id=session_id,
                state=UGSegmentState.U_PREFILL,
                anchor_request_id=f"{session_id}:u0",
            )
            self._records[session_id] = record
        elif record.closed:
            raise ValueError(f"UG session {session_id} is closed")
        elif record.state not in {UGSegmentState.U_DECODE, UGSegmentState.DONE}:
            raise ValueError(
                f"Cannot prefill UG session {session_id} from state {record.state}"
            )
        else:
            record.state = UGSegmentState.U_PREFILL

        next_context_version = record.context_version + 1
        next_anchor_request_id = f"{session_id}:u{next_context_version}"
        record.anchor_request_id = next_anchor_request_id
        self._append_srt_session_request(
            record,
            messages,
            request_id=next_anchor_request_id,
        )
        added_tokens = self.model_runner.prefill_interleaved(
            record=record, messages=messages
        )
        record.context_length += added_tokens
        record.context_version = next_context_version
        record.prefill_count += 1
        record.state = UGSegmentState.U_DECODE
        return record.handle()

    def begin_g_denoise(self, handle: UGSessionHandle) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot enter G denoise from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        record.state = UGSegmentState.G_DENOISE
        return record.handle()

    def decode_next_segment(self, handle: UGSessionHandle) -> UGDecodeResult:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot decode U segment from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        if self.srt_u_decode_max_new_tokens > 0:
            self._append_srt_u_decode_request(record)
        result = self.model_runner.decode_next_segment(record=record)
        record.decode_count += 1
        if result.type == "image_marker":
            record.state = UGSegmentState.G_DENOISE
        elif result.type == "done":
            record.state = UGSegmentState.DONE
        else:
            record.state = UGSegmentState.U_DECODE
        return result

    def predict_velocity(self, request: UGVelocityRequest) -> UGVelocityResponse:
        record = self._record_for(request.session)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot predict UG velocity from state {record.state} "
                f"for UG session {request.session.session_id}"
            )
        velocity = self.model_runner.predict_velocity_from_session(
            request=request, record=record
        )
        record.velocity_count += 1
        return UGVelocityResponse(session=record.handle(), velocity=velocity)

    def prepare_latents(
        self, request: UGLatentPrepareRequest
    ) -> UGLatentPrepareResult | None:
        record = self._record_for(request.session)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot prepare UG latents from state {record.state} "
                f"for UG session {request.session.session_id}"
            )
        return self.model_runner.prepare_latents_from_session(
            request=request,
            record=record,
        )

    def decode_latents_to_image(self, request: UGLatentDecodeRequest) -> Any | None:
        record = self._record_for(request.session)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot decode UG latents from state {record.state} "
                f"for UG session {request.session.session_id}"
            )
        return self.model_runner.decode_latents_to_image(
            request=request,
            record=record,
        )

    def append_generated_image(
        self, handle: UGSessionHandle, image: Any | None
    ) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.G_DENOISE:
            raise ValueError(
                f"Cannot append generated image from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        record.state = UGSegmentState.APPEND_IMAGE
        next_context_version = record.context_version + 1
        next_anchor_request_id = f"{record.session_id}:u{next_context_version}"
        record.anchor_request_id = next_anchor_request_id
        self._append_srt_session_request(
            record,
            [UGInterleavedMessage(type="image", content=image)],
            request_id=next_anchor_request_id,
        )
        added_tokens = self.model_runner.append_generated_image(
            record=record, image=image
        )
        record.context_length += added_tokens
        record.context_version = next_context_version
        record.append_image_count += 1
        record.state = UGSegmentState.U_DECODE
        return record.handle()

    def close_session(self, handle_or_session_id: UGSessionHandle | str) -> None:
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, UGSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is not None:
            record.closed = True
            record.state = UGSegmentState.DONE
        self.model_runner.close_session(session_id=session_id)
        self._close_srt_session(session_id)

    def get_state(self, handle_or_session_id: UGSessionHandle | str) -> UGSegmentState:
        return self._record_for(handle_or_session_id).state

    def get_debug_counters(self, handle_or_session_id: UGSessionHandle | str) -> dict:
        record = self._record_for(handle_or_session_id, allow_closed=True)
        return {
            "session_id": record.session_id,
            "state": record.state.value,
            "closed": record.closed,
            "context_length": record.context_length,
            "context_version": record.context_version,
            "prefill_count": record.prefill_count,
            "velocity_count": record.velocity_count,
            "append_image_count": record.append_image_count,
            "decode_count": record.decode_count,
            "srt_request_count": record.srt_request_count,
            "srt_last_request_id": record.srt_last_request_id,
            "srt_last_origin_input_len": record.srt_last_origin_input_len,
            "srt_last_origin_input_ids": record.srt_last_origin_input_ids,
            "srt_u_decode_request_count": record.srt_u_decode_request_count,
            "srt_last_u_decode_request_id": record.srt_last_u_decode_request_id,
            "srt_last_u_decode_origin_input_len": (
                record.srt_last_u_decode_origin_input_len
            ),
            "srt_last_u_decode_output_ids": record.srt_last_u_decode_output_ids,
            "srt_mm_offsets": record.srt_mm_offsets,
            "srt_executed_request_count": record.srt_executed_request_count,
            "srt_last_executed_request_id": record.srt_last_executed_request_id,
            "srt_last_executed_state": record.srt_last_executed_state,
            "srt_model_runner_forward_request_ids": sorted(
                record.srt_model_runner_forward_request_ids
            ),
            "srt_mm_features_released": self._srt_mm_features_released(record),
        }

    def _record_for(
        self,
        handle_or_session_id: UGSessionHandle | str,
        *,
        allow_closed: bool = False,
    ) -> UGSessionRecord:
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, UGSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is None or (record.closed and not allow_closed):
            raise ValueError(f"Unknown or closed UG session: {session_id}")
        if isinstance(handle_or_session_id, UGSessionHandle):
            if handle_or_session_id.context_version != record.context_version:
                raise ValueError(
                    "Stale UG session handle: "
                    f"{handle_or_session_id.context_version} != "
                    f"{record.context_version}"
                )
        return record

    def _append_srt_session_request(
        self,
        record: UGSessionRecord,
        messages: list[UGInterleavedMessage],
        *,
        request_id: str,
    ) -> None:
        if self.session_controller is None or not hasattr(
            self.session_controller, "get"
        ):
            return
        from sglang.srt.session.session_controller import SessionController

        input_ids, input_text, mm_inputs = self._tokenize_interleaved_messages(messages)
        req, recv_req = self._create_srt_session_req(
            record,
            request_id=request_id,
            input_ids=input_ids,
            input_text=input_text,
            mm_inputs=mm_inputs,
            max_new_tokens=0,
        )

        if mm_inputs is not None:
            SessionController.adjust_mm_offsets(recv_req, req, mm_inputs)
            req.extend_image_inputs(mm_inputs)

        self._record_srt_req(record, req, request_id=request_id)
        self._attach_srt_u_forward_metadata(
            record,
            req,
            state=record.state,
            input_text=input_text,
            messages=messages,
        )
        self._execute_srt_req(record, req, state=record.state)
        self._notify_srt_u_forward(
            record,
            req,
            state=record.state,
            input_text=input_text,
            messages=messages,
        )

    def _append_srt_u_decode_request(self, record: UGSessionRecord) -> None:
        request_id = f"{record.session_id}:d{record.decode_count + 1}"
        req, _ = self._create_srt_session_req(
            record,
            request_id=request_id,
            input_ids=[],
            input_text="",
            mm_inputs=None,
            max_new_tokens=self.srt_u_decode_max_new_tokens,
        )
        self._record_srt_req(record, req, request_id=request_id)
        self._attach_srt_u_forward_metadata(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
        )
        self._execute_srt_req(record, req, state=UGSegmentState.U_DECODE)
        record.srt_u_decode_request_count += 1
        record.srt_last_u_decode_request_id = request_id
        record.srt_last_u_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_u_decode_output_ids = list(
            req.output_ids[: req.sampling_params.max_new_tokens]
        )
        record.context_length += len(record.srt_last_u_decode_output_ids)
        self._notify_srt_u_forward(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
        )

    def _create_srt_session_req(
        self,
        record: UGSessionRecord,
        *,
        request_id: str,
        input_ids: list[int],
        input_text: str,
        mm_inputs: _UGSessionMMInputs | None,
        max_new_tokens: int,
    ):
        from sglang.srt.managers.io_struct import (
            SessionParams,
            TokenizedGenerateReqInput,
        )
        from sglang.srt.sampling.sampling_params import SamplingParams

        session = self.session_controller.get(record.session_id)
        if session is None:
            raise RuntimeError(
                f"SRT session {record.session_id} is not open for UG request"
            )
        recv_req = TokenizedGenerateReqInput(
            rid=request_id,
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(
                id=record.session_id,
                rid=record.srt_last_request_id,
                replace=False,
                drop_previous_output=False,
            ),
        )
        req = session.create_req(
            recv_req,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
        )
        if req.to_finish is not None:
            raise RuntimeError(
                f"Failed to create SRT UG session request {request_id}: "
                f"{req.to_finish.to_json()}"
            )
        return req, recv_req

    @staticmethod
    def _attach_srt_u_forward_metadata(
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
        input_text: str,
        messages: list[UGInterleavedMessage],
    ) -> None:
        req.ug_u_forward_metadata = {
            "session": record.handle(),
            "state": state.value,
            "request_id": req.rid,
            "origin_input_len": len(req.origin_input_ids),
            "origin_input_ids": tuple(req.origin_input_ids),
            "output_ids": tuple(req.output_ids[: req.sampling_params.max_new_tokens]),
            "max_new_tokens": req.sampling_params.max_new_tokens,
            "input_text": input_text,
            "mm_offsets": tuple(
                UGSessionRuntime._collect_mm_offsets(req.multimodal_inputs)
            ),
            "messages": tuple(messages),
        }

    def _execute_srt_req(
        self,
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
    ) -> None:
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        if self.srt_request_executor is not None:
            self.srt_request_executor.execute_ug_request(
                record=record,
                req=req,
                state=state,
            )
            record.srt_executed_request_count += 1
            record.srt_last_executed_request_id = req.rid
            record.srt_last_executed_state = state.value

        if getattr(self.srt_request_executor, "finish_request_after_execute", True):
            req.finished_reason = FINISH_LENGTH(len(req.output_ids))

    @staticmethod
    def _record_srt_req(
        record: UGSessionRecord,
        req: Any,
        *,
        request_id: str,
    ) -> None:
        record.srt_request_count += 1
        record.srt_last_request_id = request_id
        record.srt_last_origin_input_len = len(req.origin_input_ids)
        record.srt_last_origin_input_ids = list(req.origin_input_ids)
        record.srt_mm_inputs = req.multimodal_inputs
        record.srt_mm_offsets = UGSessionRuntime._collect_mm_offsets(
            req.multimodal_inputs
        )

    def _notify_srt_u_forward(
        self,
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
        input_text: str,
        messages: list[UGInterleavedMessage],
    ) -> None:
        observe = getattr(self.model_runner, "observe_srt_u_forward", None)
        if not callable(observe):
            return
        if (
            req.rid in record.srt_model_runner_forward_request_ids
            and state != UGSegmentState.U_DECODE
        ):
            return
        output_ids = tuple(req.output_ids[: req.sampling_params.max_new_tokens])
        observe(
            record=record,
            request=UGSRTRequestView(
                session=record.handle(),
                state=state.value,
                request_id=req.rid,
                origin_input_len=len(req.origin_input_ids),
                origin_input_ids=tuple(req.origin_input_ids),
                output_ids=output_ids,
                max_new_tokens=req.sampling_params.max_new_tokens,
                input_text=input_text,
                mm_offsets=tuple(self._collect_mm_offsets(req.multimodal_inputs)),
                metadata=self._srt_request_view_metadata(record, req, state=state),
            ),
            messages=messages,
        )

    def _observe_srt_u_forward_from_model_runner(
        self,
        *,
        request: UGSRTRequestView,
        messages: list[UGInterleavedMessage],
    ) -> None:
        record = self._records.get(request.session.session_id)
        if record is None or record.closed:
            return
        observe = getattr(self.model_runner, "observe_srt_u_forward", None)
        if not callable(observe):
            return
        observe(record=record, request=request, messages=messages)
        record.srt_model_runner_forward_request_ids.add(request.request_id)

    def _srt_request_view_metadata(
        self,
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
    ) -> dict[str, Any]:
        token_binding = self._srt_kv_token_binding(record, req, state=state)
        if token_binding is None:
            return {}
        return {"srt_kv_token_binding": token_binding}

    def _srt_kv_token_binding(
        self,
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
    ) -> UGSRTKVTokenBinding | None:
        provider = getattr(
            self.srt_request_executor, "get_ug_request_token_binding", None
        )
        if not callable(provider):
            return None

        signature = inspect.signature(provider)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs = {
            "record": record,
            "state": state,
        }
        if "req" in parameters or accepts_kwargs:
            kwargs["req"] = req
        elif "request" in parameters:
            kwargs["request"] = req
        binding = provider(**kwargs)
        if binding is None:
            return None
        if not isinstance(binding, UGSRTKVTokenBinding):
            raise TypeError(
                "UG SRT request token binding provider must return "
                f"UGSRTKVTokenBinding, got {type(binding).__name__}"
            )
        return binding

    def _tokenize_interleaved_messages(
        self, messages: list[UGInterleavedMessage]
    ) -> tuple[list[int], str, _UGSessionMMInputs | None]:
        input_ids = [self._bos_token_id()]
        text_parts: list[str] = []
        mm_items: list[_UGSessionMMItem] = []
        for message in messages:
            if message.type == "text":
                text = str(message.content)
                text_parts.append(text)
                input_ids.extend(self._text_token_ids(text))
            elif message.type == "image":
                start = len(input_ids)
                input_ids.extend([self.vocab_size + 1, self.vocab_size + 2])
                mm_items.append(_UGSessionMMItem(offsets=[(start, len(input_ids))]))
                text_parts.append("<image>")
            else:
                raise ValueError(f"Unsupported UG message type: {message.type}")
        mm_inputs = _UGSessionMMInputs(mm_items) if mm_items else None
        return input_ids, " ".join(text_parts), mm_inputs

    def _bos_token_id(self) -> int:
        token_id = getattr(self.tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(self.tokenizer, "eos_token_id", None)
        return int(token_id) if token_id is not None else 1

    def _text_token_ids(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                return list(encode(text, add_special_tokens=False))
            except TypeError:
                return list(encode(text))
        return self._fake_text_token_ids(text)

    @staticmethod
    def _fake_text_token_ids(text: str) -> list[int]:
        return [100 + (sum(word.encode("utf-8")) % 1000) for word in text.split()]

    @staticmethod
    def _collect_mm_offsets(mm_inputs: Any | None) -> list[tuple[int, int]]:
        if mm_inputs is None:
            return []
        offsets: list[tuple[int, int]] = []
        for item in getattr(mm_inputs, "mm_items", []):
            offsets.extend(getattr(item, "offsets", []) or [])
        return offsets

    @staticmethod
    def _srt_mm_features_released(record: UGSessionRecord) -> bool:
        if record.srt_mm_inputs is None:
            return False
        for item in record.srt_mm_inputs.mm_items:
            if item.feature is not None:
                return False
        return True

    def _ensure_srt_session(self, session_id: str) -> None:
        if self.session_controller is None or session_id in self.session_controller:
            return
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        output = self.session_controller.open(
            OpenSessionReqInput(
                session_id=session_id,
                capacity_of_str_len=self.capacity_of_str_len,
            )
        )
        if not getattr(output, "success", False):
            raise RuntimeError(f"Failed to open SRT session for UG: {session_id}")

    def _close_srt_session(self, session_id: str) -> None:
        if self.session_controller is None or session_id not in self.session_controller:
            return
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        self.session_controller.close(CloseSessionReqInput(session_id=session_id))
