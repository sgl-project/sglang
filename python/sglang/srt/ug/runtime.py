# SPDX-License-Identifier: Apache-2.0

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

import torch

from sglang.srt.ug.context import (
    UGSessionHandle,
    UGSRTKVTokenBinding,
    UGSRTRequestView,
)


class UGSegmentState(str, Enum):
    U_PREFILL = "u_prefill"
    U_DECODE = "u_decode"
    G_GENERATE = "g_generate"
    APPEND_IMAGE = "append_image"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class UGInterleavedMessage:
    type: Literal["text", "image"]
    content: Any


@dataclass(slots=True)
class UGSRTPreparedInput:
    """One materialized SRT input chunk for a UG U-side segment.

    `input_embeds` and `position_ids` are relative to `input_ids`. The runtime
    shifts them to the full SRT session request after `Session.create_req`
    prepends cached context and strips BOS for append requests.
    """

    input_ids: list[int]
    input_text: str
    messages: list[UGInterleavedMessage]
    input_embeds: list[list[float]] | None = None
    replace_embeds: list[list[float]] | None = None
    replace_positions: list[int] | None = None
    position_ids: list[Any] | None = None
    mm_inputs: Any | None = None
    srt_sidecar_role: str | None = None
    srt_sidecar_session_id: str | None = None
    adapter_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UGDecodeResult:
    type: Literal["text", "image_marker", "done"]
    text: str | None = None
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class UGTextDecodeResult:
    session: UGSessionHandle
    output_ids: tuple[int, ...]
    text: str
    input_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class UGVLMTextGenerationResult:
    session: UGSessionHandle
    text: str
    token_ids: tuple[int, ...] = ()
    next_token_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class UGSessionRecord:
    session_id: str
    state: UGSegmentState
    anchor_request_id: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    srt_request_count: int = 0
    srt_last_request_id: str | None = None
    srt_last_origin_input_len: int = 0
    srt_u_decode_request_count: int = 0
    srt_last_u_decode_request_id: str | None = None
    srt_last_u_decode_origin_input_len: int = 0
    srt_last_u_decode_output_ids: list[int] = field(default_factory=list)
    srt_last_u_decode_text: str = ""
    srt_mm_offsets: list[tuple[int, int]] = field(default_factory=list)
    srt_mm_inputs: Any | None = None
    srt_executed_request_count: int = 0
    srt_model_runner_forward_request_ids: set[str] = field(default_factory=set)
    srt_sidecar_session_ids: set[str] = field(default_factory=set)
    srt_sidecar_records: dict[str, "UGSessionRecord"] = field(default_factory=dict)
    srt_sidecar_request_count: int = 0
    ug_model_state: dict[str, Any] = field(default_factory=dict)
    closed: bool = False

    def handle(self) -> UGSessionHandle:
        return UGSessionHandle(
            session_id=self.session_id,
            anchor_request_id=self.anchor_request_id,
            context_length=self.context_length,
            context_version=self.context_version,
        )


class UGModelRunnerProtocol(Protocol):
    def prepare_srt_u_message_inputs(
        self,
        *,
        record: UGSessionRecord,
        message: UGInterleavedMessage,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput] | None: ...

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

    def append_generated_image(
        self, *, record: UGSessionRecord, image: Any | None
    ) -> int: ...

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


class UGSessionRuntime:
    """Lightweight UG state machine layered on top of SRT sessions."""

    def __init__(
        self,
        *,
        model_runner: UGModelRunnerProtocol,
        session_controller: Any | None = None,
        srt_request_executor: UGSRTRequestExecutorProtocol | None = None,
        capacity_of_str_len: int = 4096,
        tokenizer: Any | None = None,
        vocab_size: int = 32000,
        srt_u_decode_max_new_tokens: int = 0,
    ) -> None:
        self.model_runner = model_runner
        self.session_controller = session_controller
        self.srt_request_executor = srt_request_executor
        self.capacity_of_str_len = capacity_of_str_len
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.srt_u_decode_max_new_tokens = srt_u_decode_max_new_tokens
        self._records: dict[str, UGSessionRecord] = {}
        register_observer = getattr(
            self.srt_request_executor,
            "set_session_forward_observer",
            None,
        )
        if not callable(register_observer):
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

    def begin_g_segment(self, handle: UGSessionHandle) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot enter G segment from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        record.state = UGSegmentState.G_GENERATE
        return record.handle()

    def decode_next_segment(self, handle: UGSessionHandle) -> UGDecodeResult:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot decode U segment from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        decode_from_runtime = getattr(
            self.model_runner, "decode_next_segment_from_runtime", None
        )
        if callable(decode_from_runtime):
            result = decode_from_runtime(runtime=self, record=record)
        else:
            if self.srt_u_decode_max_new_tokens > 0:
                self._append_srt_u_decode_request(record, greedy=True)
            result = self.model_runner.decode_next_segment(record=record)
        record.decode_count += 1
        if result.type == "image_marker":
            record.state = UGSegmentState.G_GENERATE
        elif result.type == "done":
            record.state = UGSegmentState.DONE
        else:
            record.state = UGSegmentState.U_DECODE
        return result

    def decode_text(
        self,
        handle: UGSessionHandle,
        *,
        max_new_tokens: int | None = None,
        start_token_id: int | None = None,
        position_ids: list[int] | tuple[int, ...] | None = None,
        decode_position_id: int | None = None,
        drop_previous_output: bool = False,
        greedy: bool = False,
        model_state_updates: dict[str, Any] | None = None,
    ) -> UGTextDecodeResult:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot decode U text from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        input_ids = [] if start_token_id is None else [int(start_token_id)]
        output_ids = self._append_srt_u_decode_request(
            record,
            max_new_tokens=max_new_tokens,
            input_ids=input_ids,
            position_ids=position_ids,
            decode_position_id=decode_position_id,
            drop_previous_output=drop_previous_output,
            greedy=greedy,
            model_state_updates=model_state_updates,
        )
        return UGTextDecodeResult(
            session=record.handle(),
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            position_ids=tuple(int(position) for position in (position_ids or ())),
            text=record.srt_last_u_decode_text,
        )

    def commit_u_decode_input_token(
        self,
        handle: UGSessionHandle,
        *,
        token_id: int,
        position_id: int | None = None,
        model_state_updates: dict[str, Any] | None = None,
    ) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.U_DECODE:
            raise ValueError(
                f"Cannot commit U decode token from state {record.state} "
                f"for UG session {handle.session_id}"
            )
        request_id = f"{record.session_id}:d{record.srt_u_decode_request_count + 1}"
        req, _ = self._create_srt_session_req(
            record,
            request_id=request_id,
            input_ids=[int(token_id)],
            input_text="",
            mm_inputs=None,
            max_new_tokens=0,
            drop_previous_output=True,
            greedy=True,
        )
        if position_id is not None:
            prefix_len = len(req.origin_input_ids) - 1
            if prefix_len < 0:
                raise RuntimeError(
                    "UG SRT U decode commit requires a non-empty token input"
                )
            req.custom_position_ids = list(range(prefix_len)) + [int(position_id)]
        self._record_srt_req(record, req, request_id=request_id)
        adapter_metadata = {
            "ug_srt_added_token_count": 1,
            "ug_srt_rope_delta": 1,
            "ug_srt_committed_decode_token": int(token_id),
        }
        if position_id is not None:
            adapter_metadata["ug_srt_decode_position_id"] = int(position_id)
        if model_state_updates is not None:
            adapter_metadata["ug_model_state_updates"] = model_state_updates
            self._merge_ug_model_state_updates(record, model_state_updates)
        adapter_metadata["ug_model_state"] = self._copy_ug_model_state(
            record.ug_model_state
        )
        self._attach_srt_u_forward_metadata(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
            adapter_metadata=adapter_metadata,
        )
        self._execute_srt_req(record, req, state=UGSegmentState.U_DECODE)
        record.srt_u_decode_request_count += 1
        record.srt_last_u_decode_request_id = request_id
        record.srt_last_u_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_u_decode_output_ids = []
        record.srt_last_u_decode_text = ""
        record.context_length = len(req.origin_input_ids)
        self._notify_srt_u_forward(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
        )
        return record.handle()

    def append_generated_image(
        self, handle: UGSessionHandle, image: Any | None
    ) -> UGSessionHandle:
        record = self._record_for(handle)
        if record.state != UGSegmentState.G_GENERATE:
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
            sidecar_session_ids = set(record.srt_sidecar_session_ids)
        else:
            sidecar_session_ids = set()
        self.model_runner.close_session(session_id=session_id)
        self._close_srt_session(session_id)
        for sidecar_session_id in sorted(sidecar_session_ids):
            self._close_srt_session(sidecar_session_id)

    def get_srt_sidecar_handle(
        self,
        owner_handle: UGSessionHandle | str,
        role: str,
    ) -> UGSessionHandle | None:
        owner_record = self._record_for(owner_handle)
        sidecar_record = owner_record.srt_sidecar_records.get(
            f"{owner_record.session_id}:{role}"
        )
        return None if sidecar_record is None else sidecar_record.handle()

    def get_srt_sidecar_model_state(
        self,
        owner_handle: UGSessionHandle | str,
        role: str,
    ) -> dict[str, Any]:
        owner_record = self._record_for(owner_handle)
        sidecar_record = owner_record.srt_sidecar_records.get(
            f"{owner_record.session_id}:{role}"
        )
        if sidecar_record is None:
            return {}
        return self._copy_ug_model_state(sidecar_record.ug_model_state)

    def append_srt_sidecar_prepared_input(
        self,
        owner_handle: UGSessionHandle | str,
        prepared: UGSRTPreparedInput,
        *,
        state: UGSegmentState | None = None,
    ) -> UGSessionHandle:
        owner_record = self._record_for(owner_handle)
        sidecar_record = self._append_srt_sidecar_request(
            owner_record,
            prepared,
            state=state,
        )
        return sidecar_record.handle()

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
            "append_image_count": record.append_image_count,
            "decode_count": record.decode_count,
            "srt_request_count": record.srt_request_count,
            "srt_sidecar_request_count": record.srt_sidecar_request_count,
            "srt_sidecar_session_ids": sorted(record.srt_sidecar_session_ids),
            "srt_sidecar_ug_model_state": {
                session_id: self._copy_ug_model_state(sidecar_record.ug_model_state)
                for session_id, sidecar_record in sorted(
                    record.srt_sidecar_records.items()
                )
            },
            "ug_model_state": self._copy_ug_model_state(record.ug_model_state),
            "srt_last_request_id": record.srt_last_request_id,
            "srt_last_origin_input_len": record.srt_last_origin_input_len,
            "srt_u_decode_request_count": record.srt_u_decode_request_count,
            "srt_last_u_decode_request_id": record.srt_last_u_decode_request_id,
            "srt_last_u_decode_origin_input_len": (
                record.srt_last_u_decode_origin_input_len
            ),
            "srt_last_u_decode_output_ids": record.srt_last_u_decode_output_ids,
            "srt_last_u_decode_text": record.srt_last_u_decode_text,
            "srt_mm_offsets": record.srt_mm_offsets,
            "srt_executed_request_count": record.srt_executed_request_count,
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

        prepared_inputs = self._prepare_srt_u_inputs(
            record,
            messages,
            state=record.state,
        )
        total_inputs = sum(
            1 for prepared in prepared_inputs if prepared.srt_sidecar_role is None
        )
        main_input_index = 0
        for prepared in prepared_inputs:
            if prepared.srt_sidecar_role is not None:
                self._append_srt_sidecar_request(record, prepared)
                continue
            main_input_index += 1
            segment_request_id = (
                request_id if total_inputs == 1 else f"{request_id}:s{main_input_index}"
            )
            is_final_segment = main_input_index == total_inputs
            input_ids = prepared.input_ids
            input_text = prepared.input_text
            mm_inputs = prepared.mm_inputs
            req, recv_req = self._create_srt_session_req(
                record,
                request_id=segment_request_id,
                input_ids=input_ids,
                input_text=input_text,
                mm_inputs=mm_inputs,
                max_new_tokens=0,
            )

            if mm_inputs is not None:
                SessionController.adjust_mm_offsets(recv_req, req, mm_inputs)
                self._pad_srt_multimodal_input_ids(req, mm_inputs)
                req.extend_image_inputs(mm_inputs)

            adapter_metadata = self._apply_prepared_srt_input(
                req,
                record=record,
                recv_req=recv_req,
                prepared=prepared,
                is_final_segment=is_final_segment,
            )

            self._record_srt_req(record, req, request_id=segment_request_id)
            self._attach_srt_u_forward_metadata(
                record,
                req,
                state=record.state,
                input_text=input_text,
                messages=prepared.messages,
                adapter_metadata=adapter_metadata,
            )
            self._execute_srt_req(record, req, state=record.state)
            self._notify_srt_u_forward(
                record,
                req,
                state=record.state,
                input_text=input_text,
                messages=prepared.messages,
            )

    def _append_srt_sidecar_request(
        self,
        owner_record: UGSessionRecord,
        prepared: UGSRTPreparedInput,
        *,
        state: UGSegmentState | None = None,
    ) -> UGSessionRecord:
        role = prepared.srt_sidecar_role
        if not role:
            raise RuntimeError("UG SRT sidecar request requires a sidecar role")
        sidecar_session_id = (
            prepared.srt_sidecar_session_id or f"{owner_record.session_id}:{role}"
        )
        owner_record.srt_sidecar_session_ids.add(sidecar_session_id)
        self._ensure_srt_session(sidecar_session_id)
        sidecar_state = state or owner_record.state
        request_id = (
            f"{owner_record.session_id}:sidecar:{role}:"
            f"{owner_record.context_version + 1}:"
            f"{owner_record.srt_sidecar_request_count + 1}"
        )
        sidecar_record = owner_record.srt_sidecar_records.get(sidecar_session_id)
        if sidecar_record is None:
            sidecar_record = UGSessionRecord(
                session_id=sidecar_session_id,
                state=sidecar_state,
                anchor_request_id=request_id,
            )
            owner_record.srt_sidecar_records[sidecar_session_id] = sidecar_record
        sidecar_record.state = sidecar_state
        sidecar_record.anchor_request_id = request_id
        sidecar_record.context_version += 1
        req, recv_req = self._create_srt_session_req(
            sidecar_record,
            request_id=request_id,
            input_ids=prepared.input_ids,
            input_text=prepared.input_text,
            mm_inputs=prepared.mm_inputs,
            max_new_tokens=0,
        )
        if prepared.mm_inputs is not None:
            from sglang.srt.session.session_controller import SessionController

            SessionController.adjust_mm_offsets(recv_req, req, prepared.mm_inputs)
            self._pad_srt_multimodal_input_ids(req, prepared.mm_inputs)
            req.extend_image_inputs(prepared.mm_inputs)
        adapter_metadata = self._apply_prepared_srt_input(
            req,
            record=sidecar_record,
            recv_req=recv_req,
            prepared=prepared,
            is_final_segment=True,
        )
        adapter_metadata["ug_srt_owner_session_id"] = owner_record.session_id
        adapter_metadata["ug_srt_sidecar_role"] = role
        self._record_srt_req(sidecar_record, req, request_id=request_id)
        self._attach_srt_u_forward_metadata(
            sidecar_record,
            req,
            state=sidecar_state,
            input_text=prepared.input_text,
            messages=prepared.messages,
            adapter_metadata=adapter_metadata,
        )
        self._execute_srt_req(sidecar_record, req, state=sidecar_state)
        sidecar_record.context_length = len(req.origin_input_ids)
        self._notify_srt_u_forward(
            sidecar_record,
            req,
            state=sidecar_state,
            input_text=prepared.input_text,
            messages=prepared.messages,
        )
        owner_record.srt_sidecar_request_count += 1
        return sidecar_record

    def _pad_srt_multimodal_input_ids(self, req: Any, mm_inputs: Any) -> None:
        pad_input_ids = getattr(self.srt_request_executor, "pad_input_ids", None)
        if not callable(pad_input_ids):
            return
        req.origin_input_ids = pad_input_ids(req.origin_input_ids, mm_inputs)

    def _prepare_srt_u_inputs(
        self,
        record: UGSessionRecord,
        messages: list[UGInterleavedMessage],
        *,
        state: UGSegmentState,
    ) -> list[UGSRTPreparedInput]:
        prepare_all = getattr(
            self.model_runner, "prepare_srt_u_interleaved_inputs", None
        )
        if callable(prepare_all):
            custom_inputs = prepare_all(record=record, messages=messages, state=state)
            if custom_inputs is not None:
                return custom_inputs

        prepare_one = getattr(self.model_runner, "prepare_srt_u_message_inputs", None)
        if not callable(prepare_one):
            raise RuntimeError(
                f"{self.model_runner.__class__.__name__} must provide explicit "
                "UG SRT prepared inputs; runtime fallback tokenization is disabled"
            )

        prepared_inputs: list[UGSRTPreparedInput] = []
        for message in messages:
            custom = prepare_one(record=record, message=message, state=state)
            if custom is None:
                raise RuntimeError(
                    f"{self.model_runner.__class__.__name__} did not prepare UG "
                    f"SRT input for message type {message.type!r}"
                )
            prepared_inputs.extend(custom)
        if not prepared_inputs:
            raise RuntimeError("UG SRT prepared inputs must not be empty")
        return prepared_inputs

    def _apply_prepared_srt_input(
        self,
        req: Any,
        *,
        record: UGSessionRecord,
        recv_req: Any,
        prepared: UGSRTPreparedInput,
        is_final_segment: bool,
    ) -> dict[str, Any]:
        prefix_len, stripped = self._srt_prefix_and_strip_lengths(
            req, recv_req, prepared
        )
        new_token_count = len(recv_req.input_ids)

        adapter_metadata = dict(prepared.adapter_metadata)
        adapter_metadata.setdefault("ug_srt_added_token_count", new_token_count)
        adapter_metadata["ug_srt_is_final_segment"] = bool(is_final_segment)
        self._merge_ug_model_state_updates(
            record,
            adapter_metadata.get("ug_model_state_updates"),
        )
        adapter_metadata["ug_model_state"] = self._copy_ug_model_state(
            record.ug_model_state
        )

        if prepared.input_embeds is not None:
            suffix_embeds = prepared.input_embeds[stripped:]
            hidden_size = len(suffix_embeds[0]) if suffix_embeds else 0
            prefix_embeds = [[0.0] * hidden_size for _ in range(prefix_len)]
            req.input_embeds = prefix_embeds + suffix_embeds

        if prepared.replace_embeds is not None:
            replace_positions = prepared.replace_positions
            if replace_positions is None:
                raise RuntimeError(
                    "UG prepared SRT replace_embeds requires replace_positions"
                )
            shifted_embeds = []
            shifted_positions = []
            for embed, position in zip(prepared.replace_embeds, replace_positions):
                if position < stripped:
                    continue
                shifted_embeds.append(embed)
                shifted_positions.append(prefix_len + position - stripped)
            req.context_replace_embeds = shifted_embeds
            req.context_replace_positions = shifted_positions

        if prepared.position_ids is not None:
            suffix_positions = prepared.position_ids[stripped:]
            if self._uses_multidim_positions(suffix_positions):
                req.custom_position_ids = self._full_multidim_positions(
                    prefix_len=prefix_len,
                    suffix_positions=suffix_positions,
                )
                self._install_multidim_mm_positions(req)
            else:
                req.custom_position_ids = list(range(prefix_len)) + [
                    int(position) for position in suffix_positions
                ]

        return adapter_metadata

    @staticmethod
    def _srt_prefix_and_strip_lengths(
        req: Any,
        recv_req: Any,
        prepared: UGSRTPreparedInput,
    ) -> tuple[int, int]:
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        stripped = len(prepared.input_ids) - len(recv_req.input_ids)
        if prefix_len < 0 or stripped < 0:
            raise RuntimeError(
                "UG prepared SRT input length is inconsistent with session request"
            )
        return prefix_len, stripped

    @staticmethod
    def _uses_multidim_positions(positions: list[Any]) -> bool:
        return bool(positions) and isinstance(positions[0], (list, tuple))

    @staticmethod
    def _full_multidim_positions(
        *,
        prefix_len: int,
        suffix_positions: list[Any],
    ) -> list[list[int]]:
        prefix_positions = [[position, 0, 0] for position in range(prefix_len)]
        return prefix_positions + [
            [int(value) for value in position] for position in suffix_positions
        ]

    @staticmethod
    def _install_multidim_mm_positions(req: Any) -> None:
        mm_inputs = getattr(req, "multimodal_inputs", None)
        positions = getattr(req, "custom_position_ids", None)
        if mm_inputs is None or not positions:
            return
        if not isinstance(positions[0], (list, tuple)):
            return
        mrope_positions = torch.tensor(positions, dtype=torch.long).t().contiguous()
        mm_inputs.mrope_positions = mrope_positions
        mm_inputs.mrope_position_delta = (
            mrope_positions[:, -1:]
            .max(
                dim=0,
                keepdim=True,
            )
            .values
        )

    def _append_srt_u_decode_request(
        self,
        record: UGSessionRecord,
        *,
        max_new_tokens: int | None = None,
        input_ids: list[int] | None = None,
        position_ids: list[int] | tuple[int, ...] | None = None,
        decode_position_id: int | None = None,
        drop_previous_output: bool = False,
        greedy: bool = False,
        model_state_updates: dict[str, Any] | None = None,
    ) -> list[int]:
        max_new_tokens = (
            self.srt_u_decode_max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if max_new_tokens <= 0:
            raise ValueError("UG SRT U decode requires max_new_tokens > 0")
        input_ids = list(input_ids or [])
        if position_ids is not None and len(position_ids) != len(input_ids):
            raise ValueError(
                "UG SRT U decode position_ids must match input_ids length: "
                f"{len(position_ids)} != {len(input_ids)}"
            )
        if decode_position_id is not None and input_ids:
            raise ValueError(
                "UG SRT U decode_position_id is only valid for next-token decode"
            )
        request_id = f"{record.session_id}:d{record.srt_u_decode_request_count + 1}"
        req, _ = self._create_srt_session_req(
            record,
            request_id=request_id,
            input_ids=input_ids,
            input_text="",
            mm_inputs=None,
            max_new_tokens=max_new_tokens,
            drop_previous_output=drop_previous_output,
            greedy=greedy,
        )
        if position_ids is not None:
            prefix_len = len(req.origin_input_ids) - len(input_ids)
            if prefix_len < 0:
                raise RuntimeError(
                    "UG SRT U decode input length is inconsistent with session request"
                )
            req.custom_position_ids = list(range(prefix_len)) + [
                int(position) for position in position_ids
            ]
        if decode_position_id is not None:
            req.custom_decode_position_id = int(decode_position_id)
            if not input_ids:
                if not req.origin_input_ids:
                    raise RuntimeError(
                        "UG SRT U decode_position_id requires a non-empty "
                        "session context"
                    )
                req.custom_position_ids = list(range(len(req.origin_input_ids)))
                req.custom_position_ids[-1] = int(decode_position_id)
        self._record_srt_req(record, req, request_id=request_id)
        adapter_metadata = {
            "ug_srt_added_token_count": len(input_ids),
        }
        if position_ids is not None:
            adapter_metadata["ug_srt_rope_delta"] = len(input_ids)
        if decode_position_id is not None:
            adapter_metadata["ug_srt_decode_position_id"] = int(decode_position_id)
        if model_state_updates is not None:
            adapter_metadata["ug_model_state_updates"] = model_state_updates
            self._merge_ug_model_state_updates(record, model_state_updates)
        adapter_metadata["ug_model_state"] = self._copy_ug_model_state(
            record.ug_model_state
        )
        self._attach_srt_u_forward_metadata(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
            adapter_metadata=adapter_metadata,
        )
        self._execute_srt_req(record, req, state=UGSegmentState.U_DECODE)
        record.srt_u_decode_request_count += 1
        record.srt_last_u_decode_request_id = request_id
        record.srt_last_u_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_u_decode_output_ids = list(
            req.output_ids[: req.sampling_params.max_new_tokens]
        )
        record.srt_last_u_decode_text = self._decode_token_ids(
            record.srt_last_u_decode_output_ids
        )
        record.context_length = len(req.origin_input_ids) + len(
            record.srt_last_u_decode_output_ids
        )
        self._notify_srt_u_forward(
            record,
            req,
            state=UGSegmentState.U_DECODE,
            input_text="",
            messages=[],
        )
        return list(record.srt_last_u_decode_output_ids)

    def _decode_token_ids(self, token_ids: list[int] | tuple[int, ...]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            return str(decode(list(token_ids)))
        return " ".join(str(token_id) for token_id in token_ids)

    def _create_srt_session_req(
        self,
        record: UGSessionRecord,
        *,
        request_id: str,
        input_ids: list[int],
        input_text: str,
        mm_inputs: Any | None,
        max_new_tokens: int,
        drop_previous_output: bool = False,
        greedy: bool = False,
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
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=0.0 if greedy else 1.0,
        )
        sampling_params.normalize(self.tokenizer)
        recv_req = TokenizedGenerateReqInput(
            rid=request_id,
            input_text=input_text,
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(
                id=record.session_id,
                rid=record.srt_last_request_id,
                replace=False,
                drop_previous_output=drop_previous_output,
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
        adapter_metadata: dict[str, Any] | None = None,
    ) -> None:
        req.session_forward_metadata = {
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
            "adapter_metadata": dict(adapter_metadata or {}),
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
        metadata = dict(
            getattr(req, "session_forward_metadata", {}).get("adapter_metadata", {})
        )
        metadata.setdefault(
            "ug_model_state", self._copy_ug_model_state(record.ug_model_state)
        )
        if state == UGSegmentState.U_DECODE:
            metadata["srt_last_u_decode_text"] = record.srt_last_u_decode_text
        token_binding = self._srt_kv_token_binding(record, req, state=state)
        if token_binding is not None:
            metadata["srt_kv_token_binding"] = token_binding
        return metadata

    @classmethod
    def _merge_ug_model_state_updates(
        cls,
        record: UGSessionRecord,
        updates: dict[str, Any] | None,
    ) -> None:
        if not updates:
            return
        if not isinstance(updates, dict):
            raise ValueError("UG model state updates must be a dict")
        for namespace, namespace_updates in updates.items():
            if not isinstance(namespace_updates, dict):
                raise ValueError(
                    "UG model state namespace updates must be dicts: " f"{namespace!r}"
                )
            state = record.ug_model_state.setdefault(str(namespace), {})
            for key, value in namespace_updates.items():
                state[str(key)] = cls._copy_ug_model_state_value(value)

    @classmethod
    def _copy_ug_model_state(cls, state: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): cls._copy_ug_model_state_value(value)
            for key, value in state.items()
        }

    @classmethod
    def _copy_ug_model_state_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._copy_ug_model_state_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._copy_ug_model_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._copy_ug_model_state_value(item) for item in value)
        return value

    def _srt_kv_token_binding(
        self,
        record: UGSessionRecord,
        req: Any,
        *,
        state: UGSegmentState,
    ) -> UGSRTKVTokenBinding | None:
        provider = getattr(
            self.srt_request_executor, "get_request_token_binding", None
        )
        if not callable(provider):
            provider = getattr(
                self.srt_request_executor, "get_ug_request_token_binding", None
            )
        if not callable(provider):
            return None

        binding = provider(record=record, req=req, state=state)
        if binding is None:
            return None
        if not isinstance(binding, UGSRTKVTokenBinding):
            raise TypeError(
                "UG SRT request token binding provider must return "
                f"UGSRTKVTokenBinding, got {type(binding).__name__}"
            )
        return binding

    @staticmethod
    def _collect_mm_offsets(mm_inputs: Any | None) -> list[tuple[int, int]]:
        if mm_inputs is None:
            return []
        offsets: list[tuple[int, int]] = []
        for item in getattr(mm_inputs, "mm_items", []):
            offsets.extend(getattr(item, "offsets", []) or [])
        return offsets

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
