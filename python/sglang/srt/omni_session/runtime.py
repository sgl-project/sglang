# SPDX-License-Identifier: Apache-2.0
"""SRT-owned session runtime for AR/generation interleaved models.

The runtime is below the generic omni coordinator. It materializes model-
specific AR-side chunks as ordinary SRT session requests, tracks committed SRT
KV bindings, and owns the scheduler executor used by runtime-aware model hooks.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import torch

from sglang.srt.managers.io_struct import (
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, MultimodalInputs, Req
from sglang.srt.omni_session.runtime_types import (
    OmniSessionHandle,
)
from sglang.srt.omni_session.srt_executor import OmniSRTSchedulerExecutor
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController

if TYPE_CHECKING:
    from sglang.omni.entrypoints.streaming import OmniStreamSink
    from sglang.omni.model_adapters.session_model_hooks import (
        OmniSessionModelHooks,
        OmniSessionModelView,
    )


class OmniSegmentState(str, Enum):
    """State machine for one interleaved AR/media session segment.

    The normal path is:
    AR_PREFILL -> AR_DECODE -> GENERATE -> APPEND_IMAGE -> AR_DECODE.
    AR_DECODE may also stay in AR_DECODE after text-only output, or move to DONE.
    """

    # triggered by a new user turn; next action: committing prepared inputs to SRT KV
    AR_PREFILL = "ar_prefill"
    # triggered after prefill/image append; next action: AR decode to text/media boundary
    AR_DECODE = "ar_decode"
    # triggered when AR emits an image boundary; next action: multimodal generation
    GENERATE = "generate"
    # triggered after image generation; next action: committing the generated image to SRT KV
    APPEND_IMAGE = "append_image"
    # triggered by EOS/close; next action: returning final output or accepting a later user turn
    DONE = "done"


@dataclass(frozen=True, slots=True)
class OmniInterleavedMessage:
    """
    The deriviation of the message:
       OmniRequest.messages -> OmniInterleavedMessage -> OmniSRTPreparedInput

    """

    type: Literal["text", "image"]
    content: Any


@dataclass(slots=True)
class OmniSRTPreparedInput:
    """One materialized SRT input chunk for an omni AR-side segment, consumed by OmniSRT runtime

    `input_embeds` and `position_ids` are relative to `input_ids`. The runtime
    shifts them to the full SRT session request after `Session.create_req`
    prepends cached context and strips BOS for append requests.
    """

    input_ids: list[int]
    input_text: str
    messages: list[OmniInterleavedMessage]
    input_embeds: list[list[float]] | None = None
    replace_embeds: list[list[float]] | None = None
    replace_positions: list[int] | None = None
    position_ids: list[Any] | None = None
    mm_inputs: MultimodalInputs | None = None
    condition_path_role: str | None = None
    condition_path_session_id: str | None = None
    # private hints from model policies to the SRT omni runtime, not client metadata
    policy_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OmniDecodeResult:
    """interleaved boundary returned by model-specific segment decode"""

    type: Literal["text", "image_marker", "done"]
    text: str | None = None
    token_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OmniTextDecodeResult:
    """plain SRT text decode result for an existing omni session"""

    session: OmniSessionHandle
    output_ids: tuple[int, ...]
    text: str
    input_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OmniVLMTextGenerationResult:
    """plain VLM answer text decoded from an image/text-prefilled session"""

    session: OmniSessionHandle
    text: str
    token_ids: tuple[int, ...] = ()
    next_token_ids: tuple[int, ...] = ()
    position_ids: tuple[int, ...] = ()
    streamed_text: bool = False


@dataclass(slots=True)
class OmniSessionRecord:
    """Mutable server-side record for one omni conversation session."""

    session_id: str
    state: OmniSegmentState
    anchor_request_id: str
    context_length: int = 0
    context_version: int = 0
    prefill_count: int = 0
    append_image_count: int = 0
    decode_count: int = 0
    srt_request_count: int = 0
    srt_last_request_id: str | None = None
    srt_last_origin_input_len: int = 0
    srt_ar_decode_request_count: int = 0
    srt_last_ar_decode_request_id: str | None = None
    srt_last_ar_decode_origin_input_len: int = 0
    srt_last_ar_decode_output_ids: list[int] = field(default_factory=list)
    srt_last_ar_decode_text: str = ""
    srt_mm_offsets: list[tuple[int, int]] = field(default_factory=list)
    srt_mm_inputs: MultimodalInputs | None = None
    srt_executed_request_count: int = 0
    # condition paths hold auxiliary generation contexts and close with the owner session
    condition_path_session_ids: set[str] = field(default_factory=set)
    condition_path_records: dict[str, "OmniSessionRecord"] = field(default_factory=dict)
    condition_path_request_count: int = 0
    omni_model_state: dict[str, Any] = field(default_factory=dict)
    closed: bool = False

    def handle(self) -> OmniSessionHandle:
        return OmniSessionHandle(
            session_id=self.session_id,
            anchor_request_id=self.anchor_request_id,
            context_length=self.context_length,
            context_version=self.context_version,
        )


class OmniSessionRuntime:
    """Generic SRT session state machine used by model-specific session adapters.

    The runtime owns session state, asks `model_hooks` for model-specific
    prompt/decode rules, and owns the SRT scheduler executor that runs concrete
    SRT `Req` objects. Session adapters select request-level mode/options;
    hooks define token grammar and state patches for one model.
    """

    def __init__(
        self,
        *,
        model_hooks: "OmniSessionModelHooks",
        session_controller: SessionController | None = None,
        srt_request_executor: OmniSRTSchedulerExecutor | None = None,
        capacity_of_str_len: int = 4096,
        tokenizer: Any | None = None,
        vocab_size: int = 32000,
        srt_ar_decode_max_new_tokens: int = 0,
    ) -> None:
        self.model_hooks: "OmniSessionModelHooks" = model_hooks
        self.session_controller = session_controller
        self.srt_request_executor = srt_request_executor
        self.capacity_of_str_len = capacity_of_str_len
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.srt_ar_decode_max_new_tokens = srt_ar_decode_max_new_tokens
        # tracks the omni sessions
        self._records: dict[str, OmniSessionRecord] = {}

    @staticmethod
    def normalize_messages(
        *, prompt: str | list[str] | None = None, image: Any | None = None
    ) -> list[OmniInterleavedMessage]:
        messages: list[OmniInterleavedMessage] = []
        if image is not None:
            messages.append(OmniInterleavedMessage(type="image", content=image))
        if prompt is not None:
            prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt
            messages.append(OmniInterleavedMessage(type="text", content=prompt_text))
        return messages

    def _model_session_view(self, record: OmniSessionRecord) -> "OmniSessionModelView":
        from sglang.omni.model_adapters.session_model_hooks import (
            OmniSessionModelView,
        )

        return OmniSessionModelView(
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
                "omni_model_state": self._copy_omni_model_state(
                    record.omni_model_state
                ),
            },
        )

    def prefill_interleaved(
        self,
        messages: list[OmniInterleavedMessage],
        *,
        session_id: str | None = None,
    ) -> OmniSessionHandle:
        """perform a prefill based on the interleaved context"""
        if not messages:
            raise ValueError("omni prefill requires at least one text or image message")
        session_id = session_id or uuid.uuid4().hex
        record = self._records.get(session_id)
        if record is None:
            self._ensure_srt_session(session_id)
            record = OmniSessionRecord(
                session_id=session_id,
                state=OmniSegmentState.AR_PREFILL,
                anchor_request_id=f"{session_id}:ar0",
            )
            self._records[session_id] = record
        elif record.closed:
            raise ValueError(f"omni session {session_id} is closed")
        elif record.state not in {OmniSegmentState.AR_DECODE, OmniSegmentState.DONE}:
            raise ValueError(
                f"Cannot prefill omni session {session_id} from state {record.state}"
            )
        else:
            record.state = OmniSegmentState.AR_PREFILL

        next_context_version = record.context_version + 1
        next_anchor_request_id = f"{session_id}:ar{next_context_version}"
        record.anchor_request_id = next_anchor_request_id
        self._commit_generated_image_to_srt_session_kv(
            record,
            messages,
            request_id=next_anchor_request_id,
        )
        prefill_result = self.model_hooks.on_prefill_finished(
            session=self._model_session_view(record), messages=messages
        )
        record.context_length += prefill_result.added_tokens
        record.context_version = next_context_version
        record.prefill_count += 1
        record.state = OmniSegmentState.AR_DECODE
        return record.handle()

    def decode_next_segment(
        self,
        handle: OmniSessionHandle,
        *,
        stream_sink: "OmniStreamSink | None" = None,
    ) -> OmniDecodeResult:
        """advance an AR session until model hooks return the next segment boundary"""
        record = self._record_for(handle)
        if record.state != OmniSegmentState.AR_DECODE:
            raise ValueError(
                f"Cannot decode AR segment from state {record.state} "
                f"for omni session {handle.session_id}"
            )
        result = self.model_hooks.decode_next_segment_with_runtime(
            runtime=self,
            session=self._model_session_view(record),
            stream_sink=stream_sink,
        )
        record.decode_count += 1
        if result.type == "image_marker":
            record.state = OmniSegmentState.GENERATE
        elif result.type == "done":
            record.state = OmniSegmentState.DONE
        else:
            record.state = OmniSegmentState.AR_DECODE
        return result

    def decode(
        self,
        handle: OmniSessionHandle,
        *,
        max_new_tokens: int | None = None,
        start_token_id: int | None = None,
        position_ids: list[int] | tuple[int, ...] | None = None,
        decode_position_id: int | None = None,
        drop_previous_output: bool = False,
        greedy: bool = False,
        model_state_updates: dict[str, Any] | None = None,
    ) -> OmniTextDecodeResult:
        """performs decode on a request"""
        record = self._record_for(handle)
        if record.state != OmniSegmentState.AR_DECODE:
            raise ValueError(
                f"Cannot decode AR text from state {record.state} "
                f"for omni session {handle.session_id}"
            )
        input_ids = [] if start_token_id is None else [int(start_token_id)]
        output_ids = self._execute_srt_ar_decode_request(
            record,
            max_new_tokens=max_new_tokens,
            input_ids=input_ids,
            position_ids=position_ids,
            decode_position_id=decode_position_id,
            drop_previous_output=drop_previous_output,
            greedy=greedy,
            model_state_updates=model_state_updates,
        )
        return OmniTextDecodeResult(
            session=record.handle(),
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            position_ids=tuple(int(position) for position in (position_ids or ())),
            text=record.srt_last_ar_decode_text,
        )

    def commit_ar_decode_input_token(
        self,
        handle: OmniSessionHandle,
        *,
        token_id: int,
        position_id: int | None = None,
        model_state_updates: dict[str, Any] | None = None,
    ) -> OmniSessionHandle:
        record = self._record_for(handle)
        if record.state != OmniSegmentState.AR_DECODE:
            raise ValueError(
                f"Cannot commit AR decode token from state {record.state} "
                f"for omni session {handle.session_id}"
            )
        request_id = f"{record.session_id}:d{record.srt_ar_decode_request_count + 1}"
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
                    "omni SRT AR decode commit requires a non-empty token input"
                )
            req.custom_position_ids = list(range(prefix_len)) + [int(position_id)]
        self._record_srt_req(record, req, request_id=request_id)
        policy_metadata = {
            "omni_srt_added_token_count": 1,
            "omni_srt_rope_delta": 1,
            "omni_srt_committed_decode_token": int(token_id),
        }
        if position_id is not None:
            policy_metadata["omni_srt_decode_position_id"] = int(position_id)
            policy_metadata["omni_srt_position_count"] = int(position_id) + 1
        if model_state_updates is not None:
            policy_metadata["omni_model_state_updates"] = model_state_updates
            self._merge_omni_model_state_updates(record, model_state_updates)
        policy_metadata["omni_model_state"] = self._copy_omni_model_state(
            record.omni_model_state
        )
        self._attach_srt_request_overrides(req, srt_request_metadata=policy_metadata)
        self._execute_srt_req(record, req, state=OmniSegmentState.AR_DECODE)
        record.srt_ar_decode_request_count += 1
        record.srt_last_ar_decode_request_id = request_id
        record.srt_last_ar_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_ar_decode_output_ids = []
        record.srt_last_ar_decode_text = ""
        record.context_length = len(req.origin_input_ids)
        return record.handle()

    def append_ar_input_tokens(
        self,
        handle: OmniSessionHandle,
        *,
        token_ids: list[int] | tuple[int, ...],
        position_ids: list[int] | tuple[int, ...] | None = None,
        model_state_updates: dict[str, Any] | None = None,
    ) -> OmniSessionHandle:
        record = self._record_for(handle)
        if record.state != OmniSegmentState.AR_DECODE:
            raise ValueError(
                f"Cannot append AR input tokens from state {record.state} "
                f"for omni session {handle.session_id}"
            )
        input_ids = [int(token_id) for token_id in token_ids]
        if not input_ids:
            return record.handle()
        if position_ids is not None and len(position_ids) != len(input_ids):
            raise ValueError(
                "omni SRT AR append position_ids must match token_ids length: "
                f"{len(position_ids)} != {len(input_ids)}"
            )
        request_id = f"{record.session_id}:d{record.srt_ar_decode_request_count + 1}"
        req, _ = self._create_srt_session_req(
            record,
            request_id=request_id,
            input_ids=input_ids,
            input_text="",
            mm_inputs=None,
            max_new_tokens=0,
            greedy=True,
        )
        if position_ids is not None:
            prefix_len = len(req.origin_input_ids) - len(input_ids)
            if prefix_len < 0:
                raise RuntimeError(
                    "omni SRT AR append input length is inconsistent with session request"
                )
            req.custom_position_ids = list(range(prefix_len)) + [
                int(position) for position in position_ids
            ]
        self._record_srt_req(record, req, request_id=request_id)
        policy_metadata = {
            "omni_srt_added_token_count": len(input_ids),
            "omni_srt_rope_delta": len(input_ids),
            "omni_srt_position_count": (
                max(int(position) for position in position_ids) + 1
                if position_ids is not None
                else len(req.origin_input_ids)
            ),
        }
        if model_state_updates is not None:
            policy_metadata["omni_model_state_updates"] = model_state_updates
            self._merge_omni_model_state_updates(record, model_state_updates)
        policy_metadata["omni_model_state"] = self._copy_omni_model_state(
            record.omni_model_state
        )
        self._attach_srt_request_overrides(req, srt_request_metadata=policy_metadata)
        # 1. keep previous decoded think tokens and append the explicit image marker
        self._execute_srt_req(record, req, state=OmniSegmentState.AR_DECODE)
        record.srt_ar_decode_request_count += 1
        record.srt_last_ar_decode_request_id = request_id
        record.srt_last_ar_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_ar_decode_output_ids = []
        record.srt_last_ar_decode_text = ""
        record.context_length = len(req.origin_input_ids)
        return record.handle()

    def append_generated_image(
        self, handle: OmniSessionHandle, image: Any | None
    ) -> OmniSessionHandle:
        """Commit one generated image segment back into the AR session."""

        record = self._record_for(handle)
        if record.state != OmniSegmentState.GENERATE:
            raise ValueError(
                f"Cannot append generated image from state {record.state} "
                f"for omni session {handle.session_id}"
            )
        record.state = OmniSegmentState.APPEND_IMAGE
        next_context_version = record.context_version + 1
        next_anchor_request_id = f"{record.session_id}:ar{next_context_version}"
        record.anchor_request_id = next_anchor_request_id
        # 1. commit the generated image back to srt session
        self._commit_generated_image_to_srt_session_kv(
            record,
            [OmniInterleavedMessage(type="image", content=image)],
            request_id=next_anchor_request_id,
        )
        # 2. model hooks account for model-specific context growth after SRT commit
        append_result = self.model_hooks.append_generated_image(
            session=self._model_session_view(record), image=image
        )
        record.context_length += append_result.added_tokens
        record.context_version = next_context_version
        record.append_image_count += 1
        record.state = OmniSegmentState.AR_DECODE
        return record.handle()

    def close_session(self, handle_or_session_id: OmniSessionHandle | str) -> None:
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, OmniSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is not None:
            record.closed = True
            record.state = OmniSegmentState.DONE
            condition_path_session_ids = set(record.condition_path_session_ids)
        else:
            condition_path_session_ids = set()
        self.model_hooks.close_session(session_id=session_id)
        self._close_srt_session(session_id)
        for condition_path_session_id in sorted(condition_path_session_ids):
            self._close_srt_session(condition_path_session_id)
        if self.srt_request_executor is not None:
            self.srt_request_executor.run_idle_cleanup()

    def get_condition_path_handle(
        self,
        owner_handle: OmniSessionHandle | str,
        role: str,
    ) -> OmniSessionHandle | None:
        owner_record = self._record_for(owner_handle)
        condition_path_record = owner_record.condition_path_records.get(
            f"{owner_record.session_id}:{role}"
        )
        return None if condition_path_record is None else condition_path_record.handle()

    def get_condition_path_model_state(
        self,
        owner_handle: OmniSessionHandle | str,
        role: str,
    ) -> dict[str, Any]:
        owner_record = self._record_for(owner_handle)
        condition_path_record = owner_record.condition_path_records.get(
            f"{owner_record.session_id}:{role}"
        )
        if condition_path_record is None:
            return {}
        return self._copy_omni_model_state(condition_path_record.omni_model_state)

    def append_condition_path_prepared_input(
        self,
        owner_handle: OmniSessionHandle | str,
        prepared: OmniSRTPreparedInput,
        *,
        state: OmniSegmentState | None = None,
    ) -> OmniSessionHandle:
        owner_record = self._record_for(owner_handle)
        condition_path_record = self._append_condition_path_request(
            owner_record,
            prepared,
            state=state,
        )
        return condition_path_record.handle()

    def get_state(
        self, handle_or_session_id: OmniSessionHandle | str
    ) -> OmniSegmentState:
        return self._record_for(handle_or_session_id).state

    def get_debug_counters(self, handle_or_session_id: OmniSessionHandle | str) -> dict:
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
            "condition_path_request_count": record.condition_path_request_count,
            "condition_path_session_ids": sorted(record.condition_path_session_ids),
            "condition_path_omni_model_state": {
                session_id: self._copy_omni_model_state(
                    condition_path_record.omni_model_state
                )
                for session_id, condition_path_record in sorted(
                    record.condition_path_records.items()
                )
            },
            "omni_model_state": self._copy_omni_model_state(record.omni_model_state),
            "srt_last_request_id": record.srt_last_request_id,
            "srt_last_origin_input_len": record.srt_last_origin_input_len,
            "srt_ar_decode_request_count": record.srt_ar_decode_request_count,
            "srt_last_ar_decode_request_id": record.srt_last_ar_decode_request_id,
            "srt_last_ar_decode_origin_input_len": (
                record.srt_last_ar_decode_origin_input_len
            ),
            "srt_last_ar_decode_output_ids": record.srt_last_ar_decode_output_ids,
            "srt_last_ar_decode_text": record.srt_last_ar_decode_text,
            "srt_mm_offsets": record.srt_mm_offsets,
            "srt_executed_request_count": record.srt_executed_request_count,
        }

    def merge_model_state_updates(
        self,
        handle_or_session_id: OmniSessionHandle | str,
        *,
        namespace: str,
        updates: dict[str, Any],
    ) -> None:
        """Merge model-private state that must survive later AR/generation phases."""

        record = self._record_for(handle_or_session_id)
        self._merge_omni_model_state_updates(record, {namespace: dict(updates)})

    def get_model_state(
        self,
        handle_or_session_id: OmniSessionHandle | str,
        *,
        namespace: str,
    ) -> dict[str, Any]:
        record = self._record_for(handle_or_session_id)
        state = record.omni_model_state.get(str(namespace)) or {}
        return self._copy_omni_model_state(state)

    def _record_for(
        self,
        handle_or_session_id: OmniSessionHandle | str,
        *,
        allow_closed: bool = False,
    ) -> OmniSessionRecord:
        """get or build a server-side record for an omni session"""
        session_id = (
            handle_or_session_id.session_id
            if isinstance(handle_or_session_id, OmniSessionHandle)
            else handle_or_session_id
        )
        record = self._records.get(session_id)
        if record is None or (record.closed and not allow_closed):
            raise ValueError(f"Unknown or closed omni session: {session_id}")
        if isinstance(handle_or_session_id, OmniSessionHandle):
            if handle_or_session_id.context_version != record.context_version:
                raise ValueError(
                    "Stale omni session handle: "
                    f"{handle_or_session_id.context_version} != "
                    f"{record.context_version}"
                )
        return record

    def _commit_generated_image_to_srt_session_kv(
        self,
        record: OmniSessionRecord,
        messages: list[OmniInterleavedMessage],
        *,
        request_id: str,
    ) -> None:
        """materialize the generated content into SRT session by build and execute a special req with max_new_tokens=0"""
        if self.session_controller is None or not hasattr(
            self.session_controller, "get"
        ):
            return

        # prepare inputs for ar
        prepared_inputs: list[OmniSRTPreparedInput] = self._prepare_srt_ar_inputs(
            record,
            messages,
            state=record.state,
        )
        total_inputs = sum(
            1 for prepared in prepared_inputs if prepared.condition_path_role is None
        )
        main_input_index = 0
        for prepared in prepared_inputs:
            if prepared.condition_path_role is not None:
                self._append_condition_path_request(record, prepared)
                continue
            main_input_index += 1
            segment_request_id = (
                request_id if total_inputs == 1 else f"{request_id}:s{main_input_index}"
            )
            is_final_segment = main_input_index == total_inputs
            input_ids = prepared.input_ids
            input_text = prepared.input_text
            mm_inputs = prepared.mm_inputs
            # create and adjust a special req:
            req, recv_req = self._create_srt_session_req(
                record,
                request_id=segment_request_id,
                input_ids=input_ids,
                input_text=input_text,
                mm_inputs=mm_inputs,
                max_new_tokens=0,
            )

            if mm_inputs is not None:
                # adjust token-ids with the multimodal_inputs
                SessionController.adjust_mm_offsets(recv_req, req, mm_inputs)
                self._pad_srt_multimodal_input_ids(req, mm_inputs)
                req.extend_image_inputs(mm_inputs)

            srt_request_metadata = self._populate_srt_req_from_prepared_input(
                req,
                record=record,
                recv_req=recv_req,
                prepared=prepared,
                is_final_segment=is_final_segment,
            )

            self._record_srt_req(record, req, request_id=segment_request_id)
            self._attach_srt_request_overrides(
                req, srt_request_metadata=srt_request_metadata
            )

            # execute the special req, but with max_new_tokens=0, no new tokens will be generated, just to:
            # 1. write image context into KV cache by prefilling on the image input
            # 2. capture token binding: OmniSRTKVTokenBinding
            self._execute_srt_req(record, req, state=record.state)

    def _append_condition_path_request(
        self,
        owner_record: OmniSessionRecord,
        prepared: OmniSRTPreparedInput,
        *,
        state: OmniSegmentState | None = None,
    ) -> OmniSessionRecord:
        role = prepared.condition_path_role
        if not role:
            raise RuntimeError(
                "omni condition path request requires a condition path role"
            )
        # each role owns an isolated SRT KV stream for one generation branch
        condition_path_session_id = (
            prepared.condition_path_session_id or f"{owner_record.session_id}:{role}"
        )
        owner_record.condition_path_session_ids.add(condition_path_session_id)
        self._ensure_srt_session(condition_path_session_id)
        condition_path_state = state or owner_record.state
        request_id = (
            f"{owner_record.session_id}:condition_path:{role}:"
            f"{owner_record.context_version + 1}:"
            f"{owner_record.condition_path_request_count + 1}"
        )
        condition_path_record = owner_record.condition_path_records.get(
            condition_path_session_id
        )
        if condition_path_record is None:
            condition_path_record = OmniSessionRecord(
                session_id=condition_path_session_id,
                state=condition_path_state,
                anchor_request_id=request_id,
            )
            owner_record.condition_path_records[condition_path_session_id] = (
                condition_path_record
            )
        condition_path_record.state = condition_path_state
        condition_path_record.anchor_request_id = request_id
        condition_path_record.context_version += 1
        req, recv_req = self._create_srt_session_req(
            condition_path_record,
            request_id=request_id,
            input_ids=prepared.input_ids,
            input_text=prepared.input_text,
            mm_inputs=prepared.mm_inputs,
            max_new_tokens=0,
        )
        if prepared.mm_inputs is not None:

            SessionController.adjust_mm_offsets(recv_req, req, prepared.mm_inputs)
            self._pad_srt_multimodal_input_ids(req, prepared.mm_inputs)
            req.extend_image_inputs(prepared.mm_inputs)
        srt_request_metadata = self._populate_srt_req_from_prepared_input(
            req,
            record=condition_path_record,
            recv_req=recv_req,
            prepared=prepared,
            is_final_segment=True,
        )
        srt_request_metadata["omni_condition_path_owner_session_id"] = (
            owner_record.session_id
        )
        srt_request_metadata["omni_condition_path_role"] = role
        self._record_srt_req(condition_path_record, req, request_id=request_id)
        self._attach_srt_request_overrides(
            req, srt_request_metadata=srt_request_metadata
        )
        self._execute_srt_req(condition_path_record, req, state=condition_path_state)
        condition_path_record.context_length = len(req.origin_input_ids)
        owner_record.condition_path_request_count += 1
        return condition_path_record

    def _pad_srt_multimodal_input_ids(
        self, req: Req, mm_inputs: MultimodalInputs | None
    ) -> None:
        if self.srt_request_executor is None:
            return
        pad_input_ids = getattr(self.srt_request_executor, "pad_input_ids", None)
        if callable(pad_input_ids):
            req.origin_input_ids = pad_input_ids(req.origin_input_ids, mm_inputs)

    def _prepare_srt_ar_inputs(
        self,
        record: OmniSessionRecord,
        messages: list[OmniInterleavedMessage],
        *,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput]:
        """
        Convert message in an OmniRequest to OmniSRTPreparedInput (format recognized in OmniSRT runtime)
        image -> [IMG_START?] + IMG_CONTEXT_TOKEN * N + [IMG_END]
        """
        session_view = self._model_session_view(record)
        custom_inputs = self.model_hooks.prepare_srt_ar_interleaved_inputs(
            session=session_view, messages=messages, state=state
        )
        if custom_inputs is not None:
            return custom_inputs

        prepared_inputs: list[OmniSRTPreparedInput] = []
        for message in messages:
            # OmniRequest -> OmniSRTPreparedInput
            custom_input = self.model_hooks.prepare_srt_ar_message_inputs(
                session=session_view, message=message, state=state
            )
            if custom_input is None:
                raise RuntimeError(
                    f"{self.model_hooks.__class__.__name__} did not prepare omni "
                    f"SRT input for message type {message.type!r}"
                )
            prepared_inputs.extend(custom_input)
        if not prepared_inputs:
            raise RuntimeError("omni SRT prepared inputs must not be empty")
        return prepared_inputs

    def _populate_srt_req_from_prepared_input(
        self,
        req: Req,
        *,
        record: OmniSessionRecord,
        recv_req: Any,
        prepared: OmniSRTPreparedInput,
        is_final_segment: bool,
    ) -> dict[str, Any]:
        """Copy model-prepared fields onto the concrete SRT request."""

        # session.create_req may prepend cached context and strip append-only input
        prefix_len, stripped = self._srt_prefix_and_strip_lengths(
            req, recv_req, prepared
        )
        new_token_count = len(recv_req.input_ids)

        # merge model-side state updates before exposing a request-local snapshot
        srt_request_metadata = dict(prepared.policy_metadata)
        srt_request_metadata.setdefault("omni_srt_added_token_count", new_token_count)
        srt_request_metadata["omni_srt_is_final_segment"] = bool(is_final_segment)
        self._merge_omni_model_state_updates(
            record,
            srt_request_metadata.get("omni_model_state_updates"),
        )
        srt_request_metadata["omni_model_state"] = self._copy_omni_model_state(
            record.omni_model_state
        )

        if prepared.input_embeds is not None:
            # prepared embeds are chunk-relative; SRT req embeds must cover the prefix
            suffix_embeds = prepared.input_embeds[stripped:]
            hidden_size = len(suffix_embeds[0]) if suffix_embeds else 0
            prefix_embeds = [[0.0] * hidden_size for _ in range(prefix_len)]
            req.input_embeds = prefix_embeds + suffix_embeds

        if prepared.replace_embeds is not None:
            # replace positions are chunk-relative and must be shifted into the full req
            replace_positions = prepared.replace_positions
            if replace_positions is None:
                raise RuntimeError(
                    "omni prepared SRT replace_embeds requires replace_positions"
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
            # position ids follow the same prefix/strip alignment as token ids
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

        return srt_request_metadata

    @staticmethod
    def _srt_prefix_and_strip_lengths(
        req: Req,
        recv_req: Any,
        prepared: OmniSRTPreparedInput,
    ) -> tuple[int, int]:
        prefix_len = len(req.origin_input_ids) - len(recv_req.input_ids)
        stripped = len(prepared.input_ids) - len(recv_req.input_ids)
        if prefix_len < 0 or stripped < 0:
            raise RuntimeError(
                "omni prepared SRT input length is inconsistent with session request"
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
    def _install_multidim_mm_positions(req: Req) -> None:
        # adjust mrope positions for qwen-vl series
        mm_inputs = req.multimodal_inputs
        positions = req.custom_position_ids
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

    def _execute_srt_ar_decode_request(
        self,
        record: OmniSessionRecord,
        *,
        max_new_tokens: int | None = None,
        input_ids: list[int] | None = None,
        position_ids: list[int] | tuple[int, ...] | None = None,
        decode_position_id: int | None = None,
        drop_previous_output: bool = False,
        greedy: bool = False,
        model_state_updates: dict[str, Any] | None = None,
    ) -> list[int]:
        """decode with the srt scheduler executor"""
        max_new_tokens = (
            self.srt_ar_decode_max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if max_new_tokens <= 0:
            raise ValueError("omni SRT AR decode requires max_new_tokens > 0")
        input_ids = list(input_ids or [])
        if position_ids is not None and len(position_ids) != len(input_ids):
            raise ValueError(
                "omni SRT AR decode position_ids must match input_ids length: "
                f"{len(position_ids)} != {len(input_ids)}"
            )
        if decode_position_id is not None and input_ids:
            raise ValueError(
                "omni SRT AR decode_position_id is only valid for next-token decode"
            )
        request_id = f"{record.session_id}:d{record.srt_ar_decode_request_count + 1}"
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
                    "omni SRT AR decode input length is inconsistent with session request"
                )
            req.custom_position_ids = list(range(prefix_len)) + [
                int(position) for position in position_ids
            ]
        if decode_position_id is not None:
            req.custom_decode_position_id = int(decode_position_id)
            if not input_ids:
                if not req.origin_input_ids:
                    raise RuntimeError(
                        "omni SRT AR decode_position_id requires a non-empty "
                        "session context"
                    )
                req.custom_position_ids = list(range(len(req.origin_input_ids)))
                req.custom_position_ids[-1] = int(decode_position_id)
        self._record_srt_req(record, req, request_id=request_id)
        policy_metadata = {
            "omni_srt_added_token_count": len(input_ids),
        }
        if position_ids is not None:
            policy_metadata["omni_srt_rope_delta"] = len(input_ids)
        if decode_position_id is not None:
            policy_metadata["omni_srt_decode_position_id"] = int(decode_position_id)
            policy_metadata["omni_srt_position_count"] = int(decode_position_id) + 1
        if model_state_updates is not None:
            policy_metadata["omni_model_state_updates"] = model_state_updates
            self._merge_omni_model_state_updates(record, model_state_updates)
        policy_metadata["omni_model_state"] = self._copy_omni_model_state(
            record.omni_model_state
        )
        self._attach_srt_request_overrides(req, srt_request_metadata=policy_metadata)
        self._execute_srt_req(record, req, state=OmniSegmentState.AR_DECODE)
        record.srt_ar_decode_request_count += 1
        record.srt_last_ar_decode_request_id = request_id
        record.srt_last_ar_decode_origin_input_len = len(req.origin_input_ids)
        record.srt_last_ar_decode_output_ids = list(
            req.output_ids[: req.sampling_params.max_new_tokens]
        )
        record.srt_last_ar_decode_text = self._decode_token_ids(
            record.srt_last_ar_decode_output_ids
        )
        record.context_length = len(req.origin_input_ids) + len(
            record.srt_last_ar_decode_output_ids
        )
        return list(record.srt_last_ar_decode_output_ids)

    def _decode_token_ids(self, token_ids: list[int] | tuple[int, ...]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            return str(decode(list(token_ids)))
        return " ".join(str(token_id) for token_id in token_ids)

    def _create_srt_session_req(
        self,
        record: OmniSessionRecord,
        *,
        request_id: str,
        input_ids: list[int],
        input_text: str,
        mm_inputs: Any | None,
        max_new_tokens: int,
        drop_previous_output: bool = False,
        greedy: bool = False,
    ) -> tuple[Req, TokenizedGenerateReqInput]:
        """create a Req for srt scheduler"""

        session = self.session_controller.get(record.session_id)
        if session is None:
            raise RuntimeError(
                f"SRT session {record.session_id} is not open for omni request"
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
                f"Failed to create SRT omni session request {request_id}: "
                f"{req.to_finish.to_json()}"
            )
        return req, recv_req

    @staticmethod
    def _attach_srt_request_overrides(
        req: Req,
        *,
        srt_request_metadata: dict[str, Any] | None = None,
    ) -> None:
        req.omni_internal_request = True
        metadata = srt_request_metadata or {}
        attention_math_mode = metadata.get("attention_math_mode")
        if attention_math_mode is not None:
            req.attention_math_mode = str(attention_math_mode)
        position_count = metadata.get("omni_srt_position_count")
        if position_count is not None:
            req.omni_srt_position_count = int(position_count)

    def _execute_srt_req(
        self,
        record: OmniSessionRecord,
        req: Req,
        *,
        state: OmniSegmentState,
    ) -> None:
        """execute the AR request through the attached SRT scheduler executor"""
        if self.srt_request_executor is not None:
            self.srt_request_executor.execute_omni_request(
                record=record,
                req=req,
                state=state,
            )
            record.srt_executed_request_count += 1

        if self.srt_request_executor is None or getattr(
            self.srt_request_executor, "finish_request_after_execute", True
        ):
            req.finished_reason = FINISH_LENGTH(len(req.output_ids))

    @staticmethod
    def _record_srt_req(
        record: OmniSessionRecord,
        req: Req,
        *,
        request_id: str,
    ) -> None:
        record.srt_request_count += 1
        record.srt_last_request_id = request_id
        record.srt_last_origin_input_len = len(req.origin_input_ids)
        record.srt_mm_inputs = req.multimodal_inputs
        record.srt_mm_offsets = OmniSessionRuntime._collect_mm_offsets(
            req.multimodal_inputs
        )

    @classmethod
    def _merge_omni_model_state_updates(
        cls,
        record: OmniSessionRecord,
        updates: dict[str, Any] | None,
    ) -> None:
        if not updates:
            return
        if not isinstance(updates, dict):
            raise ValueError("omni model state updates must be a dict")
        for namespace, namespace_updates in updates.items():
            if not isinstance(namespace_updates, dict):
                raise ValueError(
                    "omni model state namespace updates must be dicts: "
                    f"{namespace!r}"
                )
            state = record.omni_model_state.setdefault(str(namespace), {})
            for key, value in namespace_updates.items():
                state[str(key)] = cls._copy_omni_model_state_value(value)

    @classmethod
    def _copy_omni_model_state(cls, state: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): cls._copy_omni_model_state_value(value)
            for key, value in state.items()
        }

    @classmethod
    def _copy_omni_model_state_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._copy_omni_model_state_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._copy_omni_model_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._copy_omni_model_state_value(item) for item in value)
        return value

    @staticmethod
    def _collect_mm_offsets(
        mm_inputs: MultimodalInputs | None,
    ) -> list[tuple[int, int]]:
        if mm_inputs is None:
            return []
        offsets: list[tuple[int, int]] = []
        for item in mm_inputs.mm_items:
            offsets.extend(item.offsets)
        return offsets

    def _ensure_srt_session(self, session_id: str) -> None:
        if self.session_controller is None or session_id in self.session_controller:
            return
        from sglang.srt.managers.io_struct import OpenSessionReqInput

        recv_req = OpenSessionReqInput(
            session_id=session_id,
            capacity_of_str_len=self.capacity_of_str_len,
            # omni sessions retain KV across AR/generation phases for generation-side reads
            streaming=True,
        )
        if self.srt_request_executor is None:
            output = self.session_controller.open(recv_req)
        else:
            output = self.srt_request_executor.open_session_on_scheduler_thread(
                recv_req
            )
        if not getattr(output, "success", False):
            raise RuntimeError(f"Failed to open SRT session for omni: {session_id}")

    def _close_srt_session(self, session_id: str) -> None:
        if self.session_controller is None:
            return
        from sglang.srt.managers.io_struct import CloseSessionReqInput

        if self.srt_request_executor is None:
            if session_id not in self.session_controller:
                return
            self.session_controller.close(CloseSessionReqInput(session_id=session_id))
        else:
            self.srt_request_executor.close_session_on_scheduler_thread(session_id)
