# SPDX-License-Identifier: Apache-2.0
"""SenseNova U1 session adapter for omni orchestration.

U1 keeps AR text, VLM context, and pixel-flow generation inside the same SRT ModelRunner
process for this version. The adapter is model-specific glue; generic request
orchestration lives in `sglang.omni`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sglang.omni.core.interleaved import (
    INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY,
    STREAMED_TEXT_METADATA_KEY,
    TEXT_ROLE_METADATA_KEY,
    TEXT_ROLE_THINK,
    ar_appended_token_positions,
    ar_output_position,
    build_generation_boundary_metadata,
    get_ar_decode_input_position,
    get_ar_next_output_position,
)
from sglang.omni.core.protocol import GeneratedSegment
from sglang.omni.model_adapters.model_policy import (
    OmniModelAppendImageResult,
    OmniModelPolicy,
    OmniModelPrefillResult,
    OmniModelSessionView,
)
from sglang.omni.model_adapters.sensenova_u1.context import (
    U1_EDIT_IMG_CONDITION_ROLE,
    U1_EDIT_UNCONDITION_ROLE,
    U1_IMG_START_TOKEN,
    U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    U1_SPECIAL_TOKENS,
    U1_T2I_CFG_UNCONDITION_ROLE,
    U1ModelStateUpdate,
    U1SpecialTokens,
    _u1_decode_token_ids,
    _u1_eos_token_ids,
    _u1_needs_any_cfg,
    _u1_needs_text_cfg,
    _u1_token_id,
    _u1_tokenize_to_ids,
    build_u1_native_edit_img_condition_prepared_input,
    build_u1_native_edit_prepared_input,
    build_u1_native_edit_uncondition_prepared_input,
    build_u1_native_generated_image_commit_prepared_input,
    build_u1_native_interleave_prepared_input,
    build_u1_native_interleave_text_uncondition_marker_prepared_input,
    build_u1_native_interleave_text_uncondition_prepared_input,
    build_u1_native_t2i_cfg_uncondition_prepared_input,
    build_u1_native_t2i_prepared_input,
    build_u1_native_vlm_prepared_input,
)
from sglang.srt.omni_session.runtime import (
    OmniDecodeResult,
    OmniInterleavedMessage,
    OmniSegmentState,
    OmniSessionRuntime,
    OmniSRTPreparedInput,
    OmniVLMTextGenerationResult,
)
from sglang.srt.omni_session.runtime_types import (
    OmniContextBundle,
    OmniSessionHandle,
)
from sglang.srt.omni_session.session_adapter import SRTBackedOmniSessionAdapter
from sglang.srt.omni_session.srt_executor import OmniSRTSchedulerExecutor

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
        SenseNovaU1SamplingParams,
    )
    from sglang.omni.entrypoints.streaming import OmniStreamSink
    from sglang.srt.managers.scheduler import Scheduler


DEFAULT_U1_INTERLEAVE_DECODE_MAX_NEW_TOKENS = 128
DEFAULT_U1_IMAGE_THINK_MAX_NEW_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class _U1ConditionPathSessionView:
    handle: OmniSessionHandle
    metadata: dict[str, Any]


def build_sensenova_u1_srt_session_adapter(
    *,
    scheduler: "Scheduler",
    srt_request_executor: OmniSRTSchedulerExecutor | None = None,
    srt_ar_decode_max_new_tokens: int | None = None,
) -> "SenseNovaU1SessionAdapter":
    """Build the SRT-owned session adapter for SenseNova U1.

    Omni consumes the returned adapter as a module; SRT owns
    the scheduler/session/runtime wiring here.
    """

    if srt_request_executor is None:
        srt_request_executor = OmniSRTSchedulerExecutor(scheduler)
    if srt_ar_decode_max_new_tokens is None:
        srt_ar_decode_max_new_tokens = DEFAULT_U1_INTERLEAVE_DECODE_MAX_NEW_TOKENS
    session_controller = srt_request_executor.session_controller
    tokenizer = scheduler.tokenizer
    runtime = OmniSessionRuntime(
        model_policy=U1OmniSessionModelPolicy(native_tokenizer=tokenizer),
        session_controller=session_controller,
        srt_request_executor=srt_request_executor,
        tokenizer=tokenizer,
        vocab_size=scheduler.model_config.vocab_size,
        srt_ar_decode_max_new_tokens=srt_ar_decode_max_new_tokens,
    )
    return SenseNovaU1SessionAdapter(runtime)


def _u1_text_looks_like_planning(text: str) -> bool:
    """return true for U1 final-answer text that still reads like planning"""

    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    planning_prefixes = (
        "the user's input",
        "the user has",
        "to respond",
        "my goal",
        "my plan",
        "i will",
        "1.",
        "1)",
        "instruction understanding",
    )
    if normalized.startswith(planning_prefixes):
        return True
    planning_phrases = (
        "my plan",
        "i will",
        "the visual component will",
        "explicit prompt",
        "numbered analysis",
    )
    return any(phrase in normalized[:512] for phrase in planning_phrases)


class U1OmniSessionModelPolicy(OmniModelPolicy):
    """SenseNova U1 policy shell for SRT-owned omni sessions.

    U1 uses pixel-flow generation mechanics; image-generation math stays in the
    generation backend while this policy owns token grammar and session commits.
    """

    generation_kind: str = "pixel_flow"

    def __init__(
        self,
        *,
        native_tokenizer: Any | None = None,
    ) -> None:
        self.native_tokenizer = native_tokenizer
        self.include_t2i_cfg_uncondition = False
        self.include_interleave_text_uncondition = False
        self.include_edit_img_condition = False
        self.include_edit_uncondition = False
        self.native_generation_mode: str | None = None
        self.native_interleave_think_mode = False
        self.special_tokens: U1SpecialTokens = U1_SPECIAL_TOKENS

    def is_image_generation_boundary_token(self, token_id: int) -> bool:
        """translate U1 token grammar into a generic omni image boundary"""

        return token_id == self._image_start_token_id()

    def _image_start_token_id(self) -> int:
        if self.native_tokenizer is None:
            raise RuntimeError("SenseNova U1 image boundary requires a tokenizer")
        return _u1_token_id(self.native_tokenizer, self.special_tokens.img_start)

    def prepare_srt_ar_interleaved_inputs(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        if state != OmniSegmentState.AR_PREFILL or self.native_tokenizer is None:
            return None
        has_image = any(message.type == "image" for message in messages)
        has_text = any(message.type == "text" for message in messages)
        if not has_text:
            return None
        if self.native_generation_mode == "interleave":
            prepared = []
            if self.include_interleave_text_uncondition:
                prepared.append(
                    build_u1_native_interleave_text_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_interleave_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                    think_mode=self.native_interleave_think_mode,
                )
            )
            return prepared
        if not has_image:
            prepared = []
            if self.include_t2i_cfg_uncondition:
                prepared.append(
                    build_u1_native_t2i_cfg_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_t2i_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                    think_mode=self.native_interleave_think_mode,
                )
            )
            return prepared
        if self.native_generation_mode == "edit":
            prepared = []
            if self.include_edit_img_condition:
                prepared.append(
                    build_u1_native_edit_img_condition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        messages=messages,
                        session=session,
                    )
                )
            if self.include_edit_uncondition:
                prepared.append(
                    build_u1_native_edit_uncondition_prepared_input(
                        tokenizer=self.native_tokenizer,
                        session=session,
                    )
                )
            prepared.append(
                build_u1_native_edit_prepared_input(
                    tokenizer=self.native_tokenizer,
                    messages=messages,
                    session=session,
                    think_mode=self.native_interleave_think_mode,
                )
            )
            return prepared
        return [
            build_u1_native_vlm_prepared_input(
                tokenizer=self.native_tokenizer,
                messages=messages,
                session=session,
            )
        ]

    def prepare_srt_ar_message_inputs(
        self,
        *,
        session: OmniModelSessionView,
        message: OmniInterleavedMessage,
        state: OmniSegmentState,
    ) -> list[OmniSRTPreparedInput] | None:
        if self.native_tokenizer is None:
            return None
        if message.type == "text":
            return None
        if message.type == "image":
            if state == OmniSegmentState.APPEND_IMAGE:
                return [
                    build_u1_native_generated_image_commit_prepared_input(
                        tokenizer=self.native_tokenizer,
                        image=message.content,
                        session=session,
                    )
                ]
        return None

    def on_prefill_finished(
        self,
        *,
        session: OmniModelSessionView,
        messages: list[OmniInterleavedMessage],
    ) -> OmniModelPrefillResult:
        return OmniModelPrefillResult(
            added_tokens=self._added_tokens_from_srt_session_view(session)
        )

    def decode_next_segment_with_runtime(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniModelSessionView,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniDecodeResult:
        u1_state = (session.metadata or {}).get("omni_model_state", {}).get("u1", {})
        if bool(u1_state.get("native_interleave_prompt")):
            return self._decode_native_interleave_next_segment(
                runtime=runtime,
                session=session.handle,
                u1_state=u1_state,
                stream_sink=stream_sink,
            )
        if not self._has_generated_image_commit(session):
            return OmniDecodeResult(type="image_marker")
        if self.native_tokenizer is None or runtime is None:
            raise RuntimeError("SenseNova U1 decode requires a tokenizer and runtime")
        if runtime.srt_request_executor is None:
            raise RuntimeError("SenseNova U1 decode requires a SRT request executor")
        max_new_tokens = max(
            1,
            int(runtime.srt_ar_decode_max_new_tokens or 0),
        )
        # the ar-side will automatically stop before next segment
        decoded = runtime.decode(
            session.handle,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return OmniDecodeResult(
            type="text",
            text=decoded.text,
            token_ids=tuple(int(token_id) for token_id in decoded.output_ids),
        )

    def _decode_native_interleave_next_segment(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        u1_state: dict[str, Any],
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniDecodeResult:
        """decode until a boundary / marker / eos is met"""
        if self.native_tokenizer is None or runtime is None:
            return OmniDecodeResult(type="done")
        if bool(u1_state.get("interleave_pending_image_marker")):
            boundary_metadata = dict(
                u1_state.get(INTERLEAVED_GENERATION_BOUNDARY_METADATA_KEY) or {}
            )
            self._merge_runtime_u1_state(
                runtime,
                session,
                U1ModelStateUpdate(
                    interleave_pending_image_marker=False,
                    open_image_marker=True,
                ),
            )
            return OmniDecodeResult(type="image_marker", metadata=boundary_metadata)
        if runtime.srt_request_executor is None:
            return OmniDecodeResult(type="done")

        eos_token_ids = _u1_eos_token_ids(self.native_tokenizer)
        think_end_token_id = _u1_token_id(self.native_tokenizer, "</think>")
        # 1. inside_interleave_think is the current hidden-reasoning text state
        # 2. keep it in session state because image generation can reset adapter temps
        inside_interleave_think = bool(
            u1_state.get("interleave_think_mode")
        ) and not bool(
            u1_state.get("interleave_thinking_done")
        )
        # guarded_post_think_text is U1 final-answer text that may still leak planning
        guarded_post_think_text = bool(
            u1_state.get("interleave_think_mode")
        ) and bool(
            u1_state.get("interleave_thinking_done")
        )
        stream_text_deltas = stream_sink is not None and not guarded_post_think_text

        def decode_text_and_metadata(
            token_ids: list[int],
        ) -> tuple[str, dict[str, Any], bool]:
            text = _u1_decode_token_ids(self.native_tokenizer, token_ids)
            metadata: dict[str, Any] = {}
            if stream_text_deltas:
                metadata[STREAMED_TEXT_METADATA_KEY] = True
            hidden_planning = guarded_post_think_text and _u1_text_looks_like_planning(
                text
            )
            if inside_interleave_think or hidden_planning:
                metadata[TEXT_ROLE_METADATA_KEY] = TEXT_ROLE_THINK
            return text, metadata, hidden_planning

        max_new_tokens = max(
            1,
            int(runtime.srt_ar_decode_max_new_tokens or 0),
        )
        generated_text_ids: list[int] = []
        streamed_text = ""
        # 1. official U1 stores generation_position_start as the next output position
        next_output_position = int(
            u1_state.get(
                "generation_position_start",
                session.context_length,
            )
            or 0
        )
        # 2. SRT decode replays the previous token, so its position is one step behind
        decode_input_position = get_ar_decode_input_position(next_output_position)

        # iteratively decode
        for _ in range(max_new_tokens):
            decoded = runtime.decode(
                session,
                max_new_tokens=1,
                decode_position_id=decode_input_position,
                greedy=True,
                model_state_updates=U1ModelStateUpdate(
                    last_segment_type="interleave",
                    last_source="native_interleave_decode",
                    native_interleave_prompt=True,
                    generation_position_start=get_ar_next_output_position(
                        decode_input_position
                    ),
                ).to_model_state_updates(),
            )
            if not decoded.output_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    U1ModelStateUpdate(
                        generation_position_start=get_ar_next_output_position(
                            decode_input_position
                        )
                    ),
                )
                break
            token_id = int(decoded.output_ids[-1])
            # 3. the sampled token is committed later at the next logical position
            output_position = ar_output_position(decode_input_position)
            if self.is_image_generation_boundary_token(token_id):
                boundary_metadata = build_generation_boundary_metadata(
                    modality="image",
                    token_id=token_id,
                    position_id=output_position,
                )
                # switch to image-generation, return a new segment
                session = runtime.commit_ar_decode_input_token(
                    session,
                    token_id=token_id,
                    position_id=output_position,
                    model_state_updates=U1ModelStateUpdate(
                        last_segment_type="interleave",
                        last_source="native_interleave_image_marker",
                        native_interleave_prompt=True,
                        open_image_marker=True,
                        interleave_pending_image_marker=bool(generated_text_ids),
                        generation_position_start=get_ar_next_output_position(
                            output_position
                        ),
                        generation_boundary_metadata=boundary_metadata,
                    ).to_model_state_updates(),
                )
                self._append_interleave_text_uncondition_marker(
                    runtime=runtime,
                    session=session,
                )
                if generated_text_ids:
                    text, metadata, hidden_planning = decode_text_and_metadata(
                        generated_text_ids
                    )
                    if hidden_planning:
                        return OmniDecodeResult(type="done")
                    return OmniDecodeResult(
                        type="text",
                        text=text,
                        token_ids=tuple(generated_text_ids),
                        metadata=metadata,
                    )
                return OmniDecodeResult(
                    type="image_marker",
                    metadata=boundary_metadata,
                )
            if token_id in eos_token_ids:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    U1ModelStateUpdate(
                        last_segment_type="interleave",
                        last_source="native_interleave_eos",
                        native_interleave_prompt=True,
                        generation_position_start=get_ar_next_output_position(
                            output_position
                        ),
                    ),
                )
                if generated_text_ids:
                    text, metadata, hidden_planning = decode_text_and_metadata(
                        generated_text_ids
                    )
                    if hidden_planning:
                        return OmniDecodeResult(type="done")
                    return OmniDecodeResult(
                        type="text",
                        text=text,
                        token_ids=tuple(generated_text_ids),
                        metadata=metadata,
                    )
                return OmniDecodeResult(type="done")
            generated_text_ids.append(token_id)
            # 1. the generated token becomes the next decode input position
            decode_input_position = output_position
            if stream_text_deltas:
                current_text = _u1_decode_token_ids(
                    self.native_tokenizer,
                    generated_text_ids,
                )
                delta = (
                    current_text[len(streamed_text) :]
                    if current_text.startswith(streamed_text)
                    else current_text
                )
                streamed_text = current_text
                metadata = None
                if inside_interleave_think:
                    metadata = {TEXT_ROLE_METADATA_KEY: TEXT_ROLE_THINK}
                stream_sink.text_delta(delta, token_id=token_id, metadata=metadata)
            if token_id == think_end_token_id:
                self._merge_runtime_u1_state(
                    runtime,
                    session,
                    U1ModelStateUpdate(
                        last_segment_type="interleave",
                        last_source="native_interleave_think_end",
                        native_interleave_prompt=True,
                        interleave_thinking_done=True,
                        generation_position_start=get_ar_next_output_position(
                            output_position
                        ),
                    ),
                )
                # 1. think_end_token_id is the U1 marker that closes hidden reasoning
                # 2. mark later text as final-answer text after this boundary
                metadata = {}
                if stream_sink is not None:
                    metadata[STREAMED_TEXT_METADATA_KEY] = True
                if inside_interleave_think:
                    metadata[TEXT_ROLE_METADATA_KEY] = TEXT_ROLE_THINK
                return OmniDecodeResult(
                    type="text",
                    text=_u1_decode_token_ids(
                        self.native_tokenizer, generated_text_ids
                    ),
                    token_ids=tuple(generated_text_ids),
                    metadata=metadata,
                )

        self._merge_runtime_u1_state(
            runtime,
            session,
            U1ModelStateUpdate(
                last_segment_type="interleave",
                last_source="native_interleave_decode",
                native_interleave_prompt=True,
                generation_position_start=get_ar_next_output_position(
                    decode_input_position
                ),
            ),
        )
        if generated_text_ids:
            text, metadata, hidden_planning = decode_text_and_metadata(
                generated_text_ids
            )
            if hidden_planning:
                return OmniDecodeResult(type="done")
            return OmniDecodeResult(
                type="text",
                text=text,
                token_ids=tuple(generated_text_ids),
                metadata=metadata,
            )
        return OmniDecodeResult(type="done")

    def _append_interleave_text_uncondition_marker(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
    ) -> None:
        if self.native_tokenizer is None:
            return
        condition_path_state = runtime.get_condition_path_model_state(
            session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        u1_state = (condition_path_state or {}).get("u1") or {}
        if not u1_state:
            return
        if bool(u1_state.get("open_image_marker")):
            return
        logical_position = u1_state.get("generation_position_start")
        if logical_position is None:
            return
        prepared = build_u1_native_interleave_text_uncondition_marker_prepared_input(
            tokenizer=self.native_tokenizer,
            session=session,
            logical_position=int(logical_position),
        )
        runtime.append_condition_path_prepared_input(
            session, prepared, state=OmniSegmentState.AR_DECODE
        )

    @staticmethod
    def _merge_runtime_u1_state(
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        updates: U1ModelStateUpdate,
    ) -> None:
        try:
            runtime.merge_model_state_updates(
                session,
                namespace="u1",
                updates=updates.to_state_dict(),
            )
        except ValueError:
            return

    def decode_vlm_text(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniVLMTextGenerationResult:
        """use SRT text decode for U1 image-understanding answers"""
        if runtime is None:
            raise RuntimeError("SenseNova U1 VLM text generation requires SRT runtime")
        u1_state = runtime.get_model_state(session, namespace="u1")
        if self._should_decode_native_image_think(u1_state):
            return self._decode_native_image_think(
                runtime=runtime,
                session=session,
                max_new_tokens=max_new_tokens,
                stream_sink=stream_sink,
            )
        decoded = runtime.decode(
            session,
            max_new_tokens=max_new_tokens,
            greedy=True,
        )
        return OmniVLMTextGenerationResult(
            session=decoded.session,
            text=decoded.text,
            token_ids=decoded.input_ids,
            next_token_ids=decoded.output_ids,
            position_ids=decoded.position_ids,
            streamed_text=False,
        )

    @staticmethod
    def _should_decode_native_image_think(u1_state: dict[str, Any]) -> bool:
        if bool(u1_state.get("open_image_marker")):
            return False
        return bool(u1_state.get("t2i_think_mode") or u1_state.get("edit_think_mode"))

    def _decode_native_image_think(
        self,
        *,
        runtime: OmniSessionRuntime,
        session: OmniSessionHandle,
        max_new_tokens: int,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniVLMTextGenerationResult:
        if self.native_tokenizer is None:
            raise RuntimeError("SenseNova U1 think decode requires a tokenizer")

        eos_token_ids = _u1_eos_token_ids(self.native_tokenizer)
        think_end_token_id = _u1_token_id(self.native_tokenizer, "</think>")
        token_ids: list[int] = []
        streamed_text = ""
        current_session = session
        u1_state = runtime.get_model_state(session, namespace="u1")
        next_output_position = int(
            u1_state.get("generation_position_start", session.context_length) or 0
        )
        decode_input_position = get_ar_decode_input_position(next_output_position)
        for _ in range(max_new_tokens):
            # 1. official t2i/edit think mode starts from an open <think> prefix
            decoded = runtime.decode(
                current_session,
                max_new_tokens=1,
                decode_position_id=decode_input_position,
                greedy=True,
                model_state_updates=U1ModelStateUpdate(
                    last_source="native_image_think_decode",
                    generation_position_start=get_ar_next_output_position(
                        decode_input_position
                    ),
                ).to_model_state_updates(),
            )
            current_session = decoded.session
            if not decoded.output_ids:
                break
            token_id = int(decoded.output_ids[-1])
            if token_id in eos_token_ids:
                break
            token_ids.append(token_id)
            output_position = ar_output_position(decode_input_position)
            decode_input_position = output_position
            if stream_sink is not None:
                current_text = _u1_decode_token_ids(self.native_tokenizer, token_ids)
                delta = (
                    current_text[len(streamed_text) :]
                    if current_text.startswith(streamed_text)
                    else current_text
                )
                streamed_text = current_text
                # 2. t2i/edit think text is decoded before the image boundary reaches coordinator
                stream_sink.text_delta(
                    delta,
                    token_id=token_id,
                    metadata={TEXT_ROLE_METADATA_KEY: TEXT_ROLE_THINK},
                )
            if token_id == think_end_token_id:
                break

        marker_token_ids = _u1_tokenize_to_ids(
            self.native_tokenizer,
            "\n\n" + U1_IMG_START_TOKEN,
            add_special_tokens=False,
        )
        if marker_token_ids:
            marker_position_ids = ar_appended_token_positions(
                previous_decode_input_position=decode_input_position,
                token_count=len(marker_token_ids),
            )
            # 2. append the image marker after </think> so pixel-flow reads the same prefix as official
            current_session = runtime.append_ar_input_tokens(
                current_session,
                token_ids=marker_token_ids,
                position_ids=marker_position_ids,
                model_state_updates=U1ModelStateUpdate(
                    last_source="native_image_think_append_image_marker",
                    open_image_marker=True,
                    generation_position_start=get_ar_next_output_position(
                        marker_position_ids[-1]
                    ),
                ).to_model_state_updates(),
            )

        text = _u1_decode_token_ids(self.native_tokenizer, token_ids)
        return OmniVLMTextGenerationResult(
            session=current_session,
            text=text,
            token_ids=(),
            next_token_ids=tuple(token_ids),
            position_ids=(),
            streamed_text=stream_sink is not None,
        )

    def append_generated_image(
        self,
        *,
        session: OmniModelSessionView,
        image: Any | None,
    ) -> OmniModelAppendImageResult:
        return OmniModelAppendImageResult(
            added_tokens=self._added_tokens_from_srt_session_view(session)
        )

    def close_session(self, *, session_id: str) -> None:
        return None

    def _has_generated_image_commit(self, session: OmniModelSessionView) -> bool:
        model_state = (session.metadata or {}).get("omni_model_state") or {}
        u1_state = model_state.get("u1") or {}
        if bool(u1_state.get("last_generated_image_commit")):
            return True
        return any(
            bool(segment.get("generated_image_commit"))
            for segment in u1_state.get("segments", [])
        )

    def _added_tokens_from_srt_session_view(
        self,
        session: OmniModelSessionView,
    ) -> int:
        previous_length = int(session.handle.context_length or 0)
        srt_length = int(session.srt_last_origin_input_len or 0)
        if srt_length > previous_length:
            return srt_length - previous_length
        return 0


class SenseNovaU1SessionAdapter(SRTBackedOmniSessionAdapter):
    """Pixel-flow U1 session adapter backed by the common SRT omni session runtime."""

    generation_kind: str = "pixel_flow"
    condition_path_roles = {
        "t2i_cfg_uncondition": U1_T2I_CFG_UNCONDITION_ROLE,
        "interleave_text_uncondition": U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        "edit_img_condition": U1_EDIT_IMG_CONDITION_ROLE,
        "edit_uncondition": U1_EDIT_UNCONDITION_ROLE,
    }

    def __init__(
        self,
        runtime: OmniSessionRuntime,
        *,
        max_pre_image_decode_steps: int = 8192,
    ) -> None:
        self.runtime = runtime
        self._session_adapter = SRTBackedOmniSessionAdapter(
            runtime,
            max_pre_image_decode_steps=max_pre_image_decode_steps,
        )

    def _u1_policy(self) -> U1OmniSessionModelPolicy | None:
        policy = self.runtime.model_policy
        if isinstance(policy, U1OmniSessionModelPolicy):
            return policy
        return None

    def prefill_and_decode_to_image_boundary(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        think: bool = False,
        think_max_new_tokens: int | None = None,
        sampling_params: "SenseNovaU1SamplingParams | None" = None,
        session_id: str | None = None,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniContextBundle:
        with self._temporary_generation_settings(sampling_params, think=think):
            mode = (
                None
                if sampling_params is None
                else sampling_params.omni_generation_mode
            )
            session_adapter_think = (
                False
                if sampling_params is not None
                and sampling_params.omni_generation_mode == "interleave"
                else think
            )
            session_adapter_think_max_new_tokens = think_max_new_tokens
            if (
                session_adapter_think
                and session_adapter_think_max_new_tokens is None
                and mode in {"t2i", "edit"}
            ):
                session_adapter_think_max_new_tokens = (
                    DEFAULT_U1_IMAGE_THINK_MAX_NEW_TOKENS
                )
            contexts = self._session_adapter.prefill_and_decode_to_image_boundary(
                messages=messages,
                think=session_adapter_think,
                think_max_new_tokens=session_adapter_think_max_new_tokens,
                sampling_params=sampling_params,
                session_id=session_id,
                stream_sink=stream_sink,
            )
        return contexts

    @contextmanager
    def _temporary_generation_settings(
        self,
        sampling_params: "SenseNovaU1SamplingParams | None",
        *,
        think: bool,
    ):
        policy = self._u1_policy()
        mode = None if sampling_params is None else sampling_params.omni_generation_mode
        if policy is not None:
            old_cfg = policy.include_t2i_cfg_uncondition
            old_interleave_text_uncondition = policy.include_interleave_text_uncondition
            old_edit_img_condition = policy.include_edit_img_condition
            old_edit_uncondition = policy.include_edit_uncondition
            old_mode = policy.native_generation_mode
            old_interleave_think_mode = policy.native_interleave_think_mode
            needs_cfg = _u1_needs_any_cfg(sampling_params)
            cfg_text_scale = (
                1.0
                if sampling_params is None
                else float(sampling_params.cfg_text_scale)
            )
            cfg_img_scale = (
                1.0 if sampling_params is None else float(sampling_params.cfg_img_scale)
            )
            policy.include_t2i_cfg_uncondition = (
                _u1_needs_text_cfg(sampling_params)
                and mode not in {"edit", "interleave"}
            ) or (mode == "interleave" and cfg_img_scale != 1.0)
            policy.include_interleave_text_uncondition = (
                mode == "interleave" and _u1_needs_text_cfg(sampling_params)
            )
            policy.include_edit_img_condition = (
                mode == "edit"
                and needs_cfg
                and (cfg_img_scale == 1.0 or cfg_text_scale != cfg_img_scale)
            )
            policy.include_edit_uncondition = (
                mode == "edit" and needs_cfg and cfg_img_scale != 1.0
            )
            policy.native_generation_mode = mode
            policy.native_interleave_think_mode = bool(think)
        try:
            yield
        finally:
            if policy is not None:
                policy.include_t2i_cfg_uncondition = old_cfg
                policy.include_interleave_text_uncondition = (
                    old_interleave_text_uncondition
                )
                policy.include_edit_img_condition = old_edit_img_condition
                policy.include_edit_uncondition = old_edit_uncondition
                policy.native_generation_mode = old_mode
                policy.native_interleave_think_mode = old_interleave_think_mode

    def commit_generated_segment(
        self,
        *,
        contexts: OmniContextBundle,
        segment: GeneratedSegment,
    ) -> None:
        self._session_adapter.commit_generated_segment(
            contexts=contexts,
            segment=segment,
        )
        self._commit_interleave_text_uncondition_path(
            contexts=contexts,
            segment=segment,
        )

    def finish_generated_segment_turn(
        self,
        *,
        contexts: OmniContextBundle,
    ) -> None:
        """close a persistent U1 assistant turn after generated image commit"""
        policy = self._u1_policy()
        tokenizer = None if policy is None else policy.native_tokenizer
        if tokenizer is None or contexts.full.session is None:
            return
        u1_state = self.runtime.get_model_state(
            contexts.full.session,
            namespace="u1",
        )
        position_id = int(
            u1_state.get(
                "generation_position_start",
                contexts.full.session.context_length,
            )
        )
        token_id = _u1_token_id(tokenizer, "<|im_end|>")
        # 1. append <|im_end|> at the U1 logical position after generated </img>
        session = self.runtime.commit_ar_decode_input_token(
            contexts.full.session,
            token_id=token_id,
            position_id=position_id,
            model_state_updates=U1ModelStateUpdate(
                last_segment_type="interleave",
                last_source="native_interleave_turn_end",
                native_interleave_prompt=True,
                open_image_marker=False,
                interleave_pending_image_marker=False,
                generation_position_start=position_id + 1,
            ).to_model_state_updates(),
        )
        # 2. keep all context paths attached to the closed full session
        contexts.full.request_id = session.anchor_request_id
        contexts.full.token_count = session.context_length
        contexts.full.session = session
        contexts.text_cfg.session = session
        contexts.image_cfg.session = session

    def release(self, contexts: OmniContextBundle) -> None:
        self._session_adapter.release(contexts)

    def continue_ar_decode(
        self,
        *,
        contexts: OmniContextBundle,
        stream_sink: OmniStreamSink | None = None,
    ) -> OmniDecodeResult:
        return self._session_adapter.continue_ar_decode(
            contexts=contexts,
            stream_sink=stream_sink,
        )

    def _commit_interleave_text_uncondition_path(
        self,
        *,
        contexts: OmniContextBundle,
        segment: GeneratedSegment,
    ) -> None:
        policy = self._u1_policy()
        tokenizer = None if policy is None else policy.native_tokenizer
        if tokenizer is None or contexts.full.session is None:
            return
        image = segment.commit_image
        if image is None:
            return
        condition_path_handle = self.runtime.get_condition_path_handle(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if condition_path_handle is None:
            return
        condition_path_state = self.runtime.get_condition_path_model_state(
            contexts.full.session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        )
        if not ((condition_path_state or {}).get("u1") or {}).get("open_image_marker"):
            return
        condition_path_session = _U1ConditionPathSessionView(
            handle=condition_path_handle,
            metadata={"omni_model_state": condition_path_state},
        )
        prepared = build_u1_native_generated_image_commit_prepared_input(
            tokenizer=tokenizer,
            image=image,
            session=condition_path_session,
        )
        prepared.condition_path_role = U1_INTERLEAVE_TEXT_UNCONDITION_ROLE
        prepared.condition_path_session_id = condition_path_handle.session_id
        self.runtime.append_condition_path_prepared_input(
            contexts.full.session,
            prepared,
            state=OmniSegmentState.APPEND_IMAGE,
        )

    def generate_vlm_answer(
        self,
        *,
        messages: list[OmniInterleavedMessage | dict[str, Any]],
        max_new_tokens: int,
    ) -> OmniVLMTextGenerationResult:
        """delegate U1 VLM QA to the SRT-backed adapter"""
        return self._session_adapter.generate_vlm_answer(
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
