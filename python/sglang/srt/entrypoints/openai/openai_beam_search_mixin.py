# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mixin class for beam search in OpenAI-compatible serving endpoints."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    SglExt,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIBeamSearchMixin:
    """Mixin handling beam search for OpenAI-compatible serving endpoints.

    Beam search yields a single final result (intermediate steps are suppressed
    upstream), so the streaming paths buffer the whole generator then re-emit it
    as fake chunks. The chat and completion variants share all scaffolding here;
    they differ only in the response/choice model classes and in how each beam's
    text is transformed (chat parses reasoning, completion prepends echo text).
    """

    # These attributes should be provided by the parent class
    tokenizer_manager: TokenizerManager
    reasoning_parser: Optional[str]

    # ==================== Shared scaffolding ====================

    async def _generate_beam_search_stream(
        self,
        adapted_request: GenerateReqInput,
        raw_request: Request,
        stream_results,
    ) -> AsyncGenerator[str, None]:
        """Buffer the full generator, then re-emit it via ``stream_results``."""
        try:
            all_results = []
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                all_results.append(content)

            if not all_results:
                yield "data: [DONE]\n\n"
                return

            async for chunk in stream_results(all_results):
                yield chunk

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"
            yield "data: [DONE]\n\n"

    def _collect_beam_search_choices(
        self,
        results: List[Any],
        process_results,
        request,
        sort_results: bool,
        skip_empty: bool,
    ) -> tuple[Optional[str], List[Any], List[Any]]:
        """Flatten packed results into choices, preserving batch order.

        Returns ``(request_id, request_meta_list, all_choices)``. With
        ``sort_results``, results are sorted by index so out-of-order completion
        (asyncio.wait) stays deterministic. With ``skip_empty`` (streaming),
        entries lacking beam_results are dropped; otherwise (non-stream) every
        entry is processed and counted toward usage.
        """
        request_id = None
        request_meta_list = []
        all_choices = []
        packed_results = (
            sorted(results, key=lambda x: x.get("index", 0)) if sort_results else results
        )
        for packed in packed_results:
            beam_results = packed.get("meta_info", {}).get("beam_results")
            if skip_empty and not beam_results:
                continue
            if not request_id:
                request_id = packed["meta_info"].get("id", "")
            request_meta_list.append(packed)
            choices = process_results(
                beam_results or [], request, start_index=len(all_choices)
            )
            all_choices.extend(choices)
        return request_id, request_meta_list, all_choices

    def _build_beam_search_response(
        self,
        request,
        ret: List[Dict[str, Any]],
        created: int,
        process_results,
        response_cls,
    ):
        """Build a non-stream beam search response from packed results."""
        _, request_meta_list, all_choices = self._collect_beam_search_choices(
            ret, process_results, request, sort_results=False, skip_empty=False
        )
        usage = UsageProcessor.calculate_response_usage(
            request_meta_list, 1, self.tokenizer_manager.server_args.enable_cache_report
        )
        return response_cls(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=all_choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
        )

    def _beam_search_usage_chunk(
        self,
        request,
        request_meta_list: List[Any],
        request_id: Optional[str],
        created: int,
        response_cls,
        **extra,
    ) -> Optional[str]:
        """Build the trailing usage SSE chunk, or None when not requested."""
        if not (request.stream_options and request.stream_options.include_usage):
            return None
        usage = UsageProcessor.calculate_response_usage(
            request_meta_list, 1, self.tokenizer_manager.server_args.enable_cache_report
        )
        usage_chunk = response_cls(
            id=request_id,
            created=created,
            choices=[],
            model=request.model,
            usage=usage,
            **extra,
        )
        return f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

    @staticmethod
    def _beam_finish_fields(beam_result: Dict[str, Any]) -> tuple[str, Optional[str], Optional[SglExt]]:
        """Decode the shared per-beam tail: (finish_reason, matched_stop, sglext)."""
        meta = beam_result.get("meta_info", {})
        finish_reason = meta.get("finish_reason")
        sequence_score = meta.get("sequence_score")
        return (
            finish_reason["type"] if finish_reason else "stop",
            finish_reason.get("matched") if finish_reason else None,
            SglExt(sequence_score=sequence_score) if sequence_score is not None else None,
        )

    # ==================== Chat Completion Beam Search Methods ====================

    async def _generate_chat_beam_search_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Buffer beam search results and re-emit them as chat streaming chunks."""
        async for chunk in self._generate_beam_search_stream(
            adapted_request,
            raw_request,
            lambda results: self._stream_chat_beam_search_results(results, request),
        ):
            yield chunk

    async def _stream_chat_beam_search_results(
        self,
        all_results: List[Any],
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Emit buffered chat beam results as fake streaming chunks."""
        created = int(time.time())
        request_id, request_meta_list, all_choices = self._collect_beam_search_choices(
            all_results,
            self._process_chat_beam_search_results,
            request,
            sort_results=True,
            skip_empty=True,
        )

        if not all_choices:
            yield "data: [DONE]\n\n"
            return

        def emit(index, **delta_kwargs):
            stream_choice = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(**delta_kwargs.pop("delta")),
                finish_reason=delta_kwargs.pop("finish_reason", None),
                **delta_kwargs,
            )
            resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                choices=[stream_choice],
                model=request.model,
            )
            return f"data: {resp.model_dump_json(exclude_none=True)}\n\n"

        for choice in all_choices:
            # Role chunk
            yield emit(choice.index, delta={"role": "assistant"})

            # Reasoning content, if any, goes in its own chunk
            if choice.message.reasoning_content:
                yield emit(
                    choice.index,
                    delta={"reasoning_content": choice.message.reasoning_content},
                )

            # Content chunk, if present
            if choice.message.content:
                yield emit(choice.index, delta={"content": choice.message.content})

            # Final chunk: finish_reason and sglext
            yield emit(
                choice.index,
                delta={},
                finish_reason=choice.finish_reason,
                matched_stop=choice.matched_stop,
                sglext=choice.sglext,
            )

        usage_chunk = self._beam_search_usage_chunk(
            request,
            request_meta_list,
            request_id,
            created,
            ChatCompletionStreamResponse,
        )
        if usage_chunk is not None:
            yield usage_chunk

        yield "data: [DONE]\n\n"

    def _build_chat_beam_search_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> ChatCompletionResponse:
        """Build a non-stream chat response from beam search results."""
        return self._build_beam_search_response(
            request,
            ret,
            created,
            self._process_chat_beam_search_results,
            ChatCompletionResponse,
        )

    def _process_chat_beam_search_results(
        self,
        beam_results: List[Dict[str, Any]],
        request: ChatCompletionRequest,
        start_index: int = 0,
    ) -> List[ChatCompletionResponseChoice]:
        """Convert beam results to ChatCompletionResponseChoice objects."""
        choices = []
        for beam_idx, beam_result in enumerate(beam_results):
            text = beam_result.get("text", "")

            # Split reasoning content out of the text when configured
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.debug(f"Reasoning parsing error for beam result: {e}")

            finish_reason, matched_stop, sgl_ext = self._beam_finish_fields(beam_result)
            choices.append(
                ChatCompletionResponseChoice(
                    index=start_index + beam_idx,
                    message=ChatMessage(
                        role="assistant",
                        content=text if text else None,
                        reasoning_content=(reasoning_text if reasoning_text else None),
                    ),
                    logprobs=None,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    hidden_states=None,
                    sglext=sgl_ext,
                )
            )
        return choices

    # ==================== Completion Beam Search Methods ====================

    async def _generate_completion_beam_search_stream(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Buffer beam search results and re-emit them as completion chunks."""
        async for chunk in self._generate_beam_search_stream(
            adapted_request,
            raw_request,
            lambda results: self._stream_completion_beam_search_results(
                results, request
            ),
        ):
            yield chunk

    async def _stream_completion_beam_search_results(
        self,
        all_results: List[Any],
        request: CompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Emit buffered completion beam results as fake streaming chunks."""
        created = int(time.time())
        request_id, request_meta_list, all_choices = self._collect_beam_search_choices(
            all_results,
            self._process_completion_beam_search_results,
            request,
            sort_results=True,
            skip_empty=True,
        )

        if not all_choices:
            yield "data: [DONE]\n\n"
            return

        for choice in all_choices:
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                object="text_completion",
                choices=[
                    CompletionResponseStreamChoice(
                        index=choice.index,
                        text=choice.text,
                        logprobs=choice.logprobs,
                        finish_reason=choice.finish_reason,
                        matched_stop=choice.matched_stop,
                        sglext=choice.sglext,
                    )
                ],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        usage_chunk = self._beam_search_usage_chunk(
            request,
            request_meta_list,
            request_id,
            created,
            CompletionStreamResponse,
            object="text_completion",
        )
        if usage_chunk is not None:
            yield usage_chunk

        yield "data: [DONE]\n\n"

    def _build_completion_beam_search_response(
        self,
        request: CompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> CompletionResponse:
        """Build a non-stream completion response from beam search results."""
        return self._build_beam_search_response(
            request,
            ret,
            created,
            self._process_completion_beam_search_results,
            CompletionResponse,
        )

    def _process_completion_beam_search_results(
        self,
        beam_results: List[Dict[str, Any]],
        request: CompletionRequest,
        start_index: int = 0,
    ) -> List[CompletionResponseChoice]:
        """Convert beam results to CompletionResponseChoice objects."""
        choices = []
        for beam_idx, beam_result in enumerate(beam_results):
            text = beam_result.get("text", "")

            # Prepend the echoed prompt when requested
            if request.echo:
                text = self._get_echo_text(request, start_index) + text

            finish_reason, matched_stop, sgl_ext = self._beam_finish_fields(beam_result)
            choices.append(
                CompletionResponseChoice(
                    index=start_index + beam_idx,
                    text=text,
                    logprobs=None,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    hidden_states=None,
                    sglext=sgl_ext,
                )
            )
        return choices
