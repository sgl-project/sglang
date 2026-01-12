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
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIBeamSearchMixin:
    """Mixin class for handling beam search in OpenAI-compatible serving endpoints"""

    # These attributes should be provided by the parent class
    tokenizer_manager: TokenizerManager
    reasoning_parser: Optional[str]
    tool_call_parser: Optional[str]

    # ==================== Chat Completion Beam Search Methods ====================

    async def _generate_chat_beam_search_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Beam search streaming path: collect all results and yield as streaming chunks at the end."""
        try:
            all_results = []
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                all_results.append(content)

            if not all_results:
                logger.warning("No results generated for beam search request")
                yield "data: [DONE]\n\n"
                return

            async for chunk in self._stream_chat_beam_search_results(
                all_results, request
            ):
                yield chunk

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_chat_beam_search_results(
        self,
        all_results: List[Any],
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for beam search results."""
        created = int(time.time())
        request_id = None

        # all_results is a flat list where each element is a beam result dict
        # Every n beams belong to the same request (first beam has complete meta_info with id)
        rid_to_beam_results = {}
        cur_rid = None
        for result in all_results:
            meta_info = result.get("meta_info", {})
            rid = meta_info.get("id", "")
            if rid:
                rid_to_beam_results[rid] = [result]
                cur_rid = rid
            else:
                assert cur_rid is not None
                rid_to_beam_results[cur_rid].append(result)

        if not rid_to_beam_results:
            logger.warning("No valid beam search results found in all_results")
            yield "data: [DONE]\n\n"
            return

        all_choices = []
        request_meta_list = []
        choice_index = 0
        for rid, beam_results in rid_to_beam_results.items():
            if not request_id:
                request_id = rid

            request_meta_list.append(beam_results[0])
            choices = self._process_chat_beam_search_results(
                beam_results, request, start_index=choice_index
            )
            all_choices.extend(choices)
            choice_index += len(choices)

        if not all_choices:
            logger.warning(f"No beam search choices generated for request {request_id}")
            yield "data: [DONE]\n\n"
            return

        for choice in all_choices:
            # First chunk: send role
            role_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=choice.index,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                ],
                model=request.model,
            )
            yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n"

            # If there's reasoning content, send it separately
            if choice.message.reasoning_content:
                reasoning_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=choice.index,
                            delta=DeltaMessage(
                                reasoning_content=choice.message.reasoning_content
                            ),
                            finish_reason=None,
                        )
                    ],
                    model=request.model,
                )
                yield f"data: {reasoning_chunk.model_dump_json(exclude_none=True)}\n\n"

            # If there are tool calls, send them
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    tool_call_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=choice.index,
                                delta=DeltaMessage(tool_calls=[tool_call]),
                                finish_reason=None,
                            )
                        ],
                        model=request.model,
                    )
                    yield f"data: {tool_call_chunk.model_dump_json(exclude_none=True)}\n\n"

            # Send content if present
            if choice.message.content:
                content_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=choice.index,
                            delta=DeltaMessage(content=choice.message.content),
                            finish_reason=None,
                            sequence_score=choice.sequence_score,
                        )
                    ],
                    model=request.model,
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

            # Final chunk: send finish_reason
            finish_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=choice.index,
                        delta=DeltaMessage(),
                        finish_reason=choice.finish_reason,
                        matched_stop=choice.matched_stop,
                    )
                ],
                model=request.model,
            )
            yield f"data: {finish_chunk.model_dump_json(exclude_none=True)}\n\n"

        if request.stream_options and request.stream_options.include_usage:
            usage = UsageProcessor.calculate_response_usage(
                request_meta_list,
                1,
                self.tokenizer_manager.server_args.enable_cache_report,
            )
            usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                choices=[],
                model=request.model,
                usage=usage,
            )
            yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _build_chat_beam_search_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> ChatCompletionResponse:
        """Build completion response for beam search results."""
        # Group results by request.n: every n elements belong to one request
        num_prompts = len(ret) // request.n
        assert num_prompts * request.n == len(ret)

        request_meta_list = []
        all_choices = []
        for prompt_idx in range(num_prompts):
            start_idx = prompt_idx * request.n
            end_idx = start_idx + request.n
            beam_results = ret[start_idx:end_idx]

            request_meta_list.append(beam_results[0])
            choices = self._process_chat_beam_search_results(
                beam_results, request, start_index=len(all_choices)
            )
            all_choices.extend(choices)

        usage = UsageProcessor.calculate_response_usage(
            request_meta_list, 1, self.tokenizer_manager.server_args.enable_cache_report
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=all_choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
        )

    def _process_chat_beam_search_results(
        self,
        beam_results: List[Dict[str, Any]],
        request: ChatCompletionRequest,
        start_index: int = 0,
    ) -> List[ChatCompletionResponseChoice]:
        """Convert beam search results to ChatCompletionResponseChoice objects."""
        choices = []
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)

        for beam_idx, beam_result in enumerate(beam_results):
            text = beam_result.get("text", "")

            # Parse reasoning content if applicable
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_enable_thinking_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.debug(f"Reasoning parsing error for beam result: {e}")

            beam_meta_info = beam_result.get("meta_info", {})
            finish_reason = beam_meta_info.get("finish_reason")
            sequence_score = beam_meta_info.get("sequence_score")

            # Handle tool calls
            tool_calls = None
            if (
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
            ):
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )
                # Accumulate tool calls count for next beam result
                if tool_calls:
                    history_tool_calls_cnt += len(tool_calls)

            choice_data = ChatCompletionResponseChoice(
                index=start_index + beam_idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=(reasoning_text if reasoning_text else None),
                ),
                logprobs=None,
                finish_reason=(finish_reason["type"] if finish_reason else "stop"),
                matched_stop=(finish_reason.get("matched") if finish_reason else None),
                hidden_states=None,
                sequence_score=sequence_score,
            )
            choices.append(choice_data)

        return choices

    # ==================== Completion Beam Search Methods ====================

    async def _generate_completion_beam_search_stream(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Beam search streaming path: collect all results and yield as streaming chunks at the end."""
        try:
            all_results = []
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                all_results.append(content)

            if not all_results:
                logger.warning("No results generated for beam search request")
                yield "data: [DONE]\n\n"
                return

            async for chunk in self._stream_completion_beam_search_results(
                all_results, request
            ):
                yield chunk

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"
            yield "data: [DONE]\n\n"

    async def _stream_completion_beam_search_results(
        self,
        all_results: List[Any],
        request: CompletionRequest,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for beam search results."""
        created = int(time.time())
        request_id = None

        # all_results is a flat list where each element is a beam result dict
        # Every n beams belong to the same request (first beam has complete meta_info with id)
        rid_to_beam_results = {}
        cur_rid = None
        for result in all_results:
            meta_info = result.get("meta_info", {})
            rid = meta_info.get("id", "")
            if rid:
                rid_to_beam_results[rid] = [result]
                cur_rid = rid
            else:
                assert cur_rid is not None
                rid_to_beam_results[cur_rid].append(result)

        if not rid_to_beam_results:
            logger.warning("No valid beam search results found in all_results")
            yield "data: [DONE]\n\n"
            return

        all_choices = []
        request_meta_list = []
        choice_index = 0
        for rid, beam_results in rid_to_beam_results.items():
            if not request_id:
                request_id = rid

            request_meta_list.append(beam_results[0])
            choices = self._process_completion_beam_search_results(
                beam_results, request, start_index=choice_index
            )
            all_choices.extend(choices)
            choice_index += len(choices)

        if not all_choices:
            logger.warning(f"No beam search choices generated for request {request_id}")
            yield "data: [DONE]\n\n"
            return

        for choice in all_choices:
            stream_choice = CompletionResponseStreamChoice(
                index=choice.index,
                text=choice.text,
                logprobs=choice.logprobs,
                finish_reason=choice.finish_reason,
                matched_stop=choice.matched_stop,
                sequence_score=choice.sequence_score,
            )
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                object="text_completion",
                choices=[stream_choice],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        if request.stream_options and request.stream_options.include_usage:
            usage = UsageProcessor.calculate_response_usage(
                request_meta_list,
                1,
                self.tokenizer_manager.server_args.enable_cache_report,
            )
            final_usage_chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                object="text_completion",
                choices=[],
                model=request.model,
                usage=usage,
            )
            final_usage_data = final_usage_chunk.model_dump_json(exclude_none=True)
            yield f"data: {final_usage_data}\n\n"

        yield "data: [DONE]\n\n"

    def _build_completion_beam_search_response(
        self,
        request: CompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> CompletionResponse:
        """Build completion response for beam search results."""
        # Group results by request.n: every n elements belong to one request
        num_prompts = len(ret) // request.n
        assert num_prompts * request.n == len(ret)

        request_meta_list = []
        all_choices = []
        for prompt_idx in range(num_prompts):
            start_idx = prompt_idx * request.n
            end_idx = start_idx + request.n
            beam_results = ret[start_idx:end_idx]

            request_meta_list.append(beam_results[0])
            choices = self._process_completion_beam_search_results(
                beam_results, request, start_index=len(all_choices)
            )
            all_choices.extend(choices)

        usage = UsageProcessor.calculate_response_usage(
            request_meta_list, 1, self.tokenizer_manager.server_args.enable_cache_report
        )

        return CompletionResponse(
            id=ret[0]["meta_info"]["id"],
            model=request.model,
            created=created,
            choices=all_choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
        )

    def _process_completion_beam_search_results(
        self,
        beam_results: List[Dict[str, Any]],
        request: CompletionRequest,
        start_index: int = 0,
    ) -> List[CompletionResponseChoice]:
        """Convert beam search results to CompletionResponseChoice objects."""
        choices = []
        for beam_idx, beam_result in enumerate(beam_results):
            text = beam_result.get("text", "")

            # Handle echo for beam results
            if request.echo:
                echo_text = self._get_echo_text(request, start_index)
                text = echo_text + text

            # Get finish_reason from beam's meta_info
            beam_meta_info = beam_result.get("meta_info", {})
            finish_reason = beam_meta_info.get("finish_reason")
            sequence_score = beam_meta_info.get("sequence_score")

            choice_data = CompletionResponseChoice(
                index=start_index + beam_idx,
                text=text,
                logprobs=None,
                finish_reason=(finish_reason["type"] if finish_reason else "stop"),
                matched_stop=(finish_reason.get("matched") if finish_reason else None),
                hidden_states=None,
                sequence_score=sequence_score,
            )
            choices.append(choice_data)

        return choices
