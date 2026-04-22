# Copyright 2025 SGLang Team
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
"""
OpenAI-compatible transcription endpoint handler for audio ASR models.

New ASR models are supported by subclassing ``TranscriptionAdapter`` and
registering via the ``@register_transcription_adapter`` decorator.
See ``transcription_adapters/`` for built-in implementations.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.entrypoints.openai.protocol import (
    DeltaMessage,
    ErrorResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionStreamChoice,
    TranscriptionStreamResponse,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    split_audio_chunks,
)
from sglang.srt.entrypoints.openai.transcription_adapters import resolve_adapter
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class OpenAIServingTranscription(OpenAIServingBase):
    """Handler for /v1/audio/transcriptions requests"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        super().__init__(tokenizer_manager)
        model_config = tokenizer_manager.model_config
        self._adapter = resolve_adapter(
            getattr(model_config.hf_config, "architectures", [])
        )

    def _request_id_prefix(self) -> str:
        return "trsc-"

    def _validate_request(self, request: TranscriptionRequest) -> Optional[str]:
        """Validate transcription request."""
        # Validation is done in the route handler for form data
        return None

    def _convert_to_internal_request(
        self,
        request: TranscriptionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, TranscriptionRequest]:
        """Convert transcription request to internal format."""
        sampling_params = self._adapter.build_sampling_params(request)
        adapted_request = GenerateReqInput(
            text="",  # Empty text — the multimodal processor sets proper decoder/prompt tokens
            audio_data=request.audio_data,
            sampling_params=sampling_params,
            stream=request.stream,
            modalities=["audio"],
            routing_key=self.extract_routing_key(raw_request),
        )

        return adapted_request, request

    @staticmethod
    def _get_audio_duration(audio_data: bytes) -> float:
        """Calculate audio duration in seconds."""
        try:
            import soundfile as sf

            info = sf.info(io.BytesIO(audio_data))
            return info.duration
        except Exception as e:
            logger.warning(f"Could not calculate audio duration: {e}")
            return 0.0

    async def create_transcription(
        self,
        audio_data: bytes,
        model: str,
        language: Optional[str],
        response_format: str,
        temperature: float,
        stream: bool,
        raw_request: Request,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Union[
        TranscriptionResponse,
        TranscriptionVerboseResponse,
        StreamingResponse,
        Response,
        ORJSONResponse,
    ]:
        """Main entry point for transcription requests."""
        # Calculate audio duration for usage reporting
        audio_duration_s = self._get_audio_duration(audio_data)

        # Build request
        request = TranscriptionRequest(
            audio_data=audio_data,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            stream=stream,
            audio_duration_s=audio_duration_s,
        )

        # Use the base class handle_request pattern
        return await self.handle_request(request, raw_request)

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> Union[
        TranscriptionResponse,
        TranscriptionVerboseResponse,
        ErrorResponse,
        ORJSONResponse,
        Response,
    ]:
        """Handle non-streaming transcription request."""
        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        text = self._adapter.postprocess_text(ret.get("text", ""))
        usage = TranscriptionUsage(seconds=int(math.ceil(request.audio_duration_s)))

        # Build response based on format
        if request.response_format == "text":
            return Response(content=text, media_type="text/plain")

        if request.response_format == "verbose_json":
            tokenizer = self.tokenizer_manager.tokenizer
            return self._adapter.build_verbose_response(
                request, text, ret, tokenizer, usage
            )

        # Default JSON format
        return TranscriptionResponse(text=text, usage=usage)

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming transcription request."""
        if self._adapter.supports_chunked_streaming:
            # No background abort_task: each chunk is a separate request;
            # client disconnection is detected via is_disconnected() in the loop.
            return StreamingResponse(
                self._generate_chunked_asr_stream(
                    adapted_request, request, raw_request
                ),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            self._generate_transcription_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_transcription_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming transcription response."""
        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        stream_buffer = ""

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Calculate delta (new text since last chunk)
                current_text = content.get("text", "")
                delta = current_text[len(stream_buffer) :]
                stream_buffer = current_text

                # Send content delta if there's new text
                if delta:
                    choice_data = TranscriptionStreamChoice(
                        delta=DeltaMessage(content=delta),
                        finish_reason=None,
                    )
                    chunk = TranscriptionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[choice_data],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Send finish reason when done
                if finish_reason_type:
                    choice_data = TranscriptionStreamChoice(
                        delta=DeltaMessage(),
                        finish_reason=finish_reason_type,
                    )
                    chunk = TranscriptionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model,
                        choices=[choice_data],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _generate_chunked_asr_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Chunk-based streaming for ASR with prefix rollback.

        Audio is split into chunks and each chunk is processed as an
        independent request. Partial transcripts are emitted via SSE
        with prefix rollback to reduce boundary jitter.

        TODO:
        - Token-level streaming within chunks (stream=True)
        - Encoder window caching across chunks
        - Cross-chunk KV cache reuse
        - WebSocket endpoint for real-time audio input
        """
        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        state = StreamingASRState(**self._adapter.chunked_streaming_config)
        first_word = True

        try:
            chunks = split_audio_chunks(request.audio_data, state.chunk_size_sec)

            for i, chunk_audio in enumerate(chunks):
                if await raw_request.is_disconnected():
                    logger.info("[streaming_asr] client disconnected, stopping")
                    break
                is_last = i == len(chunks) - 1
                prompt = self._adapter.prompt_template + state.get_prefix_text()

                chunk_request = GenerateReqInput(
                    text=prompt,
                    audio_data=chunk_audio,
                    sampling_params=adapted_request.sampling_params,
                    stream=False,
                    modalities=["audio"],
                    routing_key=self.extract_routing_key(raw_request),
                )

                try:
                    ret = None
                    async for ret in self.tokenizer_manager.generate_request(
                        chunk_request, raw_request
                    ):
                        break
                except asyncio.CancelledError:
                    raise
                except ValueError as e:
                    logger.warning(
                        "[streaming_asr] chunk %d failed with ValueError: %s", i, e
                    )
                    continue

                if ret is None:
                    logger.warning("[streaming_asr] empty response for chunk %d", i)
                    continue

                text = self._adapter.postprocess_text(ret.get("text", ""))

                if is_last:
                    state.full_transcript = text
                    delta = state.finalize()
                else:
                    delta = state.update(text)

                if delta:
                    for word in delta.split(" "):
                        if not word:
                            continue
                        content = word if first_word else " " + word
                        first_word = False
                        chunk_resp = TranscriptionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model,
                            choices=[
                                TranscriptionStreamChoice(
                                    delta=DeltaMessage(content=content),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk_resp.model_dump_json()}\n\n"

            # Send final stop
            chunk_resp = TranscriptionStreamResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=[
                    TranscriptionStreamChoice(
                        delta=DeltaMessage(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {chunk_resp.model_dump_json()}\n\n"

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("[streaming_asr] unrecoverable error")
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"
