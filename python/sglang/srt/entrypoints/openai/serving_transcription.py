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
OpenAI-compatible transcription endpoint handler for Whisper models.
"""

from __future__ import annotations

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
    TranscriptionSegment,
    TranscriptionStreamChoice,
    TranscriptionStreamResponse,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Whisper timestamp token constants
TIMESTAMP_BASE_TOKEN_ID = 50365  # <|0.00|>
TIMESTAMP_BASE_OFFSET = 0.02  # Each token step = 0.02 seconds


class OpenAIServingTranscription(OpenAIServingBase):
    """Handler for /v1/audio/transcriptions requests"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        super().__init__(tokenizer_manager)

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
        # Build sampling params - include language for WhisperProcessor
        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": 448,  # Whisper default max tokens
            "language": request.language,  # Pass to WhisperProcessor for language-specific decoding
        }

        if request.timestamp_granularities:
            sampling_params["timestamp_granularities"] = request.timestamp_granularities

        # For Whisper, we pass audio_data and let the processor handle it
        adapted_request = GenerateReqInput(
            text="",  # Empty text - Whisper processor will set proper decoder tokens
            audio_data=request.audio_data,
            sampling_params=sampling_params,
            stream=request.stream,
            modalities=["audio"],
            routing_key=self.extract_routing_key(raw_request),
        )

        return adapted_request, request

    def _get_audio_duration(self, audio_data: bytes) -> float:
        """Calculate audio duration in seconds."""
        try:
            import soundfile as sf

            info = sf.info(io.BytesIO(audio_data))
            return info.duration
        except Exception as e:
            logger.warning(f"Could not calculate audio duration: {e}")
            return 0.0

    def _parse_segments(
        self, output_ids: List[int], tokenizer
    ) -> tuple[str, List[TranscriptionSegment]]:
        """Parse timestamp tokens from output_ids into segments.

        The decoder prompt ends with <|0.00|>, so the first segment starts at
        t=0.  The model then outputs:
            text_tokens <|end_ts|> [<|start_ts|> text_tokens <|end_ts|> ...]
        Each timestamp token marks the end of the current segment; its value
        also becomes the start of the next segment.
        """
        # Token IDs for special tokens we want to strip from segment text
        eos_token_id = getattr(tokenizer, "eos_token_id", 50257)

        segments = []
        full_text_parts = []
        current_text_tokens = []
        current_start = 0.0  # First segment starts at 0.0 (from prompt <|0.00|>)
        seg_id = 0

        for token_id in output_ids:
            if token_id >= TIMESTAMP_BASE_TOKEN_ID:
                # This is a timestamp token — marks the end of current segment
                timestamp = (token_id - TIMESTAMP_BASE_TOKEN_ID) * TIMESTAMP_BASE_OFFSET

                if current_text_tokens:
                    text = tokenizer.decode(
                        current_text_tokens, skip_special_tokens=True
                    ).strip()
                    if text:
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                start=round(current_start, 2),
                                end=round(timestamp, 2),
                                text=text,
                            )
                        )
                        full_text_parts.append(text)
                        seg_id += 1
                    current_text_tokens = []

                # Next segment starts at this timestamp
                current_start = timestamp

            elif token_id == eos_token_id:
                # Skip end-of-text token
                continue
            else:
                # Regular text token
                current_text_tokens.append(token_id)

        # Handle any trailing text tokens without a closing timestamp
        if current_text_tokens:
            text = tokenizer.decode(
                current_text_tokens, skip_special_tokens=True
            ).strip()
            if text:
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        start=round(current_start, 2),
                        end=round(current_start, 2),
                        text=text,
                    )
                )
                full_text_parts.append(text)

        full_text = " ".join(full_text_parts)
        return full_text, segments

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

        text = ret.get("text", "")
        usage = TranscriptionUsage(seconds=int(math.ceil(request.audio_duration_s)))

        # Build response based on format
        if request.response_format == "text":
            return Response(content=text, media_type="text/plain")

        if request.response_format == "verbose_json":
            output_ids = ret.get("output_ids", [])
            tokenizer = self.tokenizer_manager.tokenizer
            parsed_text, segments = self._parse_segments(output_ids, tokenizer)

            return TranscriptionVerboseResponse(
                language=request.language or "en",
                duration=round(request.audio_duration_s, 2),
                text=parsed_text or text,
                segments=segments,
                usage=usage,
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
