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

from fastapi import Request, WebSocket
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from sglang.srt.entrypoints.openai.audio_chunking import split_audio_energy_aware
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
from sglang.srt.entrypoints.openai.realtime import (
    handle_realtime_transcription,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    needs_space,
    process_asr_chunk,
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
        # Cap concurrent /v1/realtime sessions. The Semaphore is bound to the
        # event loop on first acquire (uvicorn's loop in normal serving).
        self._session_semaphore = asyncio.Semaphore(
            tokenizer_manager.server_args.asr_max_concurrent_sessions
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
        if getattr(request, "_fused_autodetect", False):
            sampling_params = self._adapter.build_fused_autodetect_params(request)
        else:
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
        except Exception:
            # soundfile can't parse some containers (e.g. mp3 on older
            # libsndfile builds); fall back to a full decode.
            try:
                from sglang.srt.utils import load_audio

                audio = load_audio(audio_data, sr=16000, mono=True)
                return audio.shape[-1] / 16000.0
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
        # Calculate audio duration for usage reporting. Run in a thread:
        # the fallback path decodes the full file and would block the
        # event loop on long inputs.
        audio_duration_s = await asyncio.to_thread(self._get_audio_duration, audio_data)

        # Audio longer than the model's encoder window (30 s for Whisper)
        # would be silently truncated by the feature extractor. Split it
        # into chunks cut at low-energy points (pauses) so the cut never
        # lands mid-word; each chunk is transcribed as an independent
        # request and the texts are stitched back in order.
        audio_chunks: Optional[List[bytes]] = None
        chunk_offsets_s: Optional[List[float]] = None
        max_clip_s = self._adapter.max_audio_clip_s
        if max_clip_s is not None and audio_duration_s > max_clip_s:
            split_error = (
                f"Failed to split audio longer than {max_clip_s:g} seconds into "
                "supported chunks."
            )
            try:
                # In a thread: decodes + re-encodes the whole file, which
                # would otherwise block the event loop on long inputs.
                audio_chunks, chunk_offsets_s = await asyncio.to_thread(
                    split_audio_energy_aware, audio_data, max_clip_s
                )
            except Exception as e:
                logger.warning(
                    "Failed to split %.1fs audio into chunks of <=%ss: %s",
                    audio_duration_s,
                    max_clip_s,
                    e,
                )
                return self.create_error_response(split_error)
            else:
                if (
                    not audio_chunks
                    or len(audio_chunks) <= 1
                    or chunk_offsets_s is None
                    or len(chunk_offsets_s) != len(audio_chunks)
                ):
                    logger.error(
                        "Audio splitter returned an invalid result for %.1fs audio: "
                        "%d chunks and %s offsets",
                        audio_duration_s,
                        len(audio_chunks or []),
                        (
                            "no"
                            if chunk_offsets_s is None
                            else str(len(chunk_offsets_s))
                        ),
                    )
                    return self.create_error_response(split_error)
                logger.info(
                    "Split %.1fs audio into %d chunks of <=%ss for transcription",
                    audio_duration_s,
                    len(audio_chunks),
                    max_clip_s,
                )

        # When language is not specified and the adapter supports detection,
        # use a single fused request: SGLang's structured generation (regex)
        # constrains the first 3 decode tokens to the forced prefix while
        # allowing free transcription afterwards — one encoder pass, no
        # extra round-trip. The adapter picks the regex variant based on
        # whether timestamps were requested, so fused covers all four
        # combinations of (stream, timestamp_granularities):
        #   * non-streaming:     parse_fused_output strips the prefix and
        #                        scrubs trailing/embedded special tokens.
        #   * streaming:         the handler buffers until the sentinel,
        #                        re-anchors, and scrubs each delta via
        #                        adapter.strip_special_tokens.
        # verbose_json segment timing still comes from _parse_segments
        # over output_ids, which is unaffected by the string-level scrub.
        use_fused = language is None and self._adapter.supports_language_detection

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
        if use_fused:
            request._fused_autodetect = True
            # Stash the variant alongside the flag so the adapter dispatch in
            # parse_fused_output and the build_fused_autodetect_params regex
            # selection see the same boolean — and we don't recompute it on
            # every cumulative-text snapshot in streaming.
            request._fused_ts_variant = bool(timestamp_granularities)
        if audio_chunks is not None and len(audio_chunks) > 1:
            request._audio_chunks = audio_chunks
            request._chunk_offsets_s = chunk_offsets_s

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
        if getattr(request, "_audio_chunks", None):
            return await self._handle_chunked_non_streaming_request(
                adapted_request, request, raw_request
            )

        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        text = self._finalize_text(request, ret.get("text", ""))

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

    def _finalize_text(
        self, request: TranscriptionRequest, raw_text: str, strip: bool = True
    ) -> str:
        """Postprocess one generation's raw text into user-visible text.

        For fused auto-detect requests, parse_fused_output returns the
        scrubbed user-visible text and the detected language is recorded on
        the request (first non-empty parsed chunk wins for chunked audio). On
        parse failure (FSM abort, truncation) it returns (None, None) and we
        fall back to a best-effort scrub — the language stays unset rather
        than reporting a bogus detection.

        ``strip=False`` keeps the model-emitted boundary whitespace in the
        fused path; chunk stitching needs it as the natural separator.
        """
        text = self._adapter.postprocess_text(raw_text)
        if not getattr(request, "_fused_autodetect", False):
            return text
        lang, visible = self._adapter.parse_fused_output(
            text,
            ts_variant=getattr(request, "_fused_ts_variant", False),
            strip=strip,
        )
        if visible is None:
            logger.warning(
                "Fused auto-detect parse failed on non-streaming response; "
                "falling back to raw-text scrub."
            )
            return self._adapter.strip_special_tokens(text)
        if lang is not None and visible.strip() and request.language is None:
            request.language = lang
            logger.info("Auto-detected language: '%s'", lang)
        return visible

    def _build_chunk_request(
        self,
        adapted_request: GenerateReqInput,
        chunk_audio: bytes,
        stream: bool,
    ) -> GenerateReqInput:
        """Clone the adapted request for one audio chunk.

        ``sampling_params`` must be a fresh dict per chunk: the multimodal
        processor pops transcription-level keys (language,
        timestamp_granularities, the fused-autodetect flag) out of it while
        building each chunk's decoder prompt.
        """
        sampling_params = adapted_request.sampling_params
        assert isinstance(sampling_params, dict)
        chunk_request = GenerateReqInput(
            text="",
            audio_data=chunk_audio,
            sampling_params=dict(sampling_params),
            stream=stream,
            modalities=["audio"],
            routing_key=adapted_request.routing_key,
        )
        chunk_request.received_time = adapted_request.received_time
        return chunk_request

    def _abort_chunk_requests(self, chunk_requests: List[GenerateReqInput]) -> None:
        """Abort chunk generations engine-side.

        The scheduler keeps decoding until a request is aborted by rid (a
        no-op for chunks that already finished). A request whose rid has not
        been assigned yet cannot be aborted by the immediate pass, so a second
        abort fires after the dispatch window — the same approach as
        ``TokenizerManager.create_abort_task``.
        """

        def abort_assigned_rids():
            # Read rids at call time: a chunk task that had not started
            # executing during the first pass gets its rid assigned later,
            # so the delayed pass must not reuse an earlier snapshot.
            for chunk_request in chunk_requests:
                if isinstance(chunk_request.rid, str):
                    self.tokenizer_manager.abort_request(chunk_request.rid)

        abort_assigned_rids()

        async def abort_after_dispatch_window():
            await asyncio.sleep(2)
            abort_assigned_rids()

        asyncio.create_task(abort_after_dispatch_window())

    async def _handle_chunked_non_streaming_request(
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
        """Transcribe pre-split long audio (duration > max_audio_clip_s).

        Each chunk is an independent generation. Chunks run sequentially so
        one user-controlled upload cannot fan out into an unbounded number of
        tokenizer/GPU requests. Results are stitched in chunk (= audio) order.
        """
        chunk_requests = [
            self._build_chunk_request(adapted_request, chunk_audio, stream=False)
            for chunk_audio in request._audio_chunks
        ]
        rets = []
        try:
            for chunk_request in chunk_requests:
                ret = await self.tokenizer_manager.generate_request(
                    chunk_request, raw_request
                ).__anext__()
                rets.append(ret)
        except BaseException as e:
            # Abort the in-flight request on failures or parent cancellation.
            # Later chunks have not been dispatched.
            self._abort_chunk_requests(chunk_requests)
            if isinstance(e, ValueError):
                return self.create_error_response(str(e))
            raise

        fused = getattr(request, "_fused_autodetect", False)
        # Plain in-order concatenation: each chunk's model-emitted boundary
        # whitespace (a leading space for spaced scripts, nothing for
        # spaceless scripts like zh/ja/th) is the correct seam separator,
        # so the fused path keeps it (strip=False) instead of inventing
        # one. Only the full fused text is trimmed at the ends, matching
        # the single-request fused response.
        texts = [
            self._finalize_text(request, ret.get("text", ""), strip=False)
            for ret in rets
        ]
        text = "".join(texts)
        if fused:
            text = text.strip()

        usage = TranscriptionUsage(seconds=int(math.ceil(request.audio_duration_s)))

        if request.response_format == "text":
            return Response(content=text, media_type="text/plain")

        if request.response_format == "verbose_json":
            return self._adapter.build_verbose_response_chunked(
                request,
                text,
                rets,
                request._chunk_offsets_s,
                self.tokenizer_manager.tokenizer,
                usage,
            )

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
        if getattr(request, "_audio_chunks", None):
            # Long audio pre-split into chunks, transcribed sequentially.
            # No background abort_task: the in-flight chunk is aborted in
            # the generator's finally on teardown, and disconnect is checked
            # between chunks.
            return StreamingResponse(
                self._generate_long_audio_stream(adapted_request, request, raw_request),
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
        """Generate streaming transcription response.

        In fused auto-detect mode, each cumulative-text snapshot is passed
        through ``parse_fused_output`` — which returns ``(None, None)``
        while the forced prefix is still arriving and ``(lang, visible)``
        once it's in. ``visible`` is already stripped of the prefix and
        scrubbed of embedded special tokens, and it grows monotonically
        across snapshots, so deltas are a plain suffix slice.
        """
        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        visible_buffer = ""

        fused_mode = getattr(request, "_fused_autodetect", False)
        ts_variant = getattr(request, "_fused_ts_variant", False)
        # When ``incremental_streaming_output`` is enabled, each chunk's
        # ``content["text"]`` is the new delta from the detokenizer, not
        # the cumulative text. Always reconstruct cumulative text locally
        # so the rest of the loop (prefix parse + visible-buffer slice)
        # works uniformly under either mode.
        incremental = getattr(
            self.tokenizer_manager.server_args,
            "incremental_streaming_output",
            False,
        )
        cumulative_text = ""

        try:
            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                chunk_text = content.get("text", "")
                if incremental:
                    cumulative_text += chunk_text
                else:
                    cumulative_text = chunk_text

                if fused_mode:
                    lang, visible = self._adapter.parse_fused_output(
                        cumulative_text, ts_variant=ts_variant
                    )
                    if visible is None:
                        # Prefix not yet locatable. Keep buffering until the
                        # stream ends.
                        if not finish_reason_type:
                            continue
                        # Stream ended before the forced prefix was parseable —
                        # emit an SSE error frame so the client can distinguish
                        # this from "silent audio, zero transcription" and raise
                        # a real error instead of quietly succeeding.
                        logger.warning(
                            "Fused auto-detect stream finished before prefix "
                            "was parseable; returning detection-failed error."
                        )
                        error = self.create_streaming_error_response(
                            "language auto-detect failed: forced-prefix sentinel "
                            "was not produced before stream end"
                        )
                        yield f"data: {error}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if (
                        lang is not None
                        and visible.strip()
                        and request.language is None
                    ):
                        request.language = lang
                        logger.info("Auto-detected language: '%s'", lang)
                else:
                    visible = cumulative_text

                delta = visible[len(visible_buffer) :]
                visible_buffer = visible

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

    async def _generate_long_audio_stream(
        self,
        adapted_request: GenerateReqInput,
        request: TranscriptionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Stream transcription of long audio pre-split into chunks.

        Chunks are transcribed sequentially (one streaming request at a
        time, like ``_generate_chunked_asr_stream``), so the client sees
        the transcript in audio order with a single finish frame after the
        last chunk. The first abnormal chunk finish_reason (length/abort)
        wins so a truncated non-final chunk isn't masked by later chunks
        stopping cleanly. Fused auto-detect applies per chunk; the reported
        language is the first non-empty chunk's detection. Disconnect is
        checked between chunks, and the in-flight chunk is aborted on teardown.
        ``request._chunk_offsets_s`` is unused here — only the non-streaming
        verbose_json path needs segment timing.
        """
        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        fused_mode = getattr(request, "_fused_autodetect", False)
        ts_variant = getattr(request, "_fused_ts_variant", False)
        incremental = getattr(
            self.tokenizer_manager.server_args,
            "incremental_streaming_output",
            False,
        )

        def _frame(delta: Optional[str], finish_reason: Optional[str] = None) -> str:
            chunk = TranscriptionStreamResponse(
                id=request_id,
                created=created_time,
                model=model,
                choices=[
                    TranscriptionStreamChoice(
                        delta=DeltaMessage(content=delta) if delta else DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
            )
            return f"data: {chunk.model_dump_json()}\n\n"

        finish_reason_type = None
        in_flight: Optional[GenerateReqInput] = None
        emitted_text = False
        try:
            for chunk_audio in request._audio_chunks:
                if await raw_request.is_disconnected():
                    break
                in_flight = self._build_chunk_request(
                    adapted_request, chunk_audio, stream=True
                )
                cumulative_text = ""
                visible_buffer = ""
                chunk_finish_reason = None
                strip_chunk = not emitted_text
                async for content in self.tokenizer_manager.generate_request(
                    in_flight, raw_request
                ):
                    finish_reason = content["meta_info"]["finish_reason"]
                    chunk_finish_reason = (
                        finish_reason["type"] if finish_reason else None
                    )

                    chunk_text = content.get("text", "")
                    if incremental:
                        cumulative_text += chunk_text
                    else:
                        cumulative_text = chunk_text

                    if fused_mode:
                        # Strip the first chunk that emits visible text (no
                        # leading space at stream start, like the single-request
                        # path). Later visible chunks keep their model-emitted
                        # boundary whitespace, which is the correct seam
                        # separator for spaced and spaceless scripts alike.
                        lang, visible = self._adapter.parse_fused_output(
                            cumulative_text,
                            ts_variant=ts_variant,
                            strip=strip_chunk,
                        )
                        if visible is None:
                            if not chunk_finish_reason:
                                continue
                            logger.warning(
                                "Fused auto-detect stream finished before prefix "
                                "was parseable; returning detection-failed error."
                            )
                            error = self.create_streaming_error_response(
                                "language auto-detect failed: forced-prefix "
                                "sentinel was not produced before stream end"
                            )
                            yield f"data: {error}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        if (
                            lang is not None
                            and visible.strip()
                            and request.language is None
                        ):
                            request.language = lang
                            logger.info("Auto-detected language: '%s'", lang)
                    else:
                        visible = cumulative_text

                    delta = visible[len(visible_buffer) :]
                    visible_buffer = visible
                    if delta:
                        yield _frame(delta)
                        emitted_text = True

                in_flight = None
                if finish_reason_type in (None, "stop") and chunk_finish_reason:
                    finish_reason_type = chunk_finish_reason

            yield _frame(None, finish_reason=finish_reason_type or "stop")
        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"
        finally:
            # Abort the chunk still decoding if the stream is torn down
            # (client disconnect / generator close) so it doesn't keep
            # running in the scheduler. Only one chunk is ever in flight.
            if in_flight is not None and isinstance(in_flight.rid, str):
                self.tokenizer_manager.abort_request(in_flight.rid)

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
        """
        created_time = int(time.time())
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        model = request.model
        state = StreamingASRState(**self._adapter.chunked_streaming_config)
        # Track only the trailing char of the cumulative emit; `needs_space`
        # uses prev[-1] / cur[0] so we don't need to keep the full buffer.
        last_char = ""

        try:
            chunks = split_audio_chunks(request.audio_data, state.chunk_size_sec)

            for i, chunk_audio in enumerate(chunks):
                if await raw_request.is_disconnected():
                    logger.info("[streaming_asr] client disconnected, stopping")
                    break
                is_last = i == len(chunks) - 1

                delta = await process_asr_chunk(
                    tokenizer_manager=self.tokenizer_manager,
                    adapter=self._adapter,
                    state=state,
                    audio_data=chunk_audio,
                    sampling_params=adapted_request.sampling_params,
                    is_last=is_last,
                    raw_request=raw_request,
                    routing_key=self.extract_routing_key(raw_request),
                )

                if delta:
                    for word in delta.split(" "):
                        if not word:
                            continue
                        content = f" {word}" if needs_space(last_char, word) else word
                        last_char = content[-1]
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

    async def handle_websocket(self, websocket: WebSocket) -> None:
        await handle_realtime_transcription(
            websocket,
            tokenizer_manager=self.tokenizer_manager,
            adapter=self._adapter,
            server_args=self.tokenizer_manager.server_args,
            session_semaphore=self._session_semaphore,
        )
