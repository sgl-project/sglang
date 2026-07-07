"""Segment-streaming ASR over engine streaming sessions.

Chunked streaming (streaming_asr.py) re-transcribes a window of audio on
every chunk, because bidirectional audio towers can't extend KV over one
growing clip. Segment streaming takes the other trade: audio is cut into
fixed-duration segments and each segment is submitted as one turn of an
engine streaming session — user(audio) -> assistant(transcript) rounds
in a single growing context. Prior turns' KV is reused, so each second
of audio is encoded exactly once and per-segment cost is constant,
instead of growing with the utterance.

Requires --enable-streaming-session. Mirrors vLLM's SupportsRealtime
segment flow (its qwen3_asr_realtime model buffers the mic into 5 s
segments and appends each as a resumable-request turn).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from sglang.srt.entrypoints.openai.streaming_asr import normalize_whitespace
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
)
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    GenerateReqInput,
    OpenSessionReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class SegmentStreamingASR:
    """One utterance transcribed incrementally over a streaming session.

    Call ``transcribe_segment(wav)`` once per fixed-duration segment;
    always ``close()`` when the utterance ends — including error paths —
    to release the scheduler-side session. The engine session is opened
    lazily on the first segment.
    """

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        adapter: TranscriptionAdapter,
        sampling_params: Dict[str, Any],
        routing_key: Optional[str] = None,
    ) -> None:
        self.tokenizer_manager = tokenizer_manager
        self.adapter = adapter
        self.sampling_params = sampling_params
        self.routing_key = routing_key
        self.session_id: Optional[str] = None
        self.transcript = ""

    async def transcribe_segment(self, wav_data: bytes) -> str:
        """Submit one audio segment as a session turn; return its text."""
        if self.session_id is None:
            session_id = await self.tokenizer_manager.open_session(
                OpenSessionReqInput(
                    capacity_of_str_len=0,
                    session_id=f"asr-seg-{uuid.uuid4().hex}",
                    streaming=True,
                )
            )
            if session_id is None:
                raise RuntimeError(
                    "Failed to open a streaming session for segment ASR; "
                    "is the server running with --enable-streaming-session?"
                )
            self.session_id = session_id

        # One user(audio) -> assistant round per segment; the session
        # carries all previous rounds, so the model sees the full history
        # and returns only the continuation.
        req = GenerateReqInput(
            text=self.adapter.prompt_template,
            audio_data=wav_data,
            sampling_params=self.sampling_params,
            stream=False,
            modalities=["audio"],
            session_params={"id": self.session_id, "rid": None},
        )
        if self.routing_key is not None:
            req.routing_key = self.routing_key

        ret = None
        async for ret in self.tokenizer_manager.generate_request(req):
            break
        if ret is None:
            logger.warning(
                "[segment_asr] empty response for session %s", self.session_id
            )
            return ""

        text = normalize_whitespace(self.adapter.postprocess_text(ret.get("text", "")))
        if text:
            self.transcript = (
                f"{self.transcript} {text}".strip() if self.transcript else text
            )
        return text

    async def close(self) -> None:
        """Release the scheduler-side session. Idempotent."""
        if self.session_id is None:
            return
        session_id, self.session_id = self.session_id, None
        try:
            await self.tokenizer_manager.close_session(
                CloseSessionReqInput(session_id=session_id)
            )
        except Exception:
            logger.exception("[segment_asr] failed to close session %s", session_id)
