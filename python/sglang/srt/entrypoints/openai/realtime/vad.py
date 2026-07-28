"""Streaming voice-activity detection for Realtime WS transcription.

Implements OpenAI Realtime ``turn_detection: {type: "server_vad"}`` on
top of silero-vad. Ported from sglang-omni's ``serve/realtime/vad.py``,
adapted to lazy-load the model (silero-vad is an optional dependency)
and to allow injecting a scorer for deterministic unit tests.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import msgspec
import numpy as np

logger = logging.getLogger(__name__)

# silero-vad operates on 512-sample windows @ 16 kHz (32 ms each).
VAD_FRAME_SAMPLES = 512
VAD_SAMPLE_RATE = 16000


class VADConfig(msgspec.Struct, frozen=True):
    """Mirrors OpenAI Realtime ``turn_detection`` (server_vad mode)."""

    # Probs greater than or equal to threshold are considered speech.
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class VADEvent:
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


class Emit(msgspec.Struct, frozen=True):
    event_type: str
    # Offset in samples on the VAD's monotonic session clock.
    sample_offset: int


def load_silero_scorer() -> Callable[[np.ndarray], float]:
    """Load silero-vad and return a frame -> speech-probability scorer.

    Raises ImportError with an install hint if silero-vad is missing, so
    the session layer can surface a clear error to the client.
    """
    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError as e:
        raise ImportError(
            "turn_detection: server_vad requires the silero-vad package; "
            "install it with `pip install silero-vad`"
        ) from e

    model = load_silero_vad()

    def scorer(frame: np.ndarray) -> float:
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).unsqueeze(0)
            return float(model(tensor, VAD_SAMPLE_RATE).item())

    return scorer


class StreamingVAD:
    """Per-session frame-by-frame VAD state machine.

    Callers feed raw PCM16 LE mono @ 16 kHz via :meth:`process`. Up to
    one frame's worth of leftover bytes is buffered between calls so the
    caller doesn't have to align to 32 ms. ``samples_consumed`` is a
    monotonic session clock — it survives commits and clears so
    speech_started / speech_stopped timestamps stay session-relative.
    """

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        scorer: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.config = config or VADConfig()
        self.scorer = scorer if scorer is not None else load_silero_scorer()
        self.leftover_pcm = bytearray()
        self.samples_consumed = 0
        self.is_speech = False
        self.silence_run_samples = 0
        self.last_speech_offset = 0

    def process(self, pcm_bytes: bytes) -> List[Emit]:
        """Feed PCM16 LE mono @ 16 kHz; return any state transitions."""
        if not pcm_bytes:
            return []
        self.leftover_pcm.extend(pcm_bytes)
        emits: List[Emit] = []

        while len(self.leftover_pcm) >= VAD_FRAME_SAMPLES * 2:
            frame_bytes = bytes(self.leftover_pcm[: VAD_FRAME_SAMPLES * 2])
            del self.leftover_pcm[: VAD_FRAME_SAMPLES * 2]
            frame = np.frombuffer(frame_bytes, dtype="<i2").astype(np.float32) / 32768.0

            prob = self.scorer(frame)
            self.samples_consumed += VAD_FRAME_SAMPLES
            speech = prob >= self.config.threshold

            if speech:
                self.silence_run_samples = 0
                self.last_speech_offset = self.samples_consumed
                if not self.is_speech:
                    self.is_speech = True
                    # OpenAI's contract: speech_started reports the start
                    # offset *minus* prefix_padding so the caller includes
                    # a leading prefix in the committed audio.
                    pad = self.config.prefix_padding_ms * VAD_SAMPLE_RATE // 1000
                    started_at = max(0, self.samples_consumed - VAD_FRAME_SAMPLES - pad)
                    emits.append(
                        Emit(
                            event_type=VADEvent.SPEECH_STARTED,
                            sample_offset=started_at,
                        )
                    )
            else:
                self.silence_run_samples += VAD_FRAME_SAMPLES
                if self.is_speech:
                    silence_threshold = (
                        self.config.silence_duration_ms * VAD_SAMPLE_RATE // 1000
                    )
                    if self.silence_run_samples >= silence_threshold:
                        self.is_speech = False
                        emits.append(
                            Emit(
                                event_type=VADEvent.SPEECH_STOPPED,
                                sample_offset=self.last_speech_offset,
                            )
                        )

        return emits

    def end_utterance(self) -> None:
        """Reset per-utterance speech state, keeping the session clock.

        Called on input_audio_buffer.clear so abandoned speech doesn't
        leak a dangling is_speech into the next utterance.
        """
        self.leftover_pcm.clear()
        self.is_speech = False
        self.silence_run_samples = 0


def offset_to_ms(samples: int) -> int:
    return samples * 1000 // VAD_SAMPLE_RATE
