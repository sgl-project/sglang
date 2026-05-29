"""Wire schema for Realtime WS transcription sessions."""

from __future__ import annotations

from typing import Literal, Optional, Union

from openai.types.realtime import SessionUpdateEvent as _SessionUpdateEvent
from openai.types.realtime.audio_transcription import (
    AudioTranscription as _AudioTranscription,
)
from openai.types.realtime.realtime_audio_formats import AudioPCM as _AudioPCM
from openai.types.realtime.realtime_audio_formats import AudioPCMA as _AudioPCMA
from openai.types.realtime.realtime_audio_formats import AudioPCMU as _AudioPCMU
from openai.types.realtime.realtime_transcription_session_audio import (
    RealtimeTranscriptionSessionAudio as _AudioCfg,
)
from openai.types.realtime.realtime_transcription_session_audio_input import (
    RealtimeTranscriptionSessionAudioInput as _AudioInputCfg,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest as _SessionCfg,
)
from pydantic import Field
from typing_extensions import Annotated

# Fallback rate when the client omits `audio.input.format.rate`. SDK pins
# `AudioPCM.rate` to Literal[24000], so this matches the only value the SDK
# accepts when the field is present.
DEFAULT_INPUT_SAMPLE_RATE = 24000

# Wire rates we accept on `audio.input.format.rate` and resample to
# `adapter.model_sample_rate` server-side. 24000 matches the SDK pin;
# 16000 and 48000 widen it to cover common ASR-client and consumer-audio
# rates. Add a value here only after verifying transcription quality.
SUPPORTED_INPUT_SAMPLE_RATES = (16000, 24000, 48000)


class AudioPCM(_AudioPCM):
    type: Literal["audio/pcm"] = "audio/pcm"
    rate: Optional[int] = None


class AudioPCMU(_AudioPCMU):
    type: Literal["audio/pcmu"] = "audio/pcmu"


class AudioPCMA(_AudioPCMA):
    type: Literal["audio/pcma"] = "audio/pcma"


AudioInputFormat = Annotated[
    Union[AudioPCM, AudioPCMU, AudioPCMA],
    Field(discriminator="type"),
]


class AudioTranscription(_AudioTranscription):
    # SDK pins model to Literal["whisper-1", "gpt-4o-*-transcribe", ...];
    # sglang serves arbitrary ASR models (Qwen3-ASR, etc.) and treats the
    # client-supplied name as echo-only.
    model: Optional[str] = None


class TranscriptionSessionAudioInput(_AudioInputCfg):
    format: Optional[AudioInputFormat] = None
    transcription: Optional[AudioTranscription] = None


class TranscriptionSessionAudio(_AudioCfg):
    input: Optional[TranscriptionSessionAudioInput] = None


class TranscriptionSessionConfig(_SessionCfg):
    audio: Optional[TranscriptionSessionAudio] = None


class SessionUpdateEvent(_SessionUpdateEvent):
    session: TranscriptionSessionConfig
