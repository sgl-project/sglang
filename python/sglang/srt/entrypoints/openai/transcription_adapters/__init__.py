# Re-export the public API from base so callers can do:
#   from ...transcription_adapters import TranscriptionAdapter, register_transcription_adapter
from sglang.srt.entrypoints.openai.transcription_adapters.base import (  # noqa: F401
    TranscriptionAdapter,
    register_transcription_adapter,
    resolve_adapter,
)

# Import built-in adapters so they self-register via @register_transcription_adapter.
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (  # noqa: F401
    Qwen3ASRAdapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (  # noqa: F401
    WhisperAdapter,
)

__all__ = [
    "TranscriptionAdapter",
    "register_transcription_adapter",
    "resolve_adapter",
    "WhisperAdapter",
    "Qwen3ASRAdapter",
]
