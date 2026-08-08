from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import msgspec

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)


class RealtimeEncoderWindowPolicy(msgspec.Struct, frozen=True):
    """Adapter policy for long-audio encoder-window continuation."""

    min_audio_sec: float
    max_audio_context_windows: int
    supported_languages: tuple[str, ...]
    decoder_prefix_max_tokens: int = 192
    decoder_prefix_holdback_words: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_audio_sec) or self.min_audio_sec < 0:
            raise ValueError("min_audio_sec must be finite and non-negative")
        if self.max_audio_context_windows <= 0:
            raise ValueError("max_audio_context_windows must be positive")
        if self.decoder_prefix_max_tokens <= 0:
            raise ValueError("decoder_prefix_max_tokens must be positive")
        if self.decoder_prefix_holdback_words < 0:
            raise ValueError("decoder_prefix_holdback_words must be non-negative")
        if not self.supported_languages or not all(
            isinstance(language, str) and language.strip()
            for language in self.supported_languages
        ):
            raise ValueError("supported_languages must contain language codes")
        normalized_languages = tuple(
            language.strip().lower().replace("_", "-").split("-", 1)[0]
            for language in self.supported_languages
        )
        msgspec.structs.force_setattr(self, "supported_languages", normalized_languages)

    def supports_language(self, language: Optional[str]) -> bool:
        if not language:
            return False
        primary = language.strip().lower().replace("_", "-").split("-", 1)[0]
        return primary in self.supported_languages


class TranscriptionAdapter(ABC):
    """Abstract base for model-specific transcription logic.

    Subclass this and decorate with ``@register_transcription_adapter("Key")``
    to add support for a new ASR model.  See the sibling modules for
    the built-in Whisper and Qwen3-ASR implementations.
    """

    @abstractmethod
    def build_sampling_params(self, request: TranscriptionRequest) -> dict:
        """Return the ``sampling_params`` dict for ``GenerateReqInput``."""

    @property
    def supports_language_detection(self) -> bool:
        """Whether this model supports automatic language detection.

        When True, the adapter must implement the fused autodetect methods
        and the standalone detection methods below.
        """
        return False

    # -- Fused detect+transcribe (used by the server) ----------------------

    def build_fused_autodetect_params(self, request) -> dict:
        """Return ``sampling_params`` dict for a fused detect+transcribe request.

        Uses structured generation (``regex``) to constrain the output prefix
        to a valid language + task token sequence while allowing free
        transcription afterwards — all in a single request.
        """
        raise NotImplementedError

    @staticmethod
    def parse_fused_output(
        text: str, *, ts_variant: bool = False
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse the fused output into ``(language_code, user_visible_text)``.

        Called by both streaming and non-streaming handlers with the same
        contract. ``ts_variant`` indicates which forced-prefix shape was
        requested (the caller knows from ``request.timestamp_granularities``);
        adapters use it to disambiguate variants whose detokenized prefix
        differs in shape from their token-id prefix.

        * ``(None, None)`` — the forced prefix is not yet locatable.
          Streaming callers keep buffering; non-streaming / end-of-stream
          callers treat this as a parse failure and fall back to
          ``strip_special_tokens`` on the raw text.
        * ``(lang, visible)`` — prefix parsed. ``visible`` is fully
          user-visible (prefix removed, embedded special tokens scrubbed).
          It must grow monotonically across cumulative streaming snapshots
          so callers can compute deltas against it directly.
        """
        raise NotImplementedError

    @staticmethod
    def strip_special_tokens(text: str) -> str:
        """Best-effort scrub of model-specific special-token strings.

        Used as a fallback when ``parse_fused_output`` reports a parse
        failure (e.g. FSM abort). Default is an identity pass-through;
        adapters that request generation with ``skip_special_tokens=False``
        should override to strip their special-token syntax.
        """
        return text

    @property
    def supports_chunked_streaming(self) -> bool:
        """Whether this model uses chunk-based streaming instead of token-level streaming."""
        return False

    @property
    def model_sample_rate(self) -> int:
        """Target sample rate in Hz the model expects. Realtime WS path
        resamples client PCM to this rate before chunking. Default 16000
        matches Whisper / Qwen3-ASR; override for models expecting other rates.
        """
        return 16000

    @property
    def prompt_template(self) -> str:
        """Prompt template for chunked streaming requests.

        Only used when ``supports_chunked_streaming`` is True.
        The default returns an empty string.
        """
        return ""

    @property
    def chunked_streaming_config(self) -> dict:
        """Parameters for ``StreamingASRState`` when using chunked streaming.

        Only used when ``supports_chunked_streaming`` is True.
        Keys: ``chunk_size_sec``, ``unfixed_chunk_num``, ``unfixed_token_num``.
        """
        return {}

    @property
    def realtime_encoder_window_policy(self) -> Optional[RealtimeEncoderWindowPolicy]:
        """Return the long-audio window policy, or None when unsupported."""
        return None

    def postprocess_text(self, text: str) -> str:
        """Strip model-specific markers from raw decoded text.

        The default implementation is a no-op pass-through.
        """
        return text

    @abstractmethod
    def build_verbose_response(
        self,
        request: TranscriptionRequest,
        text: str,
        ret: dict,
        tokenizer,
        usage: TranscriptionUsage,
    ) -> TranscriptionVerboseResponse:
        """Build a ``verbose_json`` response with segments / timestamps."""


_ADAPTER_REGISTRY: dict[str, type[TranscriptionAdapter]] = {}
_DEFAULT_ADAPTER_KEY = "Whisper"


def register_transcription_adapter(
    key: str,
) -> callable:
    """Class decorator that registers a ``TranscriptionAdapter`` subclass.

    *key* is matched as a substring against the model's HF ``architectures``
    list at init time (e.g. ``"Whisper"`` matches
    ``"WhisperForConditionalGeneration"``).
    """

    def decorator(cls: type[TranscriptionAdapter]) -> type[TranscriptionAdapter]:
        _ADAPTER_REGISTRY[key] = cls
        return cls

    return decorator


def resolve_adapter(architectures: List[str]) -> TranscriptionAdapter:
    """Pick the right adapter by matching architecture names against the registry."""
    for arch in architectures or []:
        for key, adapter_cls in _ADAPTER_REGISTRY.items():
            if key in arch:
                return adapter_cls()
    default_cls = _ADAPTER_REGISTRY.get(_DEFAULT_ADAPTER_KEY)
    if default_cls is None:
        raise RuntimeError(
            "No transcription adapters registered. "
            "Make sure 'transcription_adapters' package is importable."
        )
    return default_cls()
