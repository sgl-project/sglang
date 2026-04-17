from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
    TranscriptionVerboseResponse,
)


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

    def parse_fused_output(self, text: str) -> tuple[Optional[str], str]:
        """Parse the fused output to extract (language_code, transcription_text).

        Return ``(None, text)`` on parse failure (FSM abort, truncation, regex
        drift). Callers must treat ``None`` as an error and not overwrite
        ``request.language`` with it.
        """
        raise NotImplementedError

    def fused_prefix_end(self, text: str) -> int:
        """Char offset in *text* where user-visible transcription begins.

        Returns ``-1`` if the forced prefix hasn't fully arrived yet. Used by
        the streaming handler to buffer deltas across the forced-prefix
        boundary without leaking special tokens to clients.
        """
        raise NotImplementedError

    # -- Standalone detection (for external callers) -----------------------

    def build_language_detection_params(self, tokenizer) -> dict:
        """Return ``sampling_params`` dict for a language-detection-only request.

        Produces a single token (the language), constrained to valid language
        token IDs.  Callers can send this via the ``/generate`` endpoint.
        """
        raise NotImplementedError

    def parse_language_detection_output(
        self, output_ids: List[int], tokenizer
    ) -> Optional[str]:
        """Parse the detected language from a detection-only output.

        Returns an ISO 639-1 language code (e.g. ``"en"``), or None on failure.
        """
        raise NotImplementedError

    @property
    def supports_chunked_streaming(self) -> bool:
        """Whether this model uses chunk-based streaming instead of token-level streaming."""
        return False

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
