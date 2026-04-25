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
