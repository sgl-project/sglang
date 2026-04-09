from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

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
