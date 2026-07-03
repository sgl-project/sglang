import importlib
import logging

# Re-export the public API from base so callers can do:
#   from ...transcription_adapters import TranscriptionAdapter, register_transcription_adapter
from sglang.srt.entrypoints.openai.transcription_adapters.base import (  # noqa: F401
    TranscriptionAdapter,
    register_transcription_adapter,
    resolve_adapter,
)

logger = logging.getLogger(__name__)


def _try_import_adapter(module_name: str, class_name: str):
    try:
        module = importlib.import_module(
            f"sglang.srt.entrypoints.openai.transcription_adapters.{module_name}"
        )
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        logger.debug("Skipping transcription adapter %s: %s", module_name, e)
        return None

    adapter_cls = getattr(module, class_name)
    globals()[class_name] = adapter_cls
    return adapter_cls


# Import built-in adapters so they self-register via @register_transcription_adapter
# when their optional dependencies are available.
WhisperAdapter = _try_import_adapter("whisper", "WhisperAdapter")
Qwen3ASRAdapter = _try_import_adapter("qwen3_asr", "Qwen3ASRAdapter")
MiMoV2ASRAdapter = _try_import_adapter("mimo_v2_asr", "MiMoV2ASRAdapter")

__all__ = [
    "TranscriptionAdapter",
    "register_transcription_adapter",
    "resolve_adapter",
]

__all__ += [
    name
    for name, adapter_cls in (
        ("WhisperAdapter", WhisperAdapter),
        ("Qwen3ASRAdapter", Qwen3ASRAdapter),
        ("MiMoV2ASRAdapter", MiMoV2ASRAdapter),
    )
    if adapter_cls is not None
]
