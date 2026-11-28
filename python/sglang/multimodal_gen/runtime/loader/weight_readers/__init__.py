# SPDX-License-Identifier: Apache-2.0
"""Ways of reading checkpoint weights, and the rule for picking one."""

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.loader.weight_readers.base import (
    WeightReader,
)
from sglang.multimodal_gen.runtime.loader.weight_readers.runai_streamer import (
    RunaiStreamerReader,
)
from sglang.multimodal_gen.runtime.loader.weight_readers.safetensors_mmap import (
    SafetensorsMmapReader,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# The fallback is last and always available, so selection cannot come up empty.
_READERS: tuple[type, ...] = (RunaiStreamerReader, SafetensorsMmapReader)
FALLBACK_READER = SafetensorsMmapReader


def available_reader_names() -> list[str]:
    return [b.name for b in _READERS if b.is_available()]


def select_weight_reader(
    *,
    requested: str | None = None,
    needs_key_filter: bool = False,
) -> WeightReader:
    """Pick a reader, honouring an explicit request where it can be honoured.

    `requested` names a reader; None means take the environment's preference.
    A reader that cannot skip keys is passed over when the caller needs to,
    because reading the whole checkpoint to discard most of it is worse than
    reading the part that was asked for more slowly.
    """
    if requested is not None:
        chosen = next((b for b in _READERS if b.name == requested), None)
        if chosen is None:
            raise ValueError(
                f"unknown weight reader {requested!r}; "
                f"available: {available_reader_names()}"
            )
    elif envs.SGLANG_USE_RUNAI_MODEL_STREAMER and RunaiStreamerReader.is_available():
        chosen = RunaiStreamerReader
    else:
        chosen = FALLBACK_READER

    if not chosen.is_available():
        logger.info(
            "Weight reader %s is not installed; using %s",
            chosen.name,
            FALLBACK_READER.name,
        )
        chosen = FALLBACK_READER
    if needs_key_filter and not chosen.supports_key_filter:
        logger.debug(
            "Weight reader %s cannot skip keys at load time; using %s",
            chosen.name,
            FALLBACK_READER.name,
        )
        chosen = FALLBACK_READER
    return chosen()


__all__ = [
    "FALLBACK_READER",
    "RunaiStreamerReader",
    "SafetensorsMmapReader",
    "WeightReader",
    "available_reader_names",
    "select_weight_reader",
]
