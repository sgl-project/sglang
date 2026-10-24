"""Sleep-mode memory-occupation tags and comm-cleanup backend contract.

Lives next to ``torch_memory_saver_adapter`` because both serve the same
feature (sleep mode). Keep this module import-light: only ``tags`` is
re-exported here, since the HTTP-facing process imports it. Platform
backends live in ``sglang.srt.utils.sleep_comm.backend``.
"""

from sglang.srt.utils.sleep_comm.tags import (
    PLATFORM_GATED_TAGS,
    all_known_tags,
    default_release_tags,
    resolve_release_tags,
    validate_tags,
)

__all__ = [
    "PLATFORM_GATED_TAGS",
    "all_known_tags",
    "default_release_tags",
    "resolve_release_tags",
    "validate_tags",
]
