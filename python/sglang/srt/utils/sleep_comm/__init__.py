"""Sleep-mode memory-occupation tags and the comm-cleanup backend contract.

Kept import-light: only ``tags`` is re-exported here, since the HTTP-facing
process imports it.
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
