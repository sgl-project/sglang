"""Memory-occupation tag schema shared by the control plane and the scheduler.

A platform adds tags by registering a ``SleepCommBackend``; core then accepts
that tag and includes it in the tag-less default set. Kept import-light because
the HTTP-facing process uses it and must not pull in torch.
"""

import logging
from typing import List, Optional, Tuple

from sglang.srt.constants import GPU_MEMORY_ALL_TYPES, GPU_MEMORY_TYPE_COMM

logger = logging.getLogger(__name__)

# Tags only honored where a platform registered a sleep-comm backend.
PLATFORM_GATED_TAGS: Tuple[str, ...] = (GPU_MEMORY_TYPE_COMM,)


def all_known_tags() -> List[str]:
    """Every accepted tag, in the default release order."""
    return list(GPU_MEMORY_ALL_TYPES) + list(PLATFORM_GATED_TAGS)


def default_release_tags() -> List[str]:
    """The tag set a tag-less release means on this platform, in release order."""
    from sglang.srt.utils.sleep_comm.backend import enabled_platform_tags

    return list(GPU_MEMORY_ALL_TYPES) + list(enabled_platform_tags())


def resolve_release_tags(raw_tags: Optional[List[str]]) -> List[str]:
    """Resolve the effective tag set, dropping platform tags this platform
    did not register so a cross-platform client can always send the full set."""
    if raw_tags is None or len(raw_tags) == 0:
        tags = default_release_tags()
    else:
        tags = list(dict.fromkeys(raw_tags))

    from sglang.srt.utils.sleep_comm.backend import enabled_platform_tags

    enabled = set(enabled_platform_tags())
    for tag in list(tags):
        if tag in PLATFORM_GATED_TAGS and tag not in enabled:
            logger.info(
                "The '%s' memory occupation tag is not supported on this "
                "platform; ignoring it.",
                tag,
            )
            tags.remove(tag)
    return tags


def validate_tags(tags: List[str]) -> None:
    unknown = sorted(set(tags) - set(all_known_tags()))
    if unknown:
        raise ValueError(f"Unknown memory occupation tag(s): {unknown}")
