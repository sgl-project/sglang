from typing import List

from sglang.srt.constants import DRAFT_RID_PREFIX, HEALTH_CHECK_RID_PREFIX

# Prefixes that mark internal requests. A base rid is expanded into sub-rids that
# keep its prefix, so a base rid using one would make every sub-request look internal.
_RESERVED_RID_PREFIXES = (HEALTH_CHECK_RID_PREFIX, DRAFT_RID_PREFIX)


def expand_base_rid(rid: str, num: int) -> List[str]:
    """Expand one base rid into ``num`` zero-padded sub-rids ``<rid>_<i>``.

    The scheduler aborts by rid prefix, so unpadded ``<rid>_1`` would also abort
    ``<rid>_10``.
    """
    if not isinstance(rid, str) or not rid:
        raise ValueError("The rid should be a non-empty string.")
    for reserved in _RESERVED_RID_PREFIXES:
        if rid.startswith(reserved):
            raise ValueError(
                f"The rid must not start with the reserved prefix {reserved!r}."
            )
    width = len(str(num - 1)) if num > 1 else 1
    return [f"{rid}_{i:0{width}d}" for i in range(num)]
