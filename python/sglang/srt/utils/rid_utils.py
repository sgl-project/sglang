from typing import List

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX


def expand_base_rid(rid: str, num: int) -> List[str]:
    """Expand one base rid into ``num`` zero-padded sub-rids ``<rid>_<i>``.

    The scheduler aborts by rid prefix, so unpadded ``<rid>_1`` would also abort
    ``<rid>_10``.
    """
    if not rid:
        raise ValueError("The rid should be a non-empty string.")
    # Sub-rids keep the base prefix, so a health-check prefix would make every
    # sub-request look internal.
    if rid.startswith(HEALTH_CHECK_RID_PREFIX):
        raise ValueError(
            f"The rid must not start with the reserved prefix "
            f"{HEALTH_CHECK_RID_PREFIX!r}."
        )
    width = len(str(num - 1)) if num > 1 else 1
    return [f"{rid}_{i:0{width}d}" for i in range(num)]
