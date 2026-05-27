from __future__ import annotations

import fnmatch
import re
from typing import Union

_VIOLATION_LINE_RE = re.compile(
    r"kv_canary violation: launch_tag=(\S+) fail_reason=(\S+)"
)


def find_violation_in_log(
    log_text: str,
    *,
    launch_tag_patterns: tuple[str, ...],
    fail_reason: Union[str, tuple[str, ...]],
) -> bool:
    """Return True iff any logged violation matches a launch_tag pattern AND any
    of the accepted fail_reasons. `fail_reason` accepts either a single string
    or a tuple, so call sites can express 'either of these reasons is fine'
    (e.g. when two checks may legitimately fire on the same corruption)."""
    accepted = (fail_reason,) if isinstance(fail_reason, str) else tuple(fail_reason)
    for match in _VIOLATION_LINE_RE.finditer(log_text):
        tag = match.group(1)
        reasons_in_field = match.group(2).split("+")
        if not any(r in reasons_in_field for r in accepted):
            continue
        if any(fnmatch.fnmatchcase(tag, pattern) for pattern in launch_tag_patterns):
            return True
    return False


def assert_no_violation_in_log(log_text: str) -> None:
    if "kv_canary violation:" in log_text:
        raise AssertionError(
            f"Unexpected canary violation found. Log tail:\n{log_text[-2000:]}"
        )
