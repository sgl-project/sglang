from __future__ import annotations

import fnmatch
import re

_VIOLATION_LINE_RE = re.compile(
    r"kv_canary violation: launch_tag=(\S+) fail_reason=(\S+)"
)


def find_violation_in_log(
    log_text: str,
    *,
    launch_tag_patterns: tuple[str, ...],
    fail_reason: str,
) -> bool:
    for match in _VIOLATION_LINE_RE.finditer(log_text):
        tag = match.group(1)
        reason_field = match.group(2)
        if fail_reason not in reason_field.split("+"):
            continue
        if any(fnmatch.fnmatchcase(tag, pattern) for pattern in launch_tag_patterns):
            return True
    return False


def assert_no_violation_in_log(log_text: str) -> None:
    if "kv_canary violation:" in log_text:
        raise AssertionError(
            f"Unexpected canary violation found. Log tail:\n{log_text[-2000:]}"
        )
