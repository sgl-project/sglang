from __future__ import annotations

import fnmatch
import re
import time
from typing import Callable, TypeVar

_RecordT = TypeVar("_RecordT")

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


def wait_for_log_records(
    read_log: Callable[[], str],
    parse: Callable[[str], list[_RecordT]],
    *,
    flush_wait_seconds: float,
    max_retries: int,
) -> tuple[list[_RecordT], str]:
    log_text = ""
    for _ in range(max_retries):
        time.sleep(flush_wait_seconds)
        log_text = read_log()
        if records := parse(log_text):
            return records, log_text
    return [], log_text
