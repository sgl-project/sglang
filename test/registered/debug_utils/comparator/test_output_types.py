import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import (
    ErrorLog,
    InfoLog,
    LogRecord,
    _split_logs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def test_split_logs_mixed_list() -> None:
    """_split_logs correctly partitions a mixed list of ErrorLog and InfoLog."""
    errors, infos = _split_logs(
        [
            ErrorLog(category="a", message="err"),
            InfoLog(category="b", message="info"),
            ErrorLog(category="c", message="err2"),
        ]
    )
    assert len(errors) == 2
    assert len(infos) == 1
    assert errors[0].message == "err"
    assert errors[1].message == "err2"
    assert infos[0].message == "info"


def test_log_record_to_text_format() -> None:
    """LogRecord.to_text() renders errors with ✗ and infos with ℹ markers."""
    record = LogRecord(
        errors=[ErrorLog(category="a", message="bad thing")],
        infos=[InfoLog(category="b", message="fyi")],
    )
    text: str = record.to_text()
    assert "✗ bad thing" in text
    assert "ℹ fyi" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
