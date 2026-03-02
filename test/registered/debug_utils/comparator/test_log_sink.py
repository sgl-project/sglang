import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.log_sink import LogSink
from sglang.srt.debug_utils.comparator.output_types import (
    ErrorLog,
    InfoLog,
)
from sglang.srt.debug_utils.comparator.report_sink import report_sink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_error_log(**overrides) -> ErrorLog:
    defaults: dict = dict(
        category="test",
        message="test warning",
    )
    defaults.update(overrides)
    return ErrorLog(**defaults)


class TestLogSink:
    def test_basic_collection(self) -> None:
        sink = LogSink()
        log = _make_error_log()

        with sink.context() as collected:
            sink.add(log)

        assert len(collected) == 1
        assert collected[0] is log

    def test_nested_contexts(self) -> None:
        sink = LogSink()
        outer_log = _make_error_log(message="outer")
        inner_log = _make_error_log(message="inner")

        with sink.context() as outer:
            sink.add(outer_log)
            with sink.context() as inner:
                sink.add(inner_log)
            assert len(inner) == 1
            assert inner[0] is inner_log

        assert len(outer) == 1
        assert outer[0] is outer_log

    def test_empty_context(self) -> None:
        sink = LogSink()
        with sink.context() as collected:
            pass
        assert collected == []

    def test_add_outside_context_prints(self, capsys) -> None:
        sink = LogSink()
        report_sink.configure(output_format="text")

        sink.add(_make_error_log())

        captured = capsys.readouterr()
        assert "test warning" in captured.out

    def test_context_captures_instead_of_printing(self, capsys) -> None:
        sink = LogSink()
        report_sink.configure(output_format="text")

        with sink.context() as collected:
            sink.add(_make_error_log())

        assert len(collected) == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_json_output_outside_context(self, capsys) -> None:
        sink = LogSink()
        report_sink.configure(output_format="json")

        sink.add(_make_error_log())

        captured = capsys.readouterr()
        parsed: dict = json.loads(captured.out.strip())
        assert "errors" in parsed
        assert len(parsed["errors"]) == 1

    def test_info_log_outside_context_routes_to_infos(self, capsys) -> None:
        """InfoLog added outside context populates LogRecord.infos, not errors."""
        sink = LogSink()
        report_sink.configure(output_format="json")

        sink.add(InfoLog(category="test", message="info msg"))

        parsed: dict = json.loads(capsys.readouterr().out.strip())
        assert len(parsed["infos"]) == 1
        assert len(parsed["errors"]) == 0

    def test_exception_in_context_cleans_stack(self, capsys) -> None:
        sink = LogSink()
        report_sink.configure(output_format="text")

        with pytest.raises(RuntimeError):
            with sink.context() as collected:
                sink.add(_make_error_log())
                raise RuntimeError("boom")

        assert len(collected) == 1

        sink.add(_make_error_log(message="after exception"))
        captured = capsys.readouterr()
        assert "after exception" in captured.out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
