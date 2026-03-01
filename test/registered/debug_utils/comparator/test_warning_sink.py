import json
import sys

import pytest

from sglang.srt.debug_utils.comparator.output_types import ReplicatedMismatchWarning
from sglang.srt.debug_utils.comparator.warning_sink import WarningSink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _make_warning(**overrides) -> ReplicatedMismatchWarning:
    defaults: dict = dict(
        axis="tp",
        group_index=0,
        differing_index=1,
        baseline_index=0,
        max_abs_diff=0.1,
    )
    defaults.update(overrides)
    return ReplicatedMismatchWarning(**defaults)


class TestWarningSink:
    def test_basic_collection(self) -> None:
        sink = WarningSink()
        warning = _make_warning()

        with sink.context() as collected:
            sink.add(warning)

        assert len(collected) == 1
        assert collected[0] is warning

    def test_nested_contexts(self) -> None:
        sink = WarningSink()
        outer_warning = _make_warning(group_index=0)
        inner_warning = _make_warning(group_index=1)

        with sink.context() as outer:
            sink.add(outer_warning)
            with sink.context() as inner:
                sink.add(inner_warning)
            assert len(inner) == 1
            assert inner[0] is inner_warning

        assert len(outer) == 1
        assert outer[0] is outer_warning

    def test_empty_context(self) -> None:
        sink = WarningSink()
        with sink.context() as collected:
            pass
        assert collected == []

    def test_add_outside_context_prints(self, capsys) -> None:
        sink = WarningSink()
        sink.set_output_format("text")

        sink.add(_make_warning())

        captured = capsys.readouterr()
        assert "Replicated along tp" in captured.out

    def test_context_captures_instead_of_printing(self, capsys) -> None:
        sink = WarningSink()
        sink.set_output_format("text")

        with sink.context() as collected:
            sink.add(_make_warning())

        assert len(collected) == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_json_output_outside_context(self, capsys) -> None:
        sink = WarningSink()
        sink.set_output_format("json")

        sink.add(_make_warning())

        captured = capsys.readouterr()
        parsed: dict = json.loads(captured.out.strip())
        assert "warnings" in parsed
        assert len(parsed["warnings"]) == 1

    def test_exception_in_context_cleans_stack(self, capsys) -> None:
        sink = WarningSink()
        sink.set_output_format("text")

        with pytest.raises(RuntimeError):
            with sink.context() as collected:
                sink.add(_make_warning())
                raise RuntimeError("boom")

        assert len(collected) == 1

        sink.add(_make_warning(group_index=99))
        captured = capsys.readouterr()
        assert "Replicated along tp" in captured.out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
