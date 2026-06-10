import pytest
from pydantic import ValidationError

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchApplicationError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu", nightly=True)
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestApplyEdits:
    """Tests for the apply_edits() source text transformation function."""

    def test_single_line_match_to_multiline_replacement(self) -> None:
        source = "def foo():\n" "    x = compute()\n" "    return x\n"
        edits = [
            EditSpec(
                match="x = compute()",
                replacement="x = compute()\nprint(x)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n" "    x = compute()\n" "    print(x)\n" "    return x\n"
        )

    def test_pure_insertion(self) -> None:
        source = "def foo():\n" "    a = 1\n" "    b = 2\n"
        edits = [
            EditSpec(
                match="a = 1",
                replacement="a = 1\nprint(a)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    a = 1\n" "    print(a)\n" "    b = 2\n")

    def test_pure_deletion_via_empty_replacement(self) -> None:
        source = "def foo():\n" "    debug_log()\n" "    return 42\n"
        edits = [
            EditSpec(
                match="debug_log()",
                replacement="",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    return 42\n")

    def test_deletion_fewer_lines(self) -> None:
        source = "def foo():\n" "    a = 1\n" "    b = 2\n" "    c = 3\n"
        edits = [
            EditSpec(
                match="a = 1\nb = 2",
                replacement="ab = 3",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    ab = 3\n" "    c = 3\n")

    def test_multiline_match_to_multiline_replacement(self) -> None:
        source = (
            "def foo():\n"
            "    result = self.attn(\n"
            "        q=q,\n"
            "        k=k,\n"
            "    )\n"
            "    return result\n"
        )
        edits = [
            EditSpec(
                match="result = self.attn(\n    q=q,\n    k=k,\n)",
                replacement="result = self.attn(\n    q=q,\n    k=k,\n    v=v,\n)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "    result = self.attn(\n"
            "        q=q,\n"
            "        k=k,\n"
            "        v=v,\n"
            "    )\n"
            "    return result\n"
        )

    def test_indent_alignment_deep_nesting(self) -> None:
        source = (
            "class Foo:\n"
            "    class Bar:\n"
            "        def method(self):\n"
            "            x = compute()\n"
            "            return x\n"
        )
        edits = [
            EditSpec(
                match="x = compute()",
                replacement="x = compute()\nprint(x)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "class Foo:\n"
            "    class Bar:\n"
            "        def method(self):\n"
            "            x = compute()\n"
            "            print(x)\n"
            "            return x\n"
        )

    def test_match_not_found_raises(self) -> None:
        source = "def foo():\n    return 1\n"
        edits = [EditSpec(match="nonexistent_call()", replacement="replaced()")]
        with pytest.raises(PatchApplicationError, match="not found"):
            apply_edits(source=source, edits=edits)

    def test_not_found_diagnostic_reports_source_len(self) -> None:
        """diagnostic includes total source line count."""
        source = "line0\nline1\nline2\nline3\nline4\n"
        edits = [EditSpec(match="absent()", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        assert "source_len=5 lines" in str(exc_info.value)

    def test_not_found_diagnostic_when_first_match_line_absent(self) -> None:
        """diagnostic says 'does NOT appear anywhere' when first line is never present."""
        source = "def foo():\n    return 1\n"
        edits = [EditSpec(match="nope_xyz()", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert "does NOT appear anywhere in source" in msg
        assert "'nope_xyz()'" in msg

    def test_not_found_diagnostic_single_window_with_marker(self) -> None:
        """first line is present once but full match doesn't fit: one window with '>' on the match-region line."""
        source = (
            "line0\n"
            "line1\n"
            "line2\n"
            "anchor()\n"
            "wrong_next()\n"
            "line5\n"
            "line6\n"
        )
        edits = [EditSpec(match="anchor()\nright_next()", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert "appears 1 time(s)" in msg
        assert msg.count("--") == 1
        assert ">    3: anchor()" in msg
        assert ">    4: wrong_next()" in msg
        assert "     1: line1" in msg
        assert "     2: line2" in msg
        assert "     5: line5" in msg
        assert "     6: line6" in msg

    def test_not_found_diagnostic_multiple_windows_separated(self) -> None:
        """when first line appears N (<=8) times, N windows are shown separated by '--'."""
        source = (
            "anchor()\n"
            "tail_a()\n"
            "filler\n"
            "anchor()\n"
            "tail_b()\n"
            "filler\n"
            "anchor()\n"
            "tail_c()\n"
        )
        edits = [EditSpec(match="anchor()\nnope()", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert "appears 3 time(s)" in msg
        assert msg.count("--") == 3
        assert ">    0: anchor()" in msg
        assert ">    3: anchor()" in msg
        assert ">    6: anchor()" in msg

    def test_not_found_diagnostic_caps_at_8_windows(self) -> None:
        """when first line appears >8 times, only the first 8 windows are rendered."""
        source = "\n".join(["anchor()"] * 12) + "\n"
        edits = [EditSpec(match="anchor()\nnope()", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert "appears 12 time(s)" in msg
        assert "up to 8 windows" in msg
        assert msg.count("--") == 8

    def test_not_found_diagnostic_window_clamps_at_source_boundaries(self) -> None:
        """window does not include negative indices or indices past the end of source."""
        source = "anchor()\nfoo\n"
        edits = [EditSpec(match="anchor()\nbar", replacement="x")]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert ">    0: anchor()" in msg
        assert ">    1: foo" in msg
        assert "-1:" not in msg
        assert "   2:" not in msg

    def test_not_found_diagnostic_multiline_match_marks_full_region(self) -> None:
        """match spanning N lines: marker '>' covers all N lines of the intended match region."""
        source = (
            "filler0\n"
            "filler1\n"
            "filler2\n"
            "filler3\n"
            "anchor()\n"
            "middle()\n"
            "wrong_tail()\n"
            "filler7\n"
            "filler8\n"
            "filler9\n"
        )
        edits = [
            EditSpec(match="anchor()\nmiddle()\nright_tail()", replacement="x"),
        ]
        with pytest.raises(PatchApplicationError) as exc_info:
            apply_edits(source=source, edits=edits)
        msg = str(exc_info.value)
        assert ">    4: anchor()" in msg
        assert ">    5: middle()" in msg
        assert ">    6: wrong_tail()" in msg
        assert "     2: filler2" in msg
        assert "     3: filler3" in msg
        assert "     7: filler7" in msg
        assert "     8: filler8" in msg
        assert "filler0" not in msg
        assert "filler1" not in msg
        assert "filler9" not in msg

    def test_match_found_multiple_times_raises(self) -> None:
        source = "def foo():\n" "    print(1)\n" "    print(1)\n"
        edits = [EditSpec(match="print(1)", replacement="print(2)")]
        with pytest.raises(PatchApplicationError, match="multiple"):
            apply_edits(source=source, edits=edits)

    def test_multiple_edits_applied_sequentially(self) -> None:
        source = "def foo():\n" "    a = 1\n" "    b = 2\n" "    return a + b\n"
        edits = [
            EditSpec(match="a = 1", replacement="a = 10"),
            EditSpec(match="b = 2", replacement="b = 20"),
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n" "    a = 10\n" "    b = 20\n" "    return a + b\n"
        )

    def test_strip_matching_ignores_leading_trailing_whitespace(self) -> None:
        source = "def foo():\n" "    x = compute()\n" "    return x\n"
        edits = [
            EditSpec(
                match="  x = compute()  ",
                replacement="x = replaced()",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    x = replaced()\n" "    return x\n")

    def test_replacement_indented_text_realigned(self) -> None:
        """replacement text with its own indentation gets realigned to match source."""
        source = "def foo():\n" "        x = compute()\n" "        return x\n"
        edits = [
            EditSpec(
                match="x = compute()",
                replacement="x = compute()\nprint(x)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "        x = compute()\n"
            "        print(x)\n"
            "        return x\n"
        )

    def test_replacement_with_existing_indent_realigned(self) -> None:
        """replacement text already has indentation that should be rebased."""
        source = "def foo():\n" "    if True:\n" "        x = 1\n" "        return x\n"
        edits = [
            EditSpec(
                match="x = 1",
                replacement="x = 1\nif x > 0:\n    print(x)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "    if True:\n"
            "        x = 1\n"
            "        if x > 0:\n"
            "            print(x)\n"
            "        return x\n"
        )

    def test_append_keeps_match_and_adds_after(self) -> None:
        source = "def foo():\n" "    x = compute()\n" "    return x\n"
        edits = [EditSpec(match="x = compute()", append="print(x)")]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n" "    x = compute()\n" "    print(x)\n" "    return x\n"
        )

    def test_append_multiline_match(self) -> None:
        source = (
            "def foo():\n"
            "    result = call(\n"
            "        a=1,\n"
            "        b=2,\n"
            "    )\n"
            "    return result\n"
        )
        edits = [
            EditSpec(
                match="result = call(\n    a=1,\n    b=2,\n)",
                append="dumper.dump('result', result)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "    result = call(\n"
            "        a=1,\n"
            "        b=2,\n"
            "    )\n"
            "    dumper.dump('result', result)\n"
            "    return result\n"
        )

    def test_prepend_adds_before_match(self) -> None:
        source = "def foo():\n" "    x = compute()\n" "    return x\n"
        edits = [EditSpec(match="x = compute()", prepend="print('before')")]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "    print('before')\n"
            "    x = compute()\n"
            "    return x\n"
        )

    def test_prepend_multiline(self) -> None:
        source = "def foo():\n" "    return x\n"
        edits = [EditSpec(match="return x", prepend="a = 1\nb = 2")]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    a = 1\n" "    b = 2\n" "    return x\n")

    def test_prepend_deep_indent(self) -> None:
        source = (
            "class Foo:\n"
            "    class Bar:\n"
            "        def method(self):\n"
            "            return x\n"
        )
        edits = [EditSpec(match="return x", prepend="dumper.dump('x', x)")]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "class Foo:\n"
            "    class Bar:\n"
            "        def method(self):\n"
            "            dumper.dump('x', x)\n"
            "            return x\n"
        )

    def test_prepend_multiline_match(self) -> None:
        source = (
            "def foo():\n"
            "    result = call(\n"
            "        a=1,\n"
            "    )\n"
            "    return result\n"
        )
        edits = [
            EditSpec(
                match="result = call(\n    a=1,\n)",
                prepend="dumper.dump('before', x)",
            )
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == (
            "def foo():\n"
            "    dumper.dump('before', x)\n"
            "    result = call(\n"
            "        a=1,\n"
            "    )\n"
            "    return result\n"
        )

    def test_replacement_and_append_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="only one of"):
            EditSpec(match="x = 1", replacement="x = 2", append="print(x)")

    def test_replacement_and_prepend_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="only one of"):
            EditSpec(match="x = 1", replacement="x = 2", prepend="print(x)")

    def test_prepend_and_append_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="only one of"):
            EditSpec(match="x = 1", prepend="a()", append="b()")

    def test_second_edit_sees_result_of_first(self) -> None:
        """Edits are applied sequentially; second edit matches modified source."""
        source = "def foo():\n" "    x = 1\n" "    return x\n"
        edits = [
            EditSpec(match="x = 1", replacement="x = 1\ny = 2"),
            EditSpec(match="y = 2", replacement="y = 20"),
        ]
        result = apply_edits(source=source, edits=edits)
        assert result == ("def foo():\n" "    x = 1\n" "    y = 20\n" "    return x\n")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
