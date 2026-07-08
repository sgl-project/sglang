import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mechanical_refactor_reproduce_utils as rr
from mechanical_refactor_reproduce_utils import (
    Repro,
    _def_span,
    _find_class,
    _find_def,
    _replace_span,
    _slice_span,
    dedent,
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
)
from reproduce_testlib import _apply, _commit, _git, _write  # noqa: F401


# --- exec_command --------------------------------------------------------------


def test_exec_command_returns_stripped_stdout_on_success() -> None:
    """A successful command returns its stdout with surrounding whitespace stripped."""
    assert exec_command("echo hello") == "hello"


def test_exec_command_respects_cwd(tmp_path: Path) -> None:
    """The command runs in the supplied working directory."""
    sub = tmp_path / "workdir"
    sub.mkdir()
    assert exec_command("pwd", cwd=str(sub)) == str(sub.resolve())


def test_exec_command_check_true_raises_on_failure() -> None:
    """With check=True a non-zero exit status raises RuntimeError with the command."""
    with pytest.raises(RuntimeError, match="exit 7"):
        exec_command("exit 7", check=True)


def test_exec_command_check_false_returns_stdout_without_exiting() -> None:
    """With check=False a failing command returns its stdout and does not exit."""
    assert exec_command("echo partial; exit 3", check=False) == "partial"


# --- git_add_and_commit --------------------------------------------------------


# --- git_add_and_commit --------------------------------------------------------


def test_git_add_and_commit_stages_and_commits(repo: Path) -> None:
    """It stages every change in the cwd and records a commit with the message."""
    _write(repo, **{"file.txt": "content\n"})
    git_add_and_commit("add file", cwd=str(repo))
    assert _git(repo, "log", "-1", "--format=%s") == "add file"
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    "message",
    [
        "subject with spaces",
        "has 'single' and \"double\" quotes",
        "shell $HOME && rm -rf / ; metacharacters",
        "trailing parens (a, b) and pipe | semicolon ;",
    ],
)
def test_git_add_and_commit_message_round_trips_with_metacharacters(
    repo: Path, message: str
) -> None:
    """Messages with shell metacharacters are quoted safely and survive verbatim."""
    _write(repo, **{"file.txt": "content\n"})
    git_add_and_commit(message, cwd=str(repo))
    assert _git(repo, "log", "-1", "--format=%B") == message


# --- dedent --------------------------------------------------------------------


# --- dedent --------------------------------------------------------------------


def test_dedent_with_zero_leaves_text_unchanged() -> None:
    """Dedenting by zero spaces returns the text untouched."""
    text = "  indented\nplain\n"
    assert dedent(text, 0) == text


def test_dedent_removes_exactly_n_leading_spaces() -> None:
    """Exactly n leading spaces are removed from each qualifying line."""
    assert dedent("    four\n        eight\n", 4) == "four\n    eight\n"


def test_dedent_leaves_lines_with_fewer_than_n_spaces_unchanged() -> None:
    """A line with fewer than n leading spaces is not modified at all."""
    assert dedent("    four\n  two\nzero\n", 4) == "four\n  two\nzero\n"


def test_dedent_does_not_strip_tabs() -> None:
    """Tab characters are never treated as the spaces dedent removes."""
    assert dedent("\t\ttabbed\n", 2) == "\t\ttabbed\n"


def test_dedent_preserves_blank_lines_and_trailing_newline() -> None:
    """Blank lines and a final newline are preserved across line boundaries."""
    assert dedent("    a\n\n    b\n", 4) == "a\n\nb\n"


def test_dedent_preserves_absence_of_trailing_newline() -> None:
    """A text without a trailing newline keeps it absent after dedenting."""
    assert dedent("    a\n    b", 4) == "a\nb"


# --- span / call helpers -------------------------------------------------------


# --- span / call helpers -------------------------------------------------------


def test_replace_span_single_line() -> None:
    """A span within one line is replaced in place."""
    assert _replace_span("ab cd ef\n", 1, 3, 1, 5, "XY") == "ab XY ef\n"


def test_replace_span_across_lines() -> None:
    """A span crossing lines collapses to the replacement between the kept prefix/suffix."""
    text = "a = foo(\n    x,\n) + 1\n"
    assert _replace_span(text, 1, 4, 3, 1, "bar()") == "a = bar() + 1\n"


def test_slice_span_returns_the_overwritten_text() -> None:
    """_slice_span returns exactly the region _replace_span would overwrite."""
    text = "a = foo(\n    x,\n) + 1\n"
    assert _slice_span(text, 1, 4, 3, 1) == "foo(\n    x,\n)"


def test_find_def_span_includes_decorators() -> None:
    """A def's span starts at its first decorator and ends at its last body line."""
    src = "class C:\n    @staticmethod\n    def foo(self):\n        return 1\n"
    node = _find_def(rr.ast.parse(src), "foo")
    assert node is not None and _def_span(node) == (2, 4)


def test_find_class_returns_named_class_or_none() -> None:
    """_find_class locates a class by name and returns None when absent."""
    tree = rr.ast.parse("class A:\n    pass\nclass B:\n    pass\n")
    assert _find_class(tree, "B").name == "B"
    assert _find_class(tree, "Z") is None
