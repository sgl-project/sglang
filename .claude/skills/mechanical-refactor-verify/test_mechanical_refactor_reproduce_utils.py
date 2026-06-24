import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_reproduce_utils as rr
from mechanical_refactor_reproduce_utils import (
    dedent,
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, **files: str | None) -> None:
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    return root


# --- exec_command --------------------------------------------------------------


def test_exec_command_returns_stripped_stdout_on_success() -> None:
    """A successful command returns its stdout with surrounding whitespace stripped."""
    assert exec_command("echo hello") == "hello"


def test_exec_command_respects_cwd(tmp_path: Path) -> None:
    """The command runs in the supplied working directory."""
    sub = tmp_path / "workdir"
    sub.mkdir()
    assert exec_command("pwd", cwd=str(sub)) == str(sub.resolve())


def test_exec_command_check_true_exits_on_failure() -> None:
    """With check=True a non-zero exit status raises SystemExit."""
    with pytest.raises(SystemExit):
        exec_command("exit 7", check=True)


def test_exec_command_check_false_returns_stdout_without_exiting() -> None:
    """With check=False a failing command returns its stdout and does not exit."""
    assert exec_command("echo partial; exit 3", check=False) == "partial"


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


# --- verify_mechanical_refactor ------------------------------------------------


def _silence_precommit(monkeypatch) -> None:
    real = rr.exec_command

    def fake(cmd: str, cwd=None, check=True):
        if cmd.startswith("pre-commit"):
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(rr, "exec_command", fake)


def test_reproduce_passes_when_transform_matches_target(
    repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """A transform that recreates the target tree reports PASS and does not exit."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def transform(root: Path) -> None:
        lines = (root / "src.py").read_text().splitlines(keepends=True)
        (root / "a.py").write_text("".join(lines[0:2]))
        (root / "b.py").write_text("".join(lines[2:3]))
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out


def test_reproduce_exits_when_transform_diverges(
    repo: Path, tmp_path: Path, monkeypatch
) -> None:
    """A transform that produces a different tree fails with a non-zero exit."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def wrong_transform(root: Path) -> None:
        (root / "a.py").write_text("WRONG\n")
        (root / "b.py").write_text("line3\n")
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    with pytest.raises(SystemExit):
        verify_mechanical_refactor(base, target, wrong_transform)


def test_reproduce_creates_verify_branch_on_pass(
    repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """A PASS run leaves a verify-mechanical-<base[:8]> branch in the repo."""
    _write(repo, **{"src.py": "line1\nline2\nline3\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": None, "a.py": "line1\nline2\n", "b.py": "line3\n"})
    target = _commit(repo, "split")

    def transform(root: Path) -> None:
        lines = (root / "src.py").read_text().splitlines(keepends=True)
        (root / "a.py").write_text("".join(lines[0:2]))
        (root / "b.py").write_text("".join(lines[2:3]))
        (root / "src.py").unlink()
        rr.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    assert _git(repo, "branch", "--list", branch).endswith(branch)


def _precommit_writes_file(monkeypatch, filename: str, contents: str) -> None:
    real = rr.exec_command

    def fake(cmd: str, cwd=None, check=True):
        if cmd.startswith("pre-commit"):
            (Path(cwd) / filename).write_text(contents)
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(rr, "exec_command", fake)


def test_reproduce_commits_pre_commit_fixes_when_tree_left_dirty(
    repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """When pre-commit reformats and leaves the tree dirty, a 'pre-commit fixes' commit
    is created on top of the transform commit."""
    _write(repo, **{"src.py": "hello\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": "hello world\n", "formatted.py": "auto\n"})
    target = _commit(repo, "edit")

    def transform(root: Path) -> None:
        (root / "src.py").write_text("hello world\n")
        rr.git_add_and_commit("transform", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(rr.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _precommit_writes_file(monkeypatch, "formatted.py", "auto\n")

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    subjects = _git(repo, "log", "--format=%s", "-2", branch).splitlines()
    assert subjects == ["pre-commit fixes", "transform"]
