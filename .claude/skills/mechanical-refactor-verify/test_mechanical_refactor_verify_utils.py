import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_verify_utils as mrv
from mechanical_refactor_verify_utils import (
    _commit_changed_lines,
    _is_plumbing,
    _normalize_block,
    verify_mechanical_refactor,
    verify_move_commit,
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


# --- _is_plumbing --------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "import os",
        "from a.b import c",
        "from a.b import (",
        "    helper,",
        "helper",
        "@staticmethod",
        "if TYPE_CHECKING:",
        "logger = logging.getLogger(__name__)",
        ")",
        "):",
        "helper(value)",
        "module.helper(a, b)",
        "return helper(value)",
        "self.x = helper(value)",
    ],
)
def test_is_plumbing_accepts_move_wiring(line: str) -> None:
    """Imports, decorators, bracket fragments and call sites count as move wiring."""
    assert _is_plumbing(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "x = a + b",
        "if condition:",
        "return value * 2",
        "raise ValueError(message)",
        "for item in items:",
        "_use_aiter = get_bool_env_var('X') and is_hip()",
    ],
)
def test_is_plumbing_rejects_logic(line: str) -> None:
    """Real statements (assignments, conditionals, expression returns) are not wiring."""
    assert _is_plumbing(line) is False


# --- _normalize_block ----------------------------------------------------------


def test_normalize_block_ignores_indentation_blanks_and_one_sided_headers() -> None:
    """Normalization strips indentation and drops blanks, @staticmethod and __future__."""
    block = [
        "    @staticmethod",
        "    def helper(value):",
        "        return value * 2",
        "",
        "from __future__ import annotations",
    ]
    assert _normalize_block(block) == Counter(
        {"def helper(value):": 1, "return value * 2": 1}
    )


# --- _commit_changed_lines -----------------------------------------------------


def test_commit_changed_lines_splits_added_removed_without_headers(repo: Path) -> None:
    """The diff parser returns +/- content lines and excludes file/hunk headers."""
    _write(repo, **{"f.py": "alpha\nbeta\n"})
    _commit(repo, "base")
    _write(repo, **{"f.py": "alpha\ngamma\n"})
    _commit(repo, "edit")
    removed, added = _commit_changed_lines("HEAD", str(repo))
    assert "beta" in removed
    assert "gamma" in added
    assert not any(line.startswith(("+++", "---")) for line in removed + added)


# --- verify_move_commit --------------------------------------------------------


def test_clean_function_move_is_certified(repo: Path) -> None:
    """A function moved verbatim to a new module, with import + call site, is clean."""
    _write(
        repo, **{"src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n"}
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "move helper to mod")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_method_to_free_function_is_certified(repo: Path) -> None:
    """A @staticmethod becoming a dedented free function still verifies as a move."""
    _write(
        repo,
        **{
            "src.py": (
                "class C:\n"
                "    @staticmethod\n"
                "    def helper(value):\n"
                "        return value * 2\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
                "\n"
                "\n"
                "def use(v):\n"
                "    return C.helper(v)\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": (
                "from mod import helper\n"
                "\n"
                "\n"
                "class C:\n"
                "    def other(self):\n"
                "        return 0\n"
                "\n"
                "\n"
                "def use(v):\n"
                "    return helper(v)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "extract helper")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_file_split_is_certified(repo: Path) -> None:
    """Splitting one file into two, with bodies copied verbatim, verifies as a move."""
    _write(
        repo,
        **{"big.py": ("def a():\n    return 1\n\n\ndef b():\n    return 2\n")},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "big.py": None,
            "a.py": "def a():\n    return 1\n",
            "b.py": "def b():\n    return 2\n",
        },
    )
    _commit(repo, "split big.py")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_changed_body_line_needs_review(repo: Path) -> None:
    """If a line inside the moved body actually changes, the move is not certified."""
    _write(repo, **{"src.py": "def helper(value):\n    return value * 2\n"})
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n",
            "mod.py": "def helper(value):\n    return value * 3\n",
        },
    )
    _commit(repo, "move and tweak")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_pure_logic_change_is_not_a_move(repo: Path) -> None:
    """A commit that only edits logic (no relocation) is not a clean move."""
    _write(repo, **{"src.py": "def helper(value):\n    return value * 2\n"})
    _commit(repo, "base")
    _write(repo, **{"src.py": "def helper(value):\n    return value * 3\n"})
    _commit(repo, "edit logic")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_changed_line_appears_in_review_output(repo: Path, capsys) -> None:
    """The non-relocated change is surfaced in the printed review list."""
    _write(repo, **{"src.py": "def helper(value):\n    return value * 2\n"})
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n",
            "mod.py": "def helper(value):\n    return value + 99\n",
        },
    )
    _commit(repo, "move and tweak")
    verify_move_commit("HEAD", repo_root=str(repo))
    out = capsys.readouterr().out
    assert "[review]" in out
    assert "return value + 99" in out


# --- verify_mechanical_refactor ------------------------------------------------


def _silence_precommit(monkeypatch) -> None:
    real = mrv.exec_command

    def fake(cmd: str, cwd=None, check=True):
        if cmd.startswith("pre-commit"):
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(mrv, "exec_command", fake)


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
        mrv.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(mrv.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
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
        mrv.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(mrv.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    with pytest.raises(SystemExit):
        verify_mechanical_refactor(base, target, wrong_transform)
