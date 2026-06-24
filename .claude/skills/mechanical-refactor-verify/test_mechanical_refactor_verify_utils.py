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
    dedent,
    exec_command,
    git_add_and_commit,
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


# --- _is_plumbing: additional wiring forms -------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "import x as y",
        "from a import b as c",
        "aliased as renamed,",
        "",
        "(",
        "]",
        "},",
        "a.b.c(x)",
    ],
)
def test_is_plumbing_accepts_more_move_wiring(line: str) -> None:
    """Aliased imports, lone bracket fragments and method-chain calls are wiring."""
    assert _is_plumbing(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "return value",
        "self.attr.x = foo(a)",
        "total = (a + b)",
        "x = f(a) + g(b)",
    ],
)
def test_is_plumbing_rejects_logic_that_superficially_resembles_calls(
    line: str,
) -> None:
    """Returns of bare names, nested-attribute assignments and multi-call
    expressions are logic, even when they contain parentheses or a call."""
    assert _is_plumbing(line) is False


# --- _normalize_block: additional edge cases -----------------------------------


def test_normalize_block_drops_classmethod_decorator() -> None:
    """The @classmethod header line is treated as one-sided and dropped."""
    assert _normalize_block(["    @classmethod", "    def m(cls):"]) == Counter(
        {"def m(cls):": 1}
    )


def test_normalize_block_counts_duplicate_lines() -> None:
    """Identical body lines are accumulated as separate counts in the Counter."""
    assert _normalize_block(["    x = 1", "    x = 1", "        x = 1"]) == Counter(
        {"x = 1": 3}
    )


def test_normalize_block_drops_whitespace_only_lines() -> None:
    """Lines that are empty or only whitespace are dropped entirely."""
    assert _normalize_block(["", "   ", "\t", "    real = 1"]) == Counter(
        {"real = 1": 1}
    )


def test_normalize_block_drops_multiple_one_sided_headers() -> None:
    """Several one-sided header lines are all removed in a single block."""
    block = [
        "    @staticmethod",
        "    @classmethod",
        "from __future__ import annotations",
        "    keep = 1",
    ]
    assert _normalize_block(block) == Counter({"keep = 1": 1})


# --- _commit_changed_lines: additional shapes ----------------------------------


def test_commit_changed_lines_add_only_file_has_no_removed_lines(repo: Path) -> None:
    """A commit that only adds a new file reports added lines and no removed lines."""
    _write(repo, **{"base.py": "kept\n"})
    _commit(repo, "base")
    _write(repo, **{"new.py": "fresh\ncontent\n"})
    _commit(repo, "add new file")
    removed, added = _commit_changed_lines("HEAD", str(repo))
    assert removed == []
    assert "fresh" in added
    assert "content" in added


def test_commit_changed_lines_delete_only_file_has_no_added_lines(repo: Path) -> None:
    """A commit that only deletes a file reports removed lines and no added lines."""
    _write(repo, **{"keep.py": "kept\n", "gone.py": "doomed\nlines\n"})
    _commit(repo, "base")
    _write(repo, **{"gone.py": None})
    _commit(repo, "delete file")
    removed, added = _commit_changed_lines("HEAD", str(repo))
    assert "doomed" in removed
    assert "lines" in removed
    assert added == []


def test_commit_changed_lines_spans_multiple_files_without_diff_headers(
    repo: Path,
) -> None:
    """Changes across several files are merged, and diff/index/hunk headers
    never leak into the returned line lists."""
    _write(repo, **{"a.py": "a_old\n", "b.py": "b_old\n"})
    _commit(repo, "base")
    _write(repo, **{"a.py": "a_new\n", "b.py": "b_new\n"})
    _commit(repo, "edit both")
    removed, added = _commit_changed_lines("HEAD", str(repo))
    assert {"a_old", "b_old"} <= set(removed)
    assert {"a_new", "b_new"} <= set(added)
    combined = removed + added
    assert not any(line.startswith("@@") for line in combined)
    assert not any(line.startswith("diff --git") for line in combined)
    assert not any(line.startswith("index ") for line in combined)


# --- verify_move_commit: additional relocation shapes --------------------------


def test_move_into_existing_module_file_is_certified(repo: Path) -> None:
    """Relocating a function into an already-existing module file is a clean move."""
    _write(
        repo,
        **{
            "src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n",
            "mod.py": "def existing():\n    return 0\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": (
                "def existing():\n    return 0\n\n\ndef helper(value):\n"
                "    return value * 2\n"
            ),
        },
    )
    _commit(repo, "move helper into existing mod")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_move_with_self_attr_call_site_rewrite_is_certified(repo: Path) -> None:
    """A move that rewrites a `self.x = foo(...)` call site is recognized as wiring."""
    _write(
        repo,
        **{
            "src.py": (
                "class C:\n"
                "    @staticmethod\n"
                "    def helper(value):\n"
                "        return value * 2\n"
                "\n"
                "    def m(self):\n"
                "        self.x = C.helper(self.v)\n"
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
                "    def m(self):\n"
                "        self.x = helper(self.v)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "extract helper; rewrite self.x call site")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_move_with_bare_call_site_rewrite_is_certified(repo: Path) -> None:
    """A move that rewrites a bare `foo(...)` statement call site is clean."""
    _write(
        repo,
        **{
            "src.py": (
                "class C:\n"
                "    @staticmethod\n"
                "    def helper(value):\n"
                "        return value * 2\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
                "\n"
                "\n"
                "C.helper(3)\n"
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
                "    def keep(self):\n"
                "        return 0\n"
                "\n"
                "\n"
                "helper(3)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "extract helper; bare call site")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_move_with_real_adaptation_line_needs_review(repo: Path, capsys) -> None:
    """A relocation that also adds a genuine logic line needs review and lists it."""
    _write(
        repo,
        **{"src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": (
                "from mod import helper\n"
                "\n"
                '_use_aiter = get_bool_env_var("X") and is_hip()\n'
                "\n"
                "helper(3)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "move helper + add adaptation line")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False
    out = capsys.readouterr().out
    assert "[review]" in out
    assert '_use_aiter = get_bool_env_var("X") and is_hip()' in out


def test_commit_that_relocates_nothing_returns_false(repo: Path) -> None:
    """A commit that adds only unrelated brand-new content relocates nothing."""
    _write(repo, **{"src.py": "def a():\n    return 1\n"})
    _commit(repo, "base")
    _write(repo, **{"other.py": "def brand_new():\n    return 99\n"})
    _commit(repo, "add unrelated file")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_move_with_reordered_body_is_not_caught_by_multiset_comparison(
    repo: Path,
) -> None:
    """Documents the known order-blind limitation: a relocation that reorders body
    lines within the moved block is certified clean because the verifier compares
    line multisets, not sequences. This asserts current behavior, not desired
    behavior; a true sequence-aware check would flag the reorder."""
    _write(
        repo,
        **{"src.py": ("def helper():\n    step_one()\n    step_two()\n\n\nhelper()\n")},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper()\n",
            "mod.py": "def helper():\n    step_two()\n    step_one()\n",
        },
    )
    _commit(repo, "move helper and reorder its body")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


# --- verify_mechanical_refactor: worktree branch and pre-commit-fixes path ------


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
        mrv.git_add_and_commit("split", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(mrv.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _silence_precommit(monkeypatch)

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    assert _git(repo, "branch", "--list", branch).endswith(branch)


def _precommit_writes_file(monkeypatch, filename: str, contents: str) -> None:
    real = mrv.exec_command

    def fake(cmd: str, cwd=None, check=True):
        if cmd.startswith("pre-commit"):
            (Path(cwd) / filename).write_text(contents)
            return ""
        return real(cmd, cwd=cwd, check=check)

    monkeypatch.setattr(mrv, "exec_command", fake)


def test_reproduce_commits_pre_commit_fixes_when_tree_left_dirty(
    repo: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    """When pre-commit reformats and leaves the tree dirty, a 'pre-commit fixes'
    commit is created on top of the transform commit."""
    _write(repo, **{"src.py": "hello\n"})
    base = _commit(repo, "base")
    _write(repo, **{"src.py": "hello world\n", "formatted.py": "auto\n"})
    target = _commit(repo, "edit")

    def transform(root: Path) -> None:
        (root / "src.py").write_text("hello world\n")
        mrv.git_add_and_commit("transform", cwd=str(root))

    monkeypatch.chdir(repo)
    monkeypatch.setattr(mrv.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wt"))
    _precommit_writes_file(monkeypatch, "formatted.py", "auto\n")

    verify_mechanical_refactor(base, target, transform)
    assert "PASS" in capsys.readouterr().out
    branch = f"verify-mechanical-{base[:8]}"
    subjects = _git(repo, "log", "--format=%s", "-2", branch).splitlines()
    assert subjects == ["pre-commit fixes", "transform"]
