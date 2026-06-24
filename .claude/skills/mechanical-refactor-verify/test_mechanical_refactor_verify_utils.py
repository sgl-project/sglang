import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_verify_utils as mrv
from mechanical_refactor_verify_utils import (
    _commit_changed_lines,
    _commit_import_texts,
    _import_line_texts,
    _moved_symbol_names,
    _normalize_block,
    _strip_moved_qualifiers,
    dedent,
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
    verify_move_commit,
)

# The rule under test is specified in verifier-spec.md; these tests assert exactly
# that rule. If a test and the spec disagree, the spec wins.


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


# --- _import_line_texts --------------------------------------------------------


def test_import_line_texts_collects_single_line_imports() -> None:
    """Plain import and from-import statements contribute their stripped text."""
    text = "import os\nfrom a.b import c\n\nx = 1\n"
    assert _import_line_texts(text) == {"import os", "from a.b import c"}


def test_import_line_texts_collects_every_line_of_a_multiline_import() -> None:
    """A parenthesised import contributes each member line and the brackets."""
    text = "from pkg import (\n    alpha,\n    beta,\n)\n\nuse(alpha)\n"
    assert _import_line_texts(text) == {
        "from pkg import (",
        "alpha,",
        "beta,",
        ")",
    }


def test_import_line_texts_excludes_non_import_code() -> None:
    """Ordinary statements are never reported as import lines."""
    text = "x = 1\ndef f():\n    return alpha\n"
    assert _import_line_texts(text) == set()


def test_import_line_texts_returns_empty_for_unparseable_text() -> None:
    """A file that does not parse as Python contributes no import lines."""
    assert _import_line_texts("this is ::: not python\n") == set()


# --- _commit_import_texts ------------------------------------------------------


def test_commit_import_texts_collects_before_and_after(repo: Path) -> None:
    """Before-version and after-version import lines are gathered per side."""
    _write(repo, **{"m.py": "import os\n\nx = 1\n"})
    _commit(repo, "base")
    _write(repo, **{"m.py": "import sys\n\nx = 1\n"})
    _commit(repo, "swap import")
    before, after = _commit_import_texts("HEAD", str(repo))
    assert "import os" in before
    assert "import sys" in after


# --- _moved_symbol_names / _strip_moved_qualifiers -----------------------------


def test_moved_symbol_names_collects_def_and_class_names() -> None:
    """def, async def and class lines in the relocated block name the moved symbols."""
    lines = ["def foo(a):", "    return a", "async def bar():", "class Baz:", "x = 1"]
    assert _moved_symbol_names(lines) == {"foo", "bar", "Baz"}


def test_moved_symbol_names_ignores_non_definition_lines() -> None:
    """Lines that are not def/class definitions contribute no moved symbols."""
    assert _moved_symbol_names(["return foo()", "x = bar(1)"]) == set()


def test_strip_moved_qualifiers_drops_qualifier_before_moved_symbol() -> None:
    """A Qualifier. prefix before a moved symbol is removed; other tokens are kept."""
    assert (
        _strip_moved_qualifiers("self.x = C.helper(self.v)", {"helper"})
        == "self.x = helper(self.v)"
    )


def test_strip_moved_qualifiers_leaves_non_moved_symbols_untouched() -> None:
    """A qualifier before a symbol that did not move is left in place."""
    assert _strip_moved_qualifiers("old.compute(a)", {"helper"}) == "old.compute(a)"


# --- _normalize_block ----------------------------------------------------------


def test_normalize_block_strips_indentation_and_drops_blanks() -> None:
    """Normalization strips indentation and trailing space and drops blank lines."""
    block = [
        "    def helper(value):",
        "        return value * 2",
        "",
        "   ",
    ]
    assert _normalize_block(block) == Counter(
        {"def helper(value):": 1, "return value * 2": 1}
    )


def test_normalize_block_keeps_decorator_lines() -> None:
    """A decorator is an ordinary line now: it is kept, not treated as one-sided."""
    assert _normalize_block(["    @staticmethod", "    def m():"]) == Counter(
        {"@staticmethod": 1, "def m():": 1}
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


# --- verify_move_commit: clean moves -------------------------------------------


def test_clean_function_move_is_certified(repo: Path) -> None:
    """A function moved verbatim to a new module, with a new import, is clean."""
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


def test_move_adding_a_multiline_import_is_certified(repo: Path) -> None:
    """The continuation lines of a parenthesised import the move adds are allowed."""
    _write(
        repo,
        **{
            "src.py": (
                "def helper(value):\n    return value * 2\n\n\n"
                "def other(value):\n    return value + 1\n\n\n"
                "result = helper(3) + other(4)\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": (
                "from mod import (\n    helper,\n    other,\n)\n\n"
                "result = helper(3) + other(4)\n"
            ),
            "mod.py": (
                "def helper(value):\n    return value * 2\n\n\n"
                "def other(value):\n    return value + 1\n"
            ),
        },
    )
    _commit(repo, "move helper and other to mod")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_pure_file_rename_is_certified(repo: Path) -> None:
    """Renaming a file with no content change is a clean move (delete + add match)."""
    _write(repo, **{"orig.py": "def a():\n    return 1\n"})
    _commit(repo, "base")
    _git(repo, "mv", "orig.py", "renamed.py")
    _commit(repo, "rename orig.py to renamed.py")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_move_with_reordered_body_is_not_caught_by_multiset_comparison(
    repo: Path,
) -> None:
    """Documents the known order-blind limitation: a relocation that reorders body
    lines within the moved block is certified clean because the verifier compares
    line multisets, not sequences. This asserts current behavior, not desired
    behavior; the --color-moved cross-check in verifier-spec.md flags the reorder."""
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


# --- verify_move_commit: not a pure move ---------------------------------------


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


def test_near_identical_line_change_needs_review(repo: Path, capsys) -> None:
    """A body line that changed by even one byte (internal spacing) is not a move."""
    _write(repo, **{"src.py": "def helper(value):\n    return value * 2\n"})
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n",
            "mod.py": "def helper(value):\n    return value*2\n",
        },
    )
    _commit(repo, "move helper but reflow the body")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False
    assert "return value*2" in capsys.readouterr().out


def test_staticmethod_drop_with_call_requalification_is_certified(repo: Path) -> None:
    """Dropping @staticmethod and requalifying C.helper -> helper are move artifacts:
    the decorator is whitelisted and the call differs only by the moved symbol's
    qualifier, so the move is clean."""
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


def test_move_with_self_attr_call_site_requalification_is_certified(repo: Path) -> None:
    """A move that requalifies a `self.x = C.helper(...)` call site is clean: only the
    moved symbol's qualifier changed, the arguments are untouched."""
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
    _commit(repo, "extract helper; requalify self.x call site")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_constant_rederivation_in_new_module_needs_review(repo: Path, capsys) -> None:
    """A constant re-derived in the destination module is a non-import added line."""
    _write(
        repo,
        **{"src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": (
                "_flag = compute_flag()\n\n\ndef helper(value):\n    return value * 2\n"
            ),
        },
    )
    _commit(repo, "move helper; re-derive _flag")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False
    assert "_flag = compute_flag()" in capsys.readouterr().out


def test_call_site_rewrite_without_a_move_is_not_clean(repo: Path) -> None:
    """A 'fix consumer' commit that only rewrites call sites relocates nothing, so it
    cannot read CLEAN merely because its lines look move-shaped."""
    _write(
        repo,
        **{"src.py": "result = old.compute(a, b)\nother = old.compute(c, d)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{"src.py": "result = new.compute(a, b)\nother = new.compute(c, d)\n"},
    )
    _commit(repo, "point consumers at new")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_requalification_does_not_forgive_a_changed_argument(repo: Path) -> None:
    """Requalifying the qualifier is allowed, but if the call's argument also changed
    the line no longer matches its old form and the move needs review."""
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
                "        self.x = helper(self.w)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "extract helper but also change the argument")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_call_rewrite_for_a_non_moved_symbol_needs_review(repo: Path) -> None:
    """A clean move plus an unrelated call rewrite for a symbol that did not move is
    flagged: requalification is scoped to the moved symbol only."""
    _write(
        repo,
        **{
            "src.py": (
                "def helper(value):\n    return value * 2\n\n\n"
                "helper(3)\n"
                "result = old.compute(7)\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": (
                "from mod import helper\n\nhelper(3)\nresult = new.compute(7)\n"
            ),
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "move helper but also repoint compute")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_pure_logic_change_is_not_a_move(repo: Path) -> None:
    """A commit that only edits logic (no relocation) is not a clean move."""
    _write(repo, **{"src.py": "def helper(value):\n    return value * 2\n"})
    _commit(repo, "base")
    _write(repo, **{"src.py": "def helper(value):\n    return value * 3\n"})
    _commit(repo, "edit logic")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_commit_that_relocates_nothing_returns_false(repo: Path) -> None:
    """A commit that adds only unrelated brand-new content relocates nothing."""
    _write(repo, **{"src.py": "def a():\n    return 1\n"})
    _commit(repo, "base")
    _write(repo, **{"other.py": "def brand_new():\n    return 99\n"})
    _commit(repo, "add unrelated file")
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


def test_added_import_is_labelled_import_not_review(repo: Path, capsys) -> None:
    """The new import a move adds is reported under [import], never [review]."""
    _write(
        repo,
        **{"src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "move helper")
    verify_move_commit("HEAD", repo_root=str(repo))
    out = capsys.readouterr().out
    assert "[import] from mod import helper" in out
    assert "[review]" not in out.split("line(s) to review:")[1]


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
