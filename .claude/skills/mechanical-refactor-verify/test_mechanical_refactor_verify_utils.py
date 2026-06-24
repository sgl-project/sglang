import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mechanical_refactor_verify_utils import (
    _block_signature,
    _commit_changed_lines,
    _commit_import_texts,
    _import_line_texts,
    _moved_symbol_names,
    _strip_moved_qualifiers,
    _strip_self_annotation,
    verify_move_commit,
    verify_move_range,
)

# The rule under test is specified in verifier-spec.md; these tests assert exactly that
# rule. If a test and the spec disagree, the spec wins.


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
    assert _import_line_texts(text) == {"from pkg import (", "alpha,", "beta,", ")"}


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
    """def, async def and class lines name the moved symbols."""
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


def test_strip_self_annotation_drops_self_type() -> None:
    """A type annotation on the self parameter of a def is removed."""
    assert (
        _strip_self_annotation("def foo(self: Target) -> None:")
        == "def foo(self) -> None:"
    )


def test_strip_self_annotation_leaves_plain_self_and_other_params() -> None:
    """A plain self, and annotations on other parameters, are untouched."""
    assert _strip_self_annotation("def foo(self) -> None:") == "def foo(self) -> None:"
    assert _strip_self_annotation("def foo(self, x: int):") == "def foo(self, x: int):"


# --- _block_signature ----------------------------------------------------------


def test_block_signature_strips_uniform_common_indent() -> None:
    """The common leading indent of the block is removed from every line."""
    assert _block_signature(["        a = 1", "        return a"]) == [
        "a = 1",
        "return a",
    ]


def test_block_signature_preserves_relative_indentation() -> None:
    """Only the common prefix is removed; relative nesting survives."""
    assert _block_signature(["    if c:", "        x = 1"]) == ["if c:", "    x = 1"]


def test_block_signature_drops_blank_lines() -> None:
    """Blank lines are dropped so separator blanks do not affect the comparison."""
    assert _block_signature(["    a", "   ", "    b"]) == ["a", "b"]


def test_block_signature_preserves_trailing_whitespace() -> None:
    """Trailing whitespace is kept, so a trailing-space change is detectable."""
    assert _block_signature(["    a  "]) == ["a  "]


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


def test_commit_changed_lines_preserves_leading_whitespace(repo: Path) -> None:
    """A changed line keeps its indentation so whitespace checks can see it."""
    _write(repo, **{"f.py": "def f():\n    return 1\n"})
    _commit(repo, "base")
    _write(repo, **{"f.py": "def f():\n        return 1\n"})
    _commit(repo, "reindent")
    removed, added = _commit_changed_lines("HEAD", str(repo))
    assert "    return 1" in removed
    assert "        return 1" in added


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
    _write(repo, **{"big.py": "def a():\n    return 1\n\n\ndef b():\n    return 2\n"})
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


def test_uniform_indentation_shift_is_certified(repo: Path) -> None:
    """A @staticmethod becoming a free function dedents the whole body by a constant;
    that uniform shift, the dropped decorator, and the requalified call are all allowed.
    """
    _write(
        repo,
        **{
            "src.py": (
                "class C:\n"
                "    @staticmethod\n"
                "    def helper(value):\n"
                "        if value:\n"
                "            return value * 2\n"
                "        return 0\n"
                "\n"
                "    def m(self):\n"
                "        return C.helper(self.v)\n"
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
                "        return helper(self.v)\n"
            ),
            "mod.py": (
                "def helper(value):\n"
                "    if value:\n"
                "        return value * 2\n"
                "    return 0\n"
            ),
        },
    )
    _commit(repo, "extract helper to module level")
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


def test_method_to_method_drops_staticmethod_and_self_annotation(repo: Path) -> None:
    """Relocating `@staticmethod def foo(self: Target)` into Target as an instance method
    `def foo(self)` is clean: the dropped decorator and the dropped self annotation are
    both mechanical move artifacts."""
    _write(
        repo,
        **{
            "src.py": (
                "class Source:\n"
                "    @staticmethod\n"
                '    def foo(self: "Target") -> None:\n'
                "        self.field_a = 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "tgt.py": "class Target:\n    def existing(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "class Source:\n    def other(self):\n        return 0\n",
            "tgt.py": (
                "class Target:\n"
                "    def existing(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self) -> None:\n"
                "        self.field_a = 1\n"
            ),
        },
    )
    _commit(repo, "move foo into Target as an instance method")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


# --- verify_move_commit: the new order / whitespace rules ----------------------


def test_reordered_body_needs_review(repo: Path) -> None:
    """A relocation that reorders lines within the moved block is NOT a clean move:
    the verifier compares the block as an ordered sequence."""
    _write(
        repo,
        **{"src.py": "def helper():\n    step_one()\n    step_two()\n\n\nhelper()\n"},
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
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_trailing_whitespace_change_needs_review(repo: Path) -> None:
    """A moved line that gains trailing whitespace is not a byte-faithful move."""
    _write(repo, **{"src.py": "def helper(v):\n    return v * 2\n\n\nhelper(3)\n"})
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(v):\n    return v * 2 \n",
        },
    )
    _commit(repo, "move helper but add trailing space")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


def test_blank_line_only_change_is_tolerated(repo: Path) -> None:
    """A blank line added inside the moved block is tolerated: blank lines never change
    behavior and separator blanks legitimately collapse when code is relocated."""
    _write(
        repo,
        **{"src.py": "def helper(v):\n    a = v\n    return a\n\n\nhelper(3)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(v):\n    a = v\n\n    return a\n",
        },
    )
    _commit(repo, "move helper but add a blank line")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is True


def test_non_uniform_indentation_needs_review(repo: Path) -> None:
    """An indentation change that is not a single uniform shift of the whole block is
    flagged, because it can change Python semantics."""
    _write(
        repo,
        **{"src.py": "def helper(v):\n    if v:\n        return v\n\n\nhelper(3)\n"},
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(v):\n    if v:\n            return v\n",
        },
    )
    _commit(repo, "move helper but over-indent the return")
    assert verify_move_commit("HEAD", repo_root=str(repo)) is False


# --- verify_move_commit: other non-moves ---------------------------------------


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
    """Dropping @staticmethod and requalifying C.helper -> helper are move artifacts."""
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
            "src.py": "from mod import helper\n\nhelper(3)\nresult = new.compute(7)\n",
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
    """The new import a move adds is reported under [import], and a clean move prints
    no [review] lines."""
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
    assert "[review]" not in out


# --- verify_move_range ---------------------------------------------------------


def _make_move_then_tweak(repo: Path) -> str:
    """Build base -> a clean move (subject has '-move') -> a logic tweak; return base."""
    _write(
        repo,
        **{"src.py": "def helper(value):\n    return value * 2\n\n\nhelper(3)\n"},
    )
    base = _commit(repo, "base")
    _write(
        repo,
        **{
            "src.py": "from mod import helper\n\nhelper(3)\n",
            "mod.py": "def helper(value):\n    return value * 2\n",
        },
    )
    _commit(repo, "extract-helper-move: move helper to mod")
    _write(repo, **{"mod.py": "def helper(value):\n    return value * 3\n"})
    _commit(repo, "tweak-helper: change the logic")
    return base


def test_verify_move_range_match_skips_non_matching_subjects(
    repo: Path, capsys
) -> None:
    """With --match only commits whose subject matches are verified; others are skipped."""
    base = _make_move_then_tweak(repo)
    result = verify_move_range(f"{base}..HEAD", match="-move", repo_root=str(repo))
    out = capsys.readouterr().out
    assert result is True
    assert "verified 1 commit(s), skipped 1" in out
    assert "extract-helper-move" in out
    assert "tweak-helper" not in out


def test_verify_move_range_without_match_verifies_every_commit(
    repo: Path, capsys
) -> None:
    """Without --match the whole range is verified; a non-move makes the result False."""
    base = _make_move_then_tweak(repo)
    result = verify_move_range(f"{base}..HEAD", repo_root=str(repo))
    out = capsys.readouterr().out
    assert result is False
    assert "verified 2 commit(s)" in out
