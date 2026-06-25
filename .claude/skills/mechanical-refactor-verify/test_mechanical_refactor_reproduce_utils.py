import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mechanical_refactor_reproduce_utils as rr
from mechanical_refactor_reproduce_utils import (
    Repro,
    _def_span,
    _find_class,
    _find_def,
    _had_magic_comma,
    _replace_span,
    _slice_span,
    dedent,
    exec_command,
    git_add_and_commit,
    verify_mechanical_refactor,
)


def _apply(repro: Repro, root: Path) -> None:
    """Run a built Repro's recorded operations against a plain directory (no git)."""
    for op in repro.ops:
        op(root)


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


def test_had_magic_comma_detects_trailing_comma() -> None:
    """A trailing comma before the closing paren is the formatter's magic comma."""
    assert _had_magic_comma("f(\n    a,\n    b,\n)")
    assert not _had_magic_comma("f(a, b)")
    assert not _had_magic_comma("f(\n    a,\n    b\n)")


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


# --- lower_call_sites ----------------------------------------------------------


def test_lower_call_sites_moves_receiver_out_of_args(tmp_path: Path) -> None:
    """Owner.foo(receiver, rest) becomes receiver.foo(rest)."""
    (tmp_path / "m.py").write_text("x = ModelRunner.foo(self.r, a, b)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(a, b)\n"


def test_lower_call_sites_handles_only_receiver_arg(tmp_path: Path) -> None:
    """Owner.foo(receiver) becomes receiver.foo() without re-lowering the result."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(self.r)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo()\n"


def test_lower_call_sites_ignores_a_different_owner(tmp_path: Path) -> None:
    """A same-named call on another receiver (e.g. the moved body's own call) is untouched."""
    (tmp_path / "m.py").write_text("worker.foo(zmq)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "worker.foo(zmq)\n"


def test_lower_call_sites_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    """A magic trailing comma is kept so the formatter re-explodes the lowered call."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(\n    self.r,\n    a,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo(a,)\n"


# --- requalify_call_sites ------------------------------------------------------


def test_requalify_call_sites_drops_the_qualifier(tmp_path: Path) -> None:
    """Owner.bar(args) becomes bar(args) when bar moves to a free function."""
    (tmp_path / "m.py").write_text("y = ModelRunner.bar(a, b)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar(a, b)\n"


# --- remove_import -------------------------------------------------------------


def test_remove_import_scoped_leaves_module_level_same_text(tmp_path: Path) -> None:
    """Scoped to a function, it removes the local import but not a same-text module-level
    one (e.g. a TYPE_CHECKING guard), and drops the import's trailing blank line."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.mod import Thing\n"
        "\n"
        "def caller(self):\n"
        "    from pkg.mod import Thing\n"
        "\n"
        "    return Thing.go(self.x)\n"
    )
    r = Repro("b", "t").remove_import(
        "m.py", "from pkg.mod import Thing", in_function="caller"
    )
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert out.count("from pkg.mod import Thing") == 1
    assert "if TYPE_CHECKING:\n    from pkg.mod import Thing" in out
    assert "def caller(self):\n    return Thing.go(self.x)\n" in out


def test_remove_import_removes_every_occurrence_in_scope(tmp_path: Path) -> None:
    """All matching local imports in the function are removed, not just the first."""
    (tmp_path / "m.py").write_text(
        "def caller(self):\n"
        "    from pkg import M\n"
        "\n"
        "    M.a(self.x)\n"
        "    if cond:\n"
        "        from pkg import M\n"
        "\n"
        "        M.b(self.y)\n"
    )
    r = Repro("b", "t").remove_import("m.py", "from pkg import M", in_function="caller")
    _apply(r, tmp_path)
    assert "from pkg import M" not in (tmp_path / "m.py").read_text()


# --- add_import ----------------------------------------------------------------


def test_add_import_appends_after_last_top_level_import(tmp_path: Path) -> None:
    """A new import is inserted right after the last module-level import."""
    (tmp_path / "m.py").write_text("import os\nimport sys\n\nx = 1\n")
    r = Repro("b", "t").add_import("m.py", "from pkg import Thing")
    _apply(r, tmp_path)
    assert (
        tmp_path / "m.py"
    ).read_text() == "import os\nimport sys\nfrom pkg import Thing\n\nx = 1\n"


# --- move_symbol ---------------------------------------------------------------


def test_move_symbol_into_class_drops_decorator_and_appends(tmp_path: Path) -> None:
    """The def leaves the source, its @staticmethod is dropped, and it lands at the end of
    the destination class with its body verbatim."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        return x + 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def existing(self):\n        return 1\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def foo" not in src_out and "def keep" in src_out
    assert "@staticmethod" not in dst_out
    assert dst_out.index("def existing") < dst_out.index("def foo")
    assert "        return x + 1\n" in dst_out


def test_move_symbol_to_module_level_with_dedent(tmp_path: Path) -> None:
    """With into_class=None and a dedent, the def lands at module level, dedented."""
    (tmp_path / "src.py").write_text(
        "class Old:\n    @staticmethod\n    def helper(x):\n        return x * 2\n"
    )
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol(
        "helper", src="src.py", dst="dst.py", into_class=None, dedent=4
    )
    _apply(r, tmp_path)
    assert "def helper(x):\n    return x * 2\n" in (tmp_path / "dst.py").read_text()
    assert "def helper" not in (tmp_path / "src.py").read_text()


def test_move_symbol_before_inserts_above_named_sibling(tmp_path: Path) -> None:
    """With before=, the relocated def lands immediately above that sibling, not at the end."""
    (tmp_path / "src.py").write_text(
        "class Old:\n    @staticmethod\n    def moved(self):\n        return 1\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n"
        "    def first(self):\n        return 0\n"
        "\n"
        "    def last(self):\n        return 2\n"
    )
    r = Repro("b", "t").move_symbol(
        "moved", src="src.py", dst="dst.py", into_class="New", before="last"
    )
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert (
        dst_out.index("def first")
        < dst_out.index("def moved")
        < dst_out.index("def last")
    )


def test_delete_file_removes_emptied_source(tmp_path: Path) -> None:
    """delete_file removes a source module left empty after its defs relocated."""
    (tmp_path / "gone.py").write_text("import os\n")
    r = Repro("b", "t").delete_file("gone.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "gone.py").exists()


def test_move_symbol_leave_delegate_keeps_forwarding_stub(tmp_path: Path) -> None:
    """With leave_delegate, the source keeps a forwarding stub through the named field and the
    destination gets the full method body."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def existing(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute",
        src="src.py",
        dst="dst.py",
        into_class="Cfg",
        leave_delegate="cfg",
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def compute(self, n: int) -> int:" in src_out
    assert "return self.cfg.compute(n)" in src_out
    assert "return n + self.cfg.base" not in src_out
    assert "return n + self.cfg.base" in dst_out


# --- Repro.run end-to-end ------------------------------------------------------


def test_repro_run_passes_on_a_faithful_call_site_lowering(
    repo: Path, monkeypatch, capsys
) -> None:
    """End-to-end: a lowering reproduces the commit byte-for-byte (pre-commit stubbed)."""
    _write(repo, **{"c.py": "r = Old.foo(self.n, 5)\n"})
    base = _commit(repo, "base")
    _write(repo, **{"c.py": "r = self.n.foo(5)\n"})
    target = _commit(repo, "lower the call site")
    monkeypatch.chdir(repo)
    _silence_precommit(monkeypatch)

    diff = Repro(base, target).lower_call_sites("foo", "Old", paths=["c.py"]).run()
    assert diff == ""
    assert "PASS" in capsys.readouterr().out


def test_repro_run_reports_residual_when_a_change_is_bundled(
    repo: Path, monkeypatch, capsys
) -> None:
    """A bundled non-relocation change surfaces as a non-empty residual diff."""
    _write(repo, **{"c.py": "r = Old.foo(self.n, 5)\nUNRELATED = 1\n"})
    base = _commit(repo, "base")
    _write(repo, **{"c.py": "r = self.n.foo(5)\nUNRELATED = 2\n"})
    target = _commit(repo, "lower the call AND change a constant")
    monkeypatch.chdir(repo)
    _silence_precommit(monkeypatch)

    diff = Repro(base, target).lower_call_sites("foo", "Old", paths=["c.py"]).run()
    assert "UNRELATED" in diff
    assert "RESIDUAL" in capsys.readouterr().out


# --- extract_to_new_module -----------------------------------------------------


def test_extract_to_new_module_cuts_trailing_block(tmp_path: Path) -> None:
    """Cuts the trailing scaffolding+def block into a new file, prepending the future import."""
    (tmp_path / "src.py").write_text(
        "class M:\n"
        "    def keep(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def foo(x):\n"
        "    return x + 1\n"
    )
    r = Repro("b", "t").extract_to_new_module(
        "src.py", "new.py", symbols=["foo"], future_import=True
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class M:\n    def keep(self):\n        return 1\n\n\n"
    )
    assert (tmp_path / "new.py").read_text() == (
        "from __future__ import annotations\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def foo(x):\n"
        "    return x + 1\n"
    )


def test_extract_to_new_module_carries_a_trailing_class(tmp_path: Path) -> None:
    """A class in the staged tail (not just a def) travels with the cut block."""
    (tmp_path / "src.py").write_text(
        "class M:\n"
        "    pass\n"
        "\n"
        "\n"
        "from dataclasses import dataclass\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Cfg:\n"
        "    x: int\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return Cfg(1)\n"
    )
    r = Repro("b", "t").extract_to_new_module(
        "src.py", "new.py", symbols=["Cfg", "foo"], future_import=False
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "class M:\n    pass\n\n\n"
    assert "class Cfg:" in (tmp_path / "new.py").read_text()
    assert "def foo():" in (tmp_path / "new.py").read_text()


# --- repath_import / add_typechecking_import -----------------------------------


def test_repath_import_rewrites_nested_import(tmp_path: Path) -> None:
    """A function-scoped import is repathed in place; the bare call is untouched."""
    (tmp_path / "c.py").write_text(
        "class K:\n"
        "    def run(self):\n"
        "        from old.mod import foo\n"
        "\n"
        "        return foo(1)\n"
    )
    r = Repro("b", "t").repath_import(
        "c.py", old_module="old.mod", new_module="new.mod", name="foo"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "c.py").read_text() == (
        "class K:\n"
        "    def run(self):\n"
        "        from new.mod import foo\n"
        "\n"
        "        return foo(1)\n"
    )


def test_repath_import_leaves_a_module_level_import(tmp_path: Path) -> None:
    """Only nested imports are repathed; a module-level import is left to the sorter."""
    (tmp_path / "c.py").write_text("from old.mod import foo\n\n\nx = foo(1)\n")
    r = Repro("b", "t").repath_import(
        "c.py", old_module="old.mod", new_module="new.mod", name="foo"
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_add_typechecking_import_inserts_in_block(tmp_path: Path) -> None:
    """The import is appended inside the existing TYPE_CHECKING block."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import X\n"
        "    from b import Y\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )


def test_move_symbol_drops_self_annotation_into_class(tmp_path: Path) -> None:
    """Moving a `def foo(self: Target)` into Target drops the now-redundant annotation."""
    (tmp_path / "src.py").write_text(
        "class M:\n"
        "    @staticmethod\n"
        "    def foo(self: Target, x):\n"
        "        return self.y + x\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Target:\n    def keep(self):\n        return 1\n"
    )
    r = Repro("b", "t").move_symbol(
        "foo",
        src="src.py",
        dst="dst.py",
        into_class="Target",
        dedent=0,
        drop_self_annotation=True,
    )
    _apply(r, tmp_path)
    text = (tmp_path / "dst.py").read_text()
    assert "def foo(self, x):" in text
    assert "self: Target" not in text
