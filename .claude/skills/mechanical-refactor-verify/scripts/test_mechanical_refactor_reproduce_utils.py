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


def test_exec_command_check_true_raises_on_failure() -> None:
    """With check=True a non-zero exit status raises RuntimeError with the command."""
    with pytest.raises(RuntimeError, match="exit 7"):
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


def test_lowered_call_text_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    """A magic trailing comma in the original call survives the textual lowering."""
    (tmp_path / "m.py").write_text("x = Old.foo(\n    self.r,\n    a,\n    b,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(\n    a,\n    b,\n)\n"


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
    assert (tmp_path / "m.py").read_text() == "self.r.foo(\n    a,\n)\n"


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


# --- remove_imported_name ------------------------------------------------------


def test_remove_imported_name_drops_one_name_from_a_multi_name_import(
    tmp_path: Path,
) -> None:
    """One name is dropped from a `from m import a, b, c`; the others stay on the line."""
    (tmp_path / "m.py").write_text("from pkg import a, moved, b\n\nx = a + b\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a, b\n\nx = a + b\n"


def test_remove_imported_name_drops_whole_statement_when_sole_name(
    tmp_path: Path,
) -> None:
    """Dropping the only name removes the whole `from` statement."""
    (tmp_path / "m.py").write_text("from pkg import moved\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_drops_a_plain_import_with_module_none(
    tmp_path: Path,
) -> None:
    """With module=None a plain `import name` statement is removed."""
    (tmp_path / "m.py").write_text("import gc\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name("m.py", module=None, name="gc")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_matches_an_asname(tmp_path: Path) -> None:
    """The alias is matched on both the name and the asname, so `import numpy as np` is found."""
    (tmp_path / "m.py").write_text("import numpy as np\nimport os\n\nx = 1\n")
    r = Repro("b", "t").remove_imported_name(
        "m.py", module=None, name="numpy", asname="np"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n\nx = 1\n"


def test_remove_imported_name_asserts_when_absent(tmp_path: Path) -> None:
    """Removing a name that is not imported raises, so a wrong recipe fails loudly."""
    (tmp_path / "m.py").write_text("from pkg import a, b\n")
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="missing")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


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


def test_move_symbol_leave_delegate_does_not_absorb_leading_comments(
    tmp_path: Path,
) -> None:
    """A leading comment before the first statement is not an AST node, so it must not be
    pulled into the forwarding stub -- the delegate is just the header plus the return.
    """
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        # explain the maths\n"
        "        # second comment line\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def existing(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "# explain the maths" not in src_out
    assert (
        src_out == "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        return self.cfg.compute(n)\n"
    )
    assert "# explain the maths" in dst_out


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


# --- extract_symbols_to_new_module ---------------------------------------------


def test_extract_symbols_to_new_module_gathers_scattered_defs(tmp_path: Path) -> None:
    """Scattered top-level defs are cut from the source and assembled under the authored header
    in the given order; the source keeps everything else."""
    (tmp_path / "src.py").write_text(
        "import os\n"
        "\n"
        "\n"
        "def keep_a():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def moved_b():\n"
        "    return 2\n"
        "\n"
        "\n"
        "def keep_c():\n"
        "    return 3\n"
        "\n"
        "\n"
        "def moved_a():\n"
        "    return 4\n"
    )
    header = (
        "from __future__ import annotations\n"
        "\n"
        "import logging\n"
        "\n"
        "logger = logging.getLogger(__name__)\n"
    )
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py",
        "new.py",
        symbols=["moved_b", "moved_a"],
        header=header,
        order=["moved_a", "moved_b"],
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    assert "def moved_a" not in src_out and "def moved_b" not in src_out
    assert "def keep_a" in src_out and "def keep_c" in src_out
    new_out = (tmp_path / "new.py").read_text()
    assert new_out.startswith("from __future__ import annotations\n")
    assert "logger = logging.getLogger(__name__)" in new_out
    assert new_out.index("def moved_a") < new_out.index("def moved_b")
    assert "    return 4\n" in new_out and "    return 2\n" in new_out


def test_extract_symbols_to_new_module_asserts_order_permutes_symbols(
    tmp_path: Path,
) -> None:
    """An order that is not a permutation of the symbols raises, so a wrong recipe fails."""
    (tmp_path / "src.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    )
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py", "n.py", symbols=["a", "b"], header="", order=["a"]
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_extract_symbols_to_new_module_asserts_when_symbol_absent(
    tmp_path: Path,
) -> None:
    """A symbol that is not a top-level def/class in the source raises."""
    (tmp_path / "src.py").write_text("def a():\n    return 1\n")
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py", "n.py", symbols=["a", "missing"], header="", order=["a", "missing"]
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_extract_symbols_to_new_module_drops_relocated_assigns(tmp_path: Path) -> None:
    """A module-level constant that moved into the new module's header is deleted from the
    source (its copy lives in the authored header); a kept assignment stays."""
    (tmp_path / "src.py").write_text(
        "import os\n"
        "\n"
        "_FLAG = os.cpu_count()\n"
        "stay = 1\n"
        "\n"
        "\n"
        "def moved():\n"
        "    return _FLAG\n"
    )
    header = (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "\n"
        "_FLAG = os.cpu_count()\n"
    )
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py",
        "new.py",
        symbols=["moved"],
        header=header,
        order=["moved"],
        drop_assigns=["_FLAG"],
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    assert "_FLAG = os.cpu_count()" not in src_out
    assert "stay = 1" in src_out
    assert "_FLAG = os.cpu_count()" in (tmp_path / "new.py").read_text()


def test_extract_symbols_to_new_module_asserts_unknown_drop_assign(
    tmp_path: Path,
) -> None:
    """A drop_assigns name that is not assigned at module level in the source raises."""
    (tmp_path / "src.py").write_text("X = 1\n\n\ndef m():\n    return X\n")
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py", "n.py", symbols=["m"], header="", order=["m"], drop_assigns=["Y"]
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


# --- extract_function ----------------------------------------------------------


def test_extract_function_relocates_body_and_replaces_with_call(tmp_path: Path) -> None:
    """An inline block is cut verbatim, re-indented under the new signature, and the call site
    replaced; the body lands at function-body indent."""
    (tmp_path / "src.py").write_text(
        "class Q:\n"
        "    def run(self, n):\n"
        "        total = 0\n"
        "        for i in range(n):\n"
        "            total += i * i\n"
        "        return total\n"
    )
    (tmp_path / "dst.py").write_text("def existing():\n    return 0\n")
    body = "        total = 0\n        for i in range(n):\n            total += i * i\n"
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="sum_squares",
        signature="def sum_squares(n):",
        body=body,
        body_indent=8,
        call="        total = sum_squares(n)\n",
        return_text="    return total\n",
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    assert "        total = sum_squares(n)\n" in src_out
    assert "for i in range(n)" not in src_out
    assert (
        "def sum_squares(n):\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        total += i * i\n"
        "    return total\n"
    ) in (tmp_path / "dst.py").read_text()


def test_extract_function_inserts_before_named_sibling(tmp_path: Path) -> None:
    """With before=, the new function lands immediately above that sibling at module level."""
    (tmp_path / "src.py").write_text("x = compute()\n")
    (tmp_path / "dst.py").write_text(
        "def a():\n    return 1\n\n\ndef c():\n    return 3\n"
    )
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="b",
        signature="def b():",
        body="x = compute()\n",
        body_indent=0,
        call="x = b()\n",
        return_text="    return x\n",
        before="c",
    )
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert dst_out.index("def a") < dst_out.index("def b") < dst_out.index("def c")
    assert "x = b()\n" == (tmp_path / "src.py").read_text()


def test_extract_function_asserts_block_not_unique(tmp_path: Path) -> None:
    """A block that occurs more than once in the source raises, so the cut is unambiguous."""
    (tmp_path / "src.py").write_text("p = f()\np = f()\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="p = f()\n",
        body_indent=0,
        call="p = g()\n",
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


# --- adversarial audit: move_symbol edge cases -----------------------------------


def test_move_symbol_moves_an_async_def_verbatim(tmp_path: Path) -> None:
    """An async def relocates with its `async` keyword and body byte-identical."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    async def foo(self):\n"
        "        return 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Old:\n\n    def keep(self):\n        return 0\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class New:\n"
        "    def e(self):\n"
        "        return 1\n"
        "\n"
        "    async def foo(self):\n"
        "        return 1\n"
    )


def test_move_symbol_moves_a_def_within_the_same_file(tmp_path: Path) -> None:
    """With src == dst the def is cut and re-inserted above its sibling in one file."""
    (tmp_path / "m.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    )
    r = Repro("b", "t").move_symbol(
        "b", src="m.py", dst="m.py", into_class=None, before="a"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "def b():\n    return 2\n\ndef a():\n    return 1\n\n\n"
    )


def test_move_symbol_prefers_a_module_level_def_over_an_earlier_class_method(
    tmp_path: Path,
) -> None:
    """When a class method and a module-level def share a name, the module-level def moves."""
    (tmp_path / "src.py").write_text(
        "class C:\n"
        "    def foo(self):\n"
        "        return 'method'\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 'module'\n"
    )
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class C:\n    def foo(self):\n        return 'method'\n\n\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "x = 1\n\ndef foo():\n    return 'module'\n"
    )


def test_move_symbol_keeps_a_real_decorator_while_dropping_classmethod(
    tmp_path: Path,
) -> None:
    """@classmethod is shed on the move but any other decorator travels verbatim."""
    (tmp_path / "src.py").write_text(
        "import functools\n"
        "\n"
        "\n"
        "class Old:\n"
        "    @classmethod\n"
        "    @functools.lru_cache(maxsize=None)\n"
        "    def foo(cls, x):\n"
        "        return x + 1\n"
    )
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class=None, dedent=4
    )
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "def z():\n"
        "    return 0\n"
        "\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def foo(cls, x):\n"
        "    return x + 1\n"
    )


def test_move_symbol_leaves_a_comment_above_the_def_in_the_source(
    tmp_path: Path,
) -> None:
    """A comment above the def is not part of its span, so it stays behind in the source."""
    (tmp_path / "src.py").write_text(
        "# explains foo\ndef foo():\n    return 1\n\n\ndef keep():\n    return 2\n"
    )
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "# explains foo\n\n\ndef keep():\n    return 2\n"
    )
    assert (tmp_path / "dst.py").read_text() == "x = 1\n\ndef foo():\n    return 1\n"


def test_move_symbol_without_trailing_newlines_keeps_moved_bytes(
    tmp_path: Path,
) -> None:
    """Files lacking a final newline lose no bytes of the moved def or the remainder."""
    (tmp_path / "src.py").write_text(
        "def keep():\n    return 0\n\n\ndef foo():\n    return 1"
    )
    (tmp_path / "dst.py").write_text("x = 1")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "def keep():\n    return 0\n\n\n"
    assert (tmp_path / "dst.py").read_text() == "x = 1\ndef foo():\n    return 1"


def test_move_symbol_dedent_leaves_string_literal_interior_lines(
    tmp_path: Path,
) -> None:
    """Dedent only strips lines with exactly n leading spaces, so string interiors survive."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    class Deep:\n"
        "        def foo(self):\n"
        "            s = '''raw\n"
        "    partial\n"
        "'''\n"
        "            return s\n"
    )
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class=None, dedent=8
    )
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "import os\n"
        "\n"
        "def foo(self):\n"
        "    s = '''raw\n"
        "    partial\n"
        "'''\n"
        "    return s\n"
    )


def test_move_symbol_asserts_when_destination_class_missing(tmp_path: Path) -> None:
    """Naming an into_class absent from the destination fails loudly."""
    (tmp_path / "src.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class="Nope"
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_staticmethod_inside_moved_body(tmp_path: Path) -> None:
    """A @staticmethod on a nested def inside the moved body must survive the move."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        class Inner:\n"
        "            @staticmethod\n"
        "            def helper(y):\n"
        "                return y\n"
        "        return Inner.helper(x)\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert "            @staticmethod\n            def helper(y):\n" in dst_out


def test_move_symbol_rejects_ambiguous_duplicate_names(tmp_path: Path) -> None:
    """Two same-named defs at equal depth must raise instead of silently picking one."""
    (tmp_path / "src.py").write_text(
        "class A:\n"
        "    def foo(self):\n"
        "        return 'A'\n"
        "\n"
        "class B:\n"
        "    def foo(self):\n"
        "        return 'B'\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_asserts_when_before_sibling_missing(tmp_path: Path) -> None:
    """A before= anchor absent from the destination must raise, not fall back to append."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol(
        "moved", src="src.py", dst="dst.py", into_class=None, before="NO_SUCH_DEF"
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_crlf_line_endings(tmp_path: Path) -> None:
    """Moving a def in a CRLF file must keep every line ending CRLF."""
    (tmp_path / "src.py").write_bytes(
        b"class Old:\r\n    def foo(self):\r\n        return 1\r\n"
    )
    (tmp_path / "dst.py").write_bytes(
        b"class New:\r\n    def keep(self):\r\n        return 0\r\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_bytes = (tmp_path / "dst.py").read_bytes()
    assert dst_bytes.count(b"\n") == dst_bytes.count(b"\r\n")


def test_move_symbol_negative_dedent_indents_into_the_class(tmp_path: Path) -> None:
    """Moving a module-level def into a class with dedent=-4 must indent it as a method."""
    (tmp_path / "src.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "helper", src="src.py", dst="dst.py", into_class="New", dedent=-4
    )
    _apply(r, tmp_path)
    assert "    def helper(x):\n        return x\n" in (tmp_path / "dst.py").read_text()


# --- adversarial audit: leave_delegate stubs -------------------------------------


def test_move_symbol_leave_delegate_keeps_a_multiline_signature_verbatim(
    tmp_path: Path,
) -> None:
    """A multi-line header is carried into the stub byte-for-byte via the bracket scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return self.cfg.compute(n, scale=scale)\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class Cfg:\n"
        "    def e(self):\n"
        "        return 0\n"
        "\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )


def test_move_symbol_leave_delegate_forwards_posonly_vararg_kwonly_kwargs(
    tmp_path: Path,
) -> None:
    """Every parameter kind is forwarded correctly in the delegate's return call."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, a, /, b, *args, c, d=3, **kw):\n"
        "        return a\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(self, a, /, b, *args, c, d=3, **kw):\n"
        "        return self.cfg.compute(a, b, *args, c=c, d=d, **kw)\n"
    )


def test_move_symbol_leave_delegate_survives_paren_in_string_default(
    tmp_path: Path,
) -> None:
    """A string default containing '(' must not break the delegate's header scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        '        sep: str = "(",\n'
        "        n: int = 0,\n"
        "    ) -> int:\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    compile(src_out, "src.py", "exec")
    assert "return self.cfg.compute(sep, n)" in src_out


def test_move_symbol_async_leave_delegate_awaits_the_forwarded_call(
    tmp_path: Path,
) -> None:
    """An async method's delegate stub must await the forwarded coroutine."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    async def compute(self, n):\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert "return await self.cfg.compute(n)" in (tmp_path / "src.py").read_text()


# --- adversarial audit: call-site rewrites ---------------------------------------


def test_requalify_call_sites_matches_a_zero_argument_call(tmp_path: Path) -> None:
    """Owner.bar() with no arguments is requalified to bar()."""
    (tmp_path / "m.py").write_text("y = Owner.bar()\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar()\n"


def test_lower_call_sites_preserves_comments_inside_a_multiline_call(
    tmp_path: Path,
) -> None:
    """A comment between arguments of the rewritten call must survive."""
    (tmp_path / "m.py").write_text(
        "x = Old.foo(\n    self.r,\n    a,  # keep me\n    b,\n)\n"
    )
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "# keep me" in (tmp_path / "m.py").read_text()


def test_lower_call_sites_preserves_arg_literal_spelling(tmp_path: Path) -> None:
    """Hex literals and quote styles inside the rewritten call must not be normalized."""
    (tmp_path / "m.py").write_text('x = Old.foo(self.r, 0x10, "s")\n')
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = self.r.foo(0x10, "s")\n'


def test_lower_call_sites_lowers_a_nested_matching_call_too(tmp_path: Path) -> None:
    """A matching call nested inside another matching call is lowered as well."""
    (tmp_path / "m.py").write_text("x = Old.foo(self.r, Old.foo(self.q, 1))\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(self.q.foo(1))\n"


def test_lower_call_sites_magic_comma_with_sole_receiver_arg_stays_valid(
    tmp_path: Path,
) -> None:
    """Lowering a magic-comma call whose only argument is the receiver stays valid Python."""
    (tmp_path / "m.py").write_text("Owner.foo(\n    self.r,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    compile(out, "m.py", "exec")


def test_call_rewrite_is_column_accurate_on_non_ascii_lines(tmp_path: Path) -> None:
    """A call after a non-ASCII string on the same line is rewritten at the right columns."""
    (tmp_path / "m.py").write_text('x = "中文"; y = Owner.foo(self.r, 1)\n')
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = "中文"; y = self.r.foo(1)\n'


def test_call_rewrite_survives_a_form_feed_line_start(tmp_path: Path) -> None:
    """A form feed at a line start must not shift the rewrite onto the wrong line."""
    (tmp_path / "m.py").write_text("a = 1\n\x0cb = 2\ny = Owner.foo(self.r, 1)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "a = 1\n\x0cb = 2\ny = self.r.foo(1)\n"


def test_requalify_call_sites_preserves_redundant_parens_in_kwargs(
    tmp_path: Path,
) -> None:
    """Redundant parentheses around a keyword value survive the requalification."""
    (tmp_path / "m.py").write_text("y = Old.bar(\n    a=1,\n    b=(2),\n)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "b=(2)" in (tmp_path / "m.py").read_text()


# --- adversarial audit: import primitives ----------------------------------------


def test_remove_import_unscoped_removes_module_level_import_and_blank(
    tmp_path: Path,
) -> None:
    """Without in_function the matching module-level import and its trailing blank go."""
    (tmp_path / "m.py").write_text("import os\nfrom pkg import Thing\n\nx = Thing\n")
    r = Repro("b", "t").remove_import("m.py", "from pkg import Thing")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\nx = Thing\n"


def test_remove_import_keeps_a_code_line_directly_after_the_import(
    tmp_path: Path,
) -> None:
    """Only a blank line after the import is absorbed; a code line stays untouched."""
    (tmp_path / "m.py").write_text("import os\nx = 1\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = 1\n"


def test_remove_import_asserts_when_text_absent(tmp_path: Path) -> None:
    """Removing an import text that matches nothing fails loudly."""
    (tmp_path / "m.py").write_text("import os\n")
    r = Repro("b", "t").remove_import("m.py", "from pkg import Q")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_remove_import_asserts_when_scope_function_missing(tmp_path: Path) -> None:
    """Scoping to a function that does not exist fails loudly."""
    (tmp_path / "m.py").write_text("def f():\n    import os\n")
    r = Repro("b", "t").remove_import("m.py", "import os", in_function="nope")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


@pytest.mark.xfail(
    strict=True,
    reason="whole source lines are deleted for any statement whose text matches, so "
    "`import os; import sys` loses the sys import (and the next line) too",
)
def test_remove_import_leaves_other_statements_on_a_semicolon_line(
    tmp_path: Path,
) -> None:
    """Removing 'import os' from a semicolon-joined line must keep 'import sys'."""
    (tmp_path / "m.py").write_text("import os; import sys\nprint(sys.path)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert "import sys" in out and "print(sys.path)" in out


@pytest.mark.xfail(
    strict=True,
    reason="the match is a plain substring, so 'import os' also deletes the separate "
    "'import os.path' statement",
)
def test_remove_import_does_not_overmatch_a_submodule_import(tmp_path: Path) -> None:
    """Removing 'import os' must not also remove 'import os.path'."""
    (tmp_path / "m.py").write_text("import os\nimport os.path\nprint(os.path.sep)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    assert "import os.path\n" in (tmp_path / "m.py").read_text()


def test_remove_imported_name_collapses_a_multiline_import_to_one_line(
    tmp_path: Path,
) -> None:
    """Pruning a name from a parenthesized import rebuilds it as a single sorted-later line."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n    a,\n    moved,\n    b,\n)\n\nx = a + b\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a, b\n\nx = a + b\n"


def test_remove_imported_name_matches_a_relative_module(tmp_path: Path) -> None:
    """A relative `from .pkg import` is matched via its level dots."""
    (tmp_path / "m.py").write_text("from .pkg import a, moved\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name("m.py", module=".pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from .pkg import a\n\nx = a\n"


@pytest.mark.xfail(
    strict=True,
    reason="rebuilding a multi-line parenthesized import from its aliases silently "
    "deletes the comments that sat on its lines",
)
def test_remove_imported_name_preserves_comments_in_a_multiline_import(
    tmp_path: Path,
) -> None:
    """Comments on surviving lines of a pruned parenthesized import must not vanish."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n"
        "    a,  # used by frobnicator\n"
        "    moved,\n"
        "    b,\n"
        ")\n"
        "\n"
        "x = a + b\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert "# used by frobnicator" in (tmp_path / "m.py").read_text()


def test_add_import_into_an_empty_file(tmp_path: Path) -> None:
    """Adding an import to an empty file writes just the statement."""
    (tmp_path / "m.py").write_text("")
    r = Repro("b", "t").add_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "import os\n"


@pytest.mark.xfail(
    strict=True,
    reason="with no existing imports the statement is inserted at line 0, above the "
    "module docstring, demoting the docstring to a plain expression",
)
def test_add_import_lands_below_a_module_docstring(tmp_path: Path) -> None:
    """In a file with only a docstring, the new import must land below the docstring."""
    (tmp_path / "m.py").write_text('"""Module doc."""\n\nx = 1\n')
    r = Repro("b", "t").add_import("m.py", "import os")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text().startswith('"""Module doc."""')


def test_add_typechecking_import_matches_qualified_typing_form(tmp_path: Path) -> None:
    """A `if typing.TYPE_CHECKING:` block is recognized and receives the import."""
    (tmp_path / "m.py").write_text(
        "import typing\n"
        "\n"
        "if typing.TYPE_CHECKING:\n"
        "    from a import X\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "import typing\n"
        "\n"
        "if typing.TYPE_CHECKING:\n"
        "    from a import X\n"
        "    from b import Y\n"
        "\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )


def test_add_typechecking_import_after_a_multiline_final_import(tmp_path: Path) -> None:
    """The insert lands after the closing paren of a multi-line final guarded import."""
    (tmp_path / "m.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import (\n"
        "        X,\n"
        "    )\n"
        "\n"
        "x = 1\n"
    )
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "from typing import TYPE_CHECKING\n"
        "\n"
        "if TYPE_CHECKING:\n"
        "    from a import (\n"
        "        X,\n"
        "    )\n"
        "    from b import Y\n"
        "\n"
        "x = 1\n"
    )


def test_add_typechecking_import_raises_without_a_block(tmp_path: Path) -> None:
    """A file lacking a TYPE_CHECKING block fails loudly."""
    (tmp_path / "m.py").write_text("import os\n\nx = 1\n")
    r = Repro("b", "t").add_typechecking_import("m.py", "from b import Y")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_repath_import_repaths_a_multiline_aliased_nested_import(
    tmp_path: Path,
) -> None:
    """A nested multi-line from-import with an alias is repathed on its first line."""
    (tmp_path / "c.py").write_text(
        "def run():\n"
        "    from old.mod import (\n"
        "        foo as f,\n"
        "    )\n"
        "\n"
        "    return f(1)\n"
    )
    r = Repro("b", "t").repath_import(
        "c.py", old_module="old.mod", new_module="new.mod", name="foo"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "c.py").read_text() == (
        "def run():\n"
        "    from new.mod import (\n"
        "        foo as f,\n"
        "    )\n"
        "\n"
        "    return f(1)\n"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a relative nested import matches on node.module but the textual replace "
    "finds nothing, so the op is a silent no-op that still asserts success",
)
def test_repath_import_rewrites_a_relative_nested_import(tmp_path: Path) -> None:
    """A nested `from .mod import` matched by module name must actually be repathed."""
    (tmp_path / "c.py").write_text(
        "def run():\n    from .mod import foo\n\n    return foo(1)\n"
    )
    r = Repro("b", "t").repath_import(
        "c.py", old_module="mod", new_module="pkg.mod", name="foo"
    )
    _apply(r, tmp_path)
    assert "from pkg.mod import foo" in (tmp_path / "c.py").read_text()


# --- adversarial audit: module extraction ----------------------------------------


def test_extract_to_new_module_asserts_when_symbol_not_in_the_tail(
    tmp_path: Path,
) -> None:
    """A wanted symbol above a non-scaffolding statement is not in the tail and raises."""
    (tmp_path / "src.py").write_text(
        "def wanted():\n    return 1\n\n\nprint('side effect')\n"
    )
    r = Repro("b", "t").extract_to_new_module("src.py", "n.py", symbols=["wanted"])
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


@pytest.mark.xfail(
    strict=True,
    reason="the tail scan treats any ast.If as scaffolding, so a trailing "
    "`if __name__ == '__main__':` block is silently relocated into the new module",
)
def test_extract_to_new_module_keeps_a_trailing_main_guard_in_source(
    tmp_path: Path,
) -> None:
    """A trailing __main__ guard is not a moved symbol and must stay in the source."""
    (tmp_path / "src.py").write_text(
        "class Keep:\n"
        "    pass\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    foo()\n"
    )
    r = Repro("b", "t").extract_to_new_module(
        "src.py", "new.py", symbols=["foo"], future_import=False
    )
    _apply(r, tmp_path)
    assert "__main__" in (tmp_path / "src.py").read_text()


def test_extract_symbols_to_new_module_joins_blocks_with_two_blank_lines(
    tmp_path: Path,
) -> None:
    """Relocated blocks are joined with exactly two blank lines (the formatter's spacing)."""
    (tmp_path / "src.py").write_text(
        "def moved_a():\n    return 1\n\n\n\n\ndef moved_b():\n    return 2\n"
    )
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py",
        "new.py",
        symbols=["moved_a", "moved_b"],
        header="",
        order=["moved_a", "moved_b"],
    )
    _apply(r, tmp_path)
    assert (tmp_path / "new.py").read_text() == (
        "def moved_a():\n    return 1\n\n\ndef moved_b():\n    return 2\n"
    )


def test_extract_symbols_to_new_module_leaves_a_comment_above_a_moved_def(
    tmp_path: Path,
) -> None:
    """A section comment directly above a moved def stays behind in the source."""
    (tmp_path / "src.py").write_text(
        "x = 1\n\n\n# --- movers ---\ndef moved():\n    return 2\n"
    )
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py", "new.py", symbols=["moved"], header="", order=["moved"]
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "x = 1\n\n\n# --- movers ---\n"
    assert (tmp_path / "new.py").read_text() == "def moved():\n    return 2\n"


@pytest.mark.xfail(
    strict=True,
    reason="drop_assigns deletes the whole statement when any target matches, so "
    "`A = B = 1` silently loses B's binding from the source",
)
def test_extract_symbols_drop_assigns_preserves_other_targets_of_chained_assign(
    tmp_path: Path,
) -> None:
    """Dropping A from `A = B = 1` must not delete B's binding from the source."""
    (tmp_path / "src.py").write_text("A = B = 1\n\n\ndef moved():\n    return A\n")
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py",
        "new.py",
        symbols=["moved"],
        header="A = 1\n",
        order=["moved"],
        drop_assigns=["A"],
    )
    _apply(r, tmp_path)
    assert "B" in (tmp_path / "src.py").read_text()


@pytest.mark.xfail(
    strict=True,
    reason="delete_file has no emptiness guard, so deleting a module that still holds "
    "live definitions is silently certifiable",
)
def test_delete_file_refuses_a_file_with_remaining_definitions(tmp_path: Path) -> None:
    """Deleting a module that still contains defs must fail loudly."""
    (tmp_path / "live.py").write_text("def still_used():\n    return 42\n")
    r = Repro("b", "t").delete_file("live.py")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
    assert (tmp_path / "live.py").exists()


def test_delete_file_on_a_missing_path_is_a_no_op(tmp_path: Path) -> None:
    """Deleting an already-absent file does nothing and raises nothing."""
    r = Repro("b", "t").delete_file("nope.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "nope.py").exists()


# --- adversarial audit: extract_function -----------------------------------------


def test_extract_function_does_not_pad_blank_lines_in_the_body(tmp_path: Path) -> None:
    """Interior blank lines of the extracted body stay bare newlines, unpadded."""
    (tmp_path / "src.py").write_text("        a = 1\n\n        b = 2\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="        a = 1\n\n        b = 2\n",
        body_indent=8,
        call="        g()\n",
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "        g()\n"
    assert (tmp_path / "dst.py").read_text() == (
        "def z():\n    return 0\n\ndef g():\n    a = 1\n\n    b = 2\n"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a positive reindent pads every non-blank line, shifting triple-quoted "
    "string interiors and changing the literal's value",
)
def test_extract_function_does_not_reindent_string_literal_interiors(
    tmp_path: Path,
) -> None:
    """Triple-quoted string interior lines keep their exact bytes through the extraction."""
    (tmp_path / "src.py").write_text(
        "TEMPLATE = '''\nliteral line\n'''\nx = TEMPLATE\n"
    )
    (tmp_path / "dst.py").write_text("def existing():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="make",
        signature="def make():",
        body="TEMPLATE = '''\nliteral line\n'''\nx = TEMPLATE\n",
        body_indent=0,
        call="x = make()\n",
        return_text="    return x\n",
    )
    _apply(r, tmp_path)
    assert "\nliteral line\n" in (tmp_path / "dst.py").read_text()


@pytest.mark.xfail(
    strict=True,
    reason="the body is matched with a raw substring count/replace, so a mid-line "
    "match splices the call into the middle of an unrelated statement",
)
def test_extract_function_rejects_a_mid_line_substring_match(tmp_path: Path) -> None:
    """A body that only matches mid-line must fail loudly instead of splicing the call."""
    (tmp_path / "src.py").write_text("value = prefix_total = 0\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="g",
        signature="def g():",
        body="total = 0\n",
        body_indent=0,
        call="total = g()\n",
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


@pytest.mark.xfail(
    strict=True,
    reason="the body is always reindented to depth 4 (a module-level function body), "
    "so into_class extraction emits a method with a mis-indented, invalid body",
)
def test_extract_function_into_class_indents_body_to_method_depth(
    tmp_path: Path,
) -> None:
    """Extracting into a class must indent the relocated body to method depth."""
    (tmp_path / "src.py").write_text("val = compute_thing()\n")
    (tmp_path / "dst.py").write_text(
        "class H:\n    def last(self):\n        return 0\n"
    )
    r = Repro("b", "t").extract_function(
        "src.py",
        "dst.py",
        name="helper",
        signature="    def helper(self):",
        body="val = compute_thing()\n",
        body_indent=0,
        call="val = h.helper()\n",
        return_text="        return val\n",
        into_class="H",
    )
    _apply(r, tmp_path)
    out = (tmp_path / "dst.py").read_text()
    compile(out, "dst.py", "exec")
    assert "    def helper(self):\n        val = compute_thing()\n" in out
