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
