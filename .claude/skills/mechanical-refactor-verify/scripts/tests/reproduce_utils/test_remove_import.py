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


def test_remove_import_leaves_other_statements_on_a_semicolon_line(
    tmp_path: Path,
) -> None:
    """Removing 'import os' from a semicolon-joined line must keep 'import sys'."""
    (tmp_path / "m.py").write_text("import os; import sys\nprint(sys.path)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    assert "import sys" in out and "print(sys.path)" in out


def test_remove_import_does_not_overmatch_a_submodule_import(tmp_path: Path) -> None:
    """Removing 'import os' must not also remove 'import os.path'."""
    (tmp_path / "m.py").write_text("import os\nimport os.path\nprint(os.path.sep)\n")
    r = Repro("b", "t").remove_import("m.py", "import os")
    _apply(r, tmp_path)
    assert "import os.path\n" in (tmp_path / "m.py").read_text()
