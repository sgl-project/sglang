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
