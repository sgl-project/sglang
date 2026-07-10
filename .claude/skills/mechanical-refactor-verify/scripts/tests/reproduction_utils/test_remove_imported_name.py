import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mechanical_refactor_reproduction_utils as rr
from mechanical_refactor_reproduction_utils import (
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
from reproduction_testlib import _apply, _commit, _git, _write  # noqa: F401

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


def test_remove_imported_name_preserves_the_multiline_form(
    tmp_path: Path,
) -> None:
    """Pruning a name from an exploded import deletes only that line, so the parens and the
    magic trailing comma survive and the formatter keeps it multi-line (a flat rebuild would
    collapse an import the target left multi-line)."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n    a,\n    moved,\n    b,\n)\n\nx = a + b\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (
        tmp_path / "m.py"
    ).read_text() == "from pkg import (\n    a,\n    b,\n)\n\nx = a + b\n"


def test_remove_imported_name_multiline_down_to_one_collapses(
    tmp_path: Path,
) -> None:
    """Pruning an exploded import down to a single surviving name collapses it to one line:
    the formatter does not keep a lone name exploded, so a preserved-multiline form would
    not match the target."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n    moved,\n    a,\n)\n\nx = a\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "from pkg import a\n\nx = a\n"


def test_remove_imported_name_down_to_one_with_a_comment_stays_exploded(
    tmp_path: Path,
) -> None:
    """A lone survivor that carries a comment stays exploded (a rebuild would drop the
    comment); only its own line is deleted."""
    (tmp_path / "m.py").write_text(
        "from pkg import (\n    moved,\n    a,  # keep me\n)\n\nx = a\n"
    )
    r = Repro("b", "t").remove_imported_name("m.py", module="pkg", name="moved")
    _apply(r, tmp_path)
    assert (
        tmp_path / "m.py"
    ).read_text() == "from pkg import (\n    a,  # keep me\n)\n\nx = a\n"


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


def test_remove_imported_name_keep_exploded_holds_a_lone_survivor_multiline(
    tmp_path: Path,
) -> None:
    """With keep_exploded, pruning down to a single survivor deletes only the removed line,
    so the survivor keeps its magic trailing comma and the import stays multi-line (the
    author's choice, which the source cannot reveal). A regenerating impl would collapse."""
    (tmp_path / "m.py").write_text("from pkg import (\n    moved,\n    a,\n)\n\nx = a\n")
    r = Repro("b", "t").remove_imported_name(
        "m.py", module="pkg", name="moved", keep_exploded=True
    )
    _apply(r, tmp_path)
    assert (
        tmp_path / "m.py"
    ).read_text() == "from pkg import (\n    a,\n)\n\nx = a\n"
