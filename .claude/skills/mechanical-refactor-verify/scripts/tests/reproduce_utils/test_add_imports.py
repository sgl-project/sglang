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
