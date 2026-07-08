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


def test_delete_file_removes_emptied_source(tmp_path: Path) -> None:
    """delete_file removes a source module left empty after its defs relocated."""
    (tmp_path / "gone.py").write_text("import os\n")
    r = Repro("b", "t").delete_file("gone.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "gone.py").exists()


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
