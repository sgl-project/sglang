import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_reproduction_utils import Repro
from reproduction_testlib import _apply  # noqa: F401


def test_move_assign_relocates_a_module_constant(tmp_path: Path) -> None:
    """The assignment is cut verbatim from the source and lands after the destination's imports."""
    (tmp_path / "src.py").write_text(
        "import os\n\nLIMIT = 480  # seconds\n\n\ndef stay():\n    return LIMIT\n"
    )
    (tmp_path / "dst.py").write_text("import sys\n\n\ndef keep():\n    return 1\n")
    r = Repro("b", "t").move_assign("LIMIT", src="src.py", dst="dst.py")
    _apply(r, tmp_path)
    assert "LIMIT" not in (tmp_path / "src.py").read_text().split("def stay")[0]
    assert (tmp_path / "dst.py").read_text() == (
        "import sys\n"
        "\n"
        "LIMIT = 480  # seconds\n"
        "\n"
        "\n"
        "def keep():\n"
        "    return 1\n"
    )


def test_move_assign_pastes_above_the_named_sibling(tmp_path: Path) -> None:
    """With before=, the constant lands immediately above the named top-level statement."""
    (tmp_path / "src.py").write_text("RATIO = 3\n")
    (tmp_path / "dst.py").write_text("def first():\n    return 1\n")
    r = Repro("b", "t").move_assign("RATIO", src="src.py", dst="dst.py", before="first")
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "RATIO = 3\n\ndef first():\n    return 1\n"
    )


def test_move_assign_missing_source_raises(tmp_path: Path) -> None:
    """A name with no module-level assignment in the source fails loudly."""
    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_assign("MISSING", src="src.py", dst="dst.py")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
