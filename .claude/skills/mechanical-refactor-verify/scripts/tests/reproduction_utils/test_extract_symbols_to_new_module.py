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


def test_extract_symbols_to_new_module_allows_a_rederived_surviving_constant(
    tmp_path: Path,
) -> None:
    """A header constant that also survives verbatim in the source (re-derived boilerplate,
    e.g. `_is_hip = is_hip()`) is allowed: it is provably not fiction because the same
    statement remains in the source."""
    (tmp_path / "src.py").write_text(
        "from pkg import is_hip\n"
        "\n"
        "_is_hip = is_hip()\n"
        "\n"
        "\n"
        "def moved():\n"
        "    return _is_hip\n"
    )
    header = "from pkg import is_hip\n\n_is_hip = is_hip()\n"
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py", "new.py", symbols=["moved"], header=header, order=["moved"]
    )
    _apply(r, tmp_path)
    assert "_is_hip = is_hip()" in (tmp_path / "src.py").read_text()
    assert "_is_hip = is_hip()" in (tmp_path / "new.py").read_text()


def test_extract_symbols_to_new_module_rejects_a_fictional_header_constant(
    tmp_path: Path,
) -> None:
    """A header constant that is neither dropped from nor surviving in the source is fiction
    and raises: the audit refuses code the extraction cannot vouch for."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    r = Repro("b", "t").extract_symbols_to_new_module(
        "src.py",
        "new.py",
        symbols=["moved"],
        header="_fake = evil()\n",
        order=["moved"],
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)
