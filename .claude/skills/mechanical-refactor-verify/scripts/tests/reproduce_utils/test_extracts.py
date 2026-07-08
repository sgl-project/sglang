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


def test_extract_to_new_module_refuses_a_trailing_main_guard(tmp_path: Path) -> None:
    """A trailing __main__ guard is executable code, not scaffolding: the tail cut raises."""
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
    with pytest.raises(AssertionError):
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
