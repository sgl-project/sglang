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
