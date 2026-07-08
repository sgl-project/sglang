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


def test_lowered_call_text_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    """A magic trailing comma in the original call survives the textual lowering."""
    (tmp_path / "m.py").write_text("x = Old.foo(\n    self.r,\n    a,\n    b,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(\n    a,\n    b,\n)\n"


# --- lower_call_sites ----------------------------------------------------------


def test_lower_call_sites_moves_receiver_out_of_args(tmp_path: Path) -> None:
    """Owner.foo(receiver, rest) becomes receiver.foo(rest)."""
    (tmp_path / "m.py").write_text("x = ModelRunner.foo(self.r, a, b)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(a, b)\n"


def test_lower_call_sites_handles_only_receiver_arg(tmp_path: Path) -> None:
    """Owner.foo(receiver) becomes receiver.foo() without re-lowering the result."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(self.r)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo()\n"


def test_lower_call_sites_ignores_a_different_owner(tmp_path: Path) -> None:
    """A same-named call on another receiver (e.g. the moved body's own call) is untouched."""
    (tmp_path / "m.py").write_text("worker.foo(zmq)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "worker.foo(zmq)\n"


def test_lower_call_sites_preserves_magic_trailing_comma(tmp_path: Path) -> None:
    """A magic trailing comma is kept so the formatter re-explodes the lowered call."""
    (tmp_path / "m.py").write_text("ModelRunner.foo(\n    self.r,\n    a,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "self.r.foo(\n    a,\n)\n"


# --- requalify_call_sites ------------------------------------------------------


# --- requalify_call_sites ------------------------------------------------------


def test_requalify_call_sites_drops_the_qualifier(tmp_path: Path) -> None:
    """Owner.bar(args) becomes bar(args) when bar moves to a free function."""
    (tmp_path / "m.py").write_text("y = ModelRunner.bar(a, b)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "ModelRunner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar(a, b)\n"


# --- adversarial audit: call-site rewrites ---------------------------------------


# --- adversarial audit: call-site rewrites ---------------------------------------


def test_requalify_call_sites_matches_a_zero_argument_call(tmp_path: Path) -> None:
    """Owner.bar() with no arguments is requalified to bar()."""
    (tmp_path / "m.py").write_text("y = Owner.bar()\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "y = bar()\n"


def test_lower_call_sites_preserves_comments_inside_a_multiline_call(
    tmp_path: Path,
) -> None:
    """A comment between arguments of the rewritten call must survive."""
    (tmp_path / "m.py").write_text(
        "x = Old.foo(\n    self.r,\n    a,  # keep me\n    b,\n)\n"
    )
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "# keep me" in (tmp_path / "m.py").read_text()


def test_lower_call_sites_preserves_arg_literal_spelling(tmp_path: Path) -> None:
    """Hex literals and quote styles inside the rewritten call must not be normalized."""
    (tmp_path / "m.py").write_text('x = Old.foo(self.r, 0x10, "s")\n')
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = self.r.foo(0x10, "s")\n'


def test_lower_call_sites_lowers_a_nested_matching_call_too(tmp_path: Path) -> None:
    """A matching call nested inside another matching call is lowered as well."""
    (tmp_path / "m.py").write_text("x = Old.foo(self.r, Old.foo(self.q, 1))\n")
    r = Repro("b", "t").lower_call_sites("foo", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "x = self.r.foo(self.q.foo(1))\n"


def test_lower_call_sites_magic_comma_with_sole_receiver_arg_stays_valid(
    tmp_path: Path,
) -> None:
    """Lowering a magic-comma call whose only argument is the receiver stays valid Python."""
    (tmp_path / "m.py").write_text("Owner.foo(\n    self.r,\n)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    out = (tmp_path / "m.py").read_text()
    compile(out, "m.py", "exec")


def test_call_rewrite_is_column_accurate_on_non_ascii_lines(tmp_path: Path) -> None:
    """A call after a non-ASCII string on the same line is rewritten at the right columns."""
    (tmp_path / "m.py").write_text('x = "中文"; y = Owner.foo(self.r, 1)\n')
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == 'x = "中文"; y = self.r.foo(1)\n'


def test_call_rewrite_survives_a_form_feed_line_start(tmp_path: Path) -> None:
    """A form feed at a line start must not shift the rewrite onto the wrong line."""
    (tmp_path / "m.py").write_text("a = 1\n\x0cb = 2\ny = Owner.foo(self.r, 1)\n")
    r = Repro("b", "t").lower_call_sites("foo", "Owner", paths=["m.py"])
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == "a = 1\n\x0cb = 2\ny = self.r.foo(1)\n"


def test_requalify_call_sites_preserves_redundant_parens_in_kwargs(
    tmp_path: Path,
) -> None:
    """Redundant parentheses around a keyword value survive the requalification."""
    (tmp_path / "m.py").write_text("y = Old.bar(\n    a=1,\n    b=(2),\n)\n")
    r = Repro("b", "t").requalify_call_sites("bar", "Old", paths=["m.py"])
    _apply(r, tmp_path)
    assert "b=(2)" in (tmp_path / "m.py").read_text()
