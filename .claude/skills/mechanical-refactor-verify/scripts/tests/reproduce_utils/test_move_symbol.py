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


# --- move_symbol ---------------------------------------------------------------


def test_move_symbol_into_class_drops_decorator_and_appends(tmp_path: Path) -> None:
    """The def leaves the source, its @staticmethod is dropped, and it lands at the end of
    the destination class with its body verbatim."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        return x + 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def existing(self):\n        return 1\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def foo" not in src_out and "def keep" in src_out
    assert "@staticmethod" not in dst_out
    assert dst_out.index("def existing") < dst_out.index("def foo")
    assert "        return x + 1\n" in dst_out


def test_move_symbol_to_module_level_with_dedent(tmp_path: Path) -> None:
    """With into_class=None and a dedent, the def lands at module level, dedented."""
    (tmp_path / "src.py").write_text(
        "class Old:\n    @staticmethod\n    def helper(x):\n        return x * 2\n"
    )
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol(
        "helper", src="src.py", dst="dst.py", into_class=None, dedent=4
    )
    _apply(r, tmp_path)
    assert "def helper(x):\n    return x * 2\n" in (tmp_path / "dst.py").read_text()
    assert "def helper" not in (tmp_path / "src.py").read_text()


def test_move_symbol_before_inserts_above_named_sibling(tmp_path: Path) -> None:
    """With before=, the relocated def lands immediately above that sibling, not at the end."""
    (tmp_path / "src.py").write_text(
        "class Old:\n    @staticmethod\n    def moved(self):\n        return 1\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n"
        "    def first(self):\n        return 0\n"
        "\n"
        "    def last(self):\n        return 2\n"
    )
    r = Repro("b", "t").move_symbol(
        "moved", src="src.py", dst="dst.py", into_class="New", before="last"
    )
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert (
        dst_out.index("def first")
        < dst_out.index("def moved")
        < dst_out.index("def last")
    )


def test_delete_file_removes_emptied_source(tmp_path: Path) -> None:
    """delete_file removes a source module left empty after its defs relocated."""
    (tmp_path / "gone.py").write_text("import os\n")
    r = Repro("b", "t").delete_file("gone.py")
    _apply(r, tmp_path)
    assert not (tmp_path / "gone.py").exists()


def test_move_symbol_leave_delegate_keeps_forwarding_stub(tmp_path: Path) -> None:
    """With leave_delegate, the source keeps a forwarding stub through the named field and the
    destination gets the full method body."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def existing(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute",
        src="src.py",
        dst="dst.py",
        into_class="Cfg",
        leave_delegate="cfg",
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "def compute(self, n: int) -> int:" in src_out
    assert "return self.cfg.compute(n)" in src_out
    assert "return n + self.cfg.base" not in src_out
    assert "return n + self.cfg.base" in dst_out


def test_move_symbol_leave_delegate_does_not_absorb_leading_comments(
    tmp_path: Path,
) -> None:
    """A leading comment before the first statement is not an AST node, so it must not be
    pulled into the forwarding stub -- the delegate is just the header plus the return.
    """
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        # explain the maths\n"
        "        # second comment line\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def existing(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    dst_out = (tmp_path / "dst.py").read_text()
    assert "# explain the maths" not in src_out
    assert (
        src_out == "class Mixin:\n"
        "    def compute(self, n: int) -> int:\n"
        "        return self.cfg.compute(n)\n"
    )
    assert "# explain the maths" in dst_out


# --- adversarial audit: move_symbol edge cases -----------------------------------


def test_move_symbol_moves_an_async_def_verbatim(tmp_path: Path) -> None:
    """An async def relocates with its `async` keyword and body byte-identical."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    async def foo(self):\n"
        "        return 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Old:\n\n    def keep(self):\n        return 0\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class New:\n"
        "    def e(self):\n"
        "        return 1\n"
        "\n"
        "    async def foo(self):\n"
        "        return 1\n"
    )


def test_move_symbol_moves_a_def_within_the_same_file(tmp_path: Path) -> None:
    """With src == dst the def is cut and re-inserted above its sibling in one file."""
    (tmp_path / "m.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    )
    r = Repro("b", "t").move_symbol(
        "b", src="m.py", dst="m.py", into_class=None, before="a"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "m.py").read_text() == (
        "def b():\n    return 2\n\ndef a():\n    return 1\n\n\n"
    )


def test_move_symbol_prefers_a_module_level_def_over_an_earlier_class_method(
    tmp_path: Path,
) -> None:
    """When a class method and a module-level def share a name, the module-level def moves."""
    (tmp_path / "src.py").write_text(
        "class C:\n"
        "    def foo(self):\n"
        "        return 'method'\n"
        "\n"
        "\n"
        "def foo():\n"
        "    return 'module'\n"
    )
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class C:\n    def foo(self):\n        return 'method'\n\n\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "x = 1\n\ndef foo():\n    return 'module'\n"
    )


def test_move_symbol_keeps_a_real_decorator_while_dropping_classmethod(
    tmp_path: Path,
) -> None:
    """@classmethod is shed on the move but any other decorator travels verbatim."""
    (tmp_path / "src.py").write_text(
        "import functools\n"
        "\n"
        "\n"
        "class Old:\n"
        "    @classmethod\n"
        "    @functools.lru_cache(maxsize=None)\n"
        "    def foo(cls, x):\n"
        "        return x + 1\n"
    )
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class=None, dedent=4
    )
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "def z():\n"
        "    return 0\n"
        "\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def foo(cls, x):\n"
        "    return x + 1\n"
    )


def test_move_symbol_leaves_a_comment_above_the_def_in_the_source(
    tmp_path: Path,
) -> None:
    """A comment above the def is not part of its span, so it stays behind in the source."""
    (tmp_path / "src.py").write_text(
        "# explains foo\ndef foo():\n    return 1\n\n\ndef keep():\n    return 2\n"
    )
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "# explains foo\n\n\ndef keep():\n    return 2\n"
    )
    assert (tmp_path / "dst.py").read_text() == "x = 1\n\ndef foo():\n    return 1\n"


def test_move_symbol_without_trailing_newlines_keeps_moved_bytes(
    tmp_path: Path,
) -> None:
    """Files lacking a final newline lose no bytes of the moved def or the remainder."""
    (tmp_path / "src.py").write_text(
        "def keep():\n    return 0\n\n\ndef foo():\n    return 1"
    )
    (tmp_path / "dst.py").write_text("x = 1")
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class=None)
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == "def keep():\n    return 0\n\n\n"
    assert (tmp_path / "dst.py").read_text() == "x = 1\ndef foo():\n    return 1"


def test_move_symbol_dedent_leaves_string_literal_interior_lines(
    tmp_path: Path,
) -> None:
    """Dedent only strips lines with exactly n leading spaces, so string interiors survive."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    class Deep:\n"
        "        def foo(self):\n"
        "            s = '''raw\n"
        "    partial\n"
        "'''\n"
        "            return s\n"
    )
    (tmp_path / "dst.py").write_text("import os\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class=None, dedent=8
    )
    _apply(r, tmp_path)
    assert (tmp_path / "dst.py").read_text() == (
        "import os\n"
        "\n"
        "def foo(self):\n"
        "    s = '''raw\n"
        "    partial\n"
        "'''\n"
        "    return s\n"
    )


def test_move_symbol_asserts_when_destination_class_missing(tmp_path: Path) -> None:
    """Naming an into_class absent from the destination fails loudly."""
    (tmp_path / "src.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "dst.py").write_text("x = 1\n")
    r = Repro("b", "t").move_symbol(
        "foo", src="src.py", dst="dst.py", into_class="Nope"
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_staticmethod_inside_moved_body(tmp_path: Path) -> None:
    """A @staticmethod on a nested def inside the moved body must survive the move."""
    (tmp_path / "src.py").write_text(
        "class Old:\n"
        "    @staticmethod\n"
        "    def foo(x):\n"
        "        class Inner:\n"
        "            @staticmethod\n"
        "            def helper(y):\n"
        "                return y\n"
        "        return Inner.helper(x)\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_out = (tmp_path / "dst.py").read_text()
    assert "            @staticmethod\n            def helper(y):\n" in dst_out


def test_move_symbol_rejects_ambiguous_duplicate_names(tmp_path: Path) -> None:
    """Two same-named defs at equal depth must raise instead of silently picking one."""
    (tmp_path / "src.py").write_text(
        "class A:\n"
        "    def foo(self):\n"
        "        return 'A'\n"
        "\n"
        "class B:\n"
        "    def foo(self):\n"
        "        return 'B'\n"
    )
    (tmp_path / "dst.py").write_text(
        "class New:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_asserts_when_before_sibling_missing(tmp_path: Path) -> None:
    """A before= anchor absent from the destination must raise, not fall back to append."""
    (tmp_path / "src.py").write_text("def moved():\n    return 1\n")
    (tmp_path / "dst.py").write_text("def z():\n    return 0\n")
    r = Repro("b", "t").move_symbol(
        "moved", src="src.py", dst="dst.py", into_class=None, before="NO_SUCH_DEF"
    )
    with pytest.raises(AssertionError):
        _apply(r, tmp_path)


def test_move_symbol_preserves_crlf_line_endings(tmp_path: Path) -> None:
    """Moving a def in a CRLF file must keep every line ending CRLF."""
    (tmp_path / "src.py").write_bytes(
        b"class Old:\r\n    def foo(self):\r\n        return 1\r\n"
    )
    (tmp_path / "dst.py").write_bytes(
        b"class New:\r\n    def keep(self):\r\n        return 0\r\n"
    )
    r = Repro("b", "t").move_symbol("foo", src="src.py", dst="dst.py", into_class="New")
    _apply(r, tmp_path)
    dst_bytes = (tmp_path / "dst.py").read_bytes()
    assert dst_bytes.count(b"\n") == dst_bytes.count(b"\r\n")


def test_move_symbol_negative_dedent_indents_into_the_class(tmp_path: Path) -> None:
    """Moving a module-level def into a class with dedent=-4 must indent it as a method."""
    (tmp_path / "src.py").write_text("def helper(x):\n    return x\n")
    (tmp_path / "dst.py").write_text("class New:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "helper", src="src.py", dst="dst.py", into_class="New", dedent=-4
    )
    _apply(r, tmp_path)
    assert "    def helper(x):\n        return x\n" in (tmp_path / "dst.py").read_text()


# --- adversarial audit: leave_delegate stubs -------------------------------------


def test_move_symbol_leave_delegate_keeps_a_multiline_signature_verbatim(
    tmp_path: Path,
) -> None:
    """A multi-line header is carried into the stub byte-for-byte via the bracket scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return self.cfg.compute(n, scale=scale)\n"
    )
    assert (tmp_path / "dst.py").read_text() == (
        "class Cfg:\n"
        "    def e(self):\n"
        "        return 0\n"
        "\n"
        "    def compute(\n"
        "        self,\n"
        "        n: int,\n"
        "        *,\n"
        "        scale: float = 1.0,\n"
        "    ) -> int:\n"
        "        return int(n * scale) + self.cfg.base\n"
    )


def test_move_symbol_leave_delegate_forwards_posonly_vararg_kwonly_kwargs(
    tmp_path: Path,
) -> None:
    """Every parameter kind is forwarded correctly in the delegate's return call."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(self, a, /, b, *args, c, d=3, **kw):\n"
        "        return a\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert (tmp_path / "src.py").read_text() == (
        "class Mixin:\n"
        "    def compute(self, a, /, b, *args, c, d=3, **kw):\n"
        "        return self.cfg.compute(a, b, *args, c=c, d=d, **kw)\n"
    )


def test_move_symbol_leave_delegate_survives_paren_in_string_default(
    tmp_path: Path,
) -> None:
    """A string default containing '(' must not break the delegate's header scan."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    def compute(\n"
        "        self,\n"
        '        sep: str = "(",\n'
        "        n: int = 0,\n"
        "    ) -> int:\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text(
        "class Cfg:\n    def keep(self):\n        return 0\n"
    )
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    src_out = (tmp_path / "src.py").read_text()
    compile(src_out, "src.py", "exec")
    assert "return self.cfg.compute(sep, n)" in src_out


def test_move_symbol_async_leave_delegate_awaits_the_forwarded_call(
    tmp_path: Path,
) -> None:
    """An async method's delegate stub must await the forwarded coroutine."""
    (tmp_path / "src.py").write_text(
        "class Mixin:\n"
        "    async def compute(self, n):\n"
        "        return n + self.cfg.base\n"
    )
    (tmp_path / "dst.py").write_text("class Cfg:\n    def e(self):\n        return 0\n")
    r = Repro("b", "t").move_symbol(
        "compute", src="src.py", dst="dst.py", into_class="Cfg", leave_delegate="cfg"
    )
    _apply(r, tmp_path)
    assert "return await self.cfg.compute(n)" in (tmp_path / "src.py").read_text()
