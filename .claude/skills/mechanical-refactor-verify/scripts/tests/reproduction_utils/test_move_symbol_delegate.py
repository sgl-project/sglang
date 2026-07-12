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
