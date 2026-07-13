import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import _commit, _git, _write  # noqa: F401
from mechanical_refactor_proof_generator import (
    build_repro,
    infer_recipe,
    recipe_to_script,
)


def test_infer_extract_function_with_returned_local(repo: Path) -> None:
    """A block ending in ``pool = make(...)`` carved into a helper that returns ``pool`` infers
    an extract_function whose body is the verbatim block and whose return_text is authored.
    """
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            x = self.a\n"
                "            y = x + n\n"
                "            pool = make(\n"
                "                a=x,\n"
                "                b=y,\n"
                "            )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(n=n)\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        x = self.a\n"
                "        y = x + n\n"
                "        pool = make(\n"
                "            a=x,\n"
                "            b=y,\n"
                "        )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "extract _build_pool from dispatch")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.moves == []
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["name"] == "_build_pool"
    assert ex["src"] == "kv.py" and ex["dst"] == "kv.py"
    assert ex["into_class"] == "C"
    assert ex["before"] == "keep"
    assert ex["body_indent"] == 12
    assert ex["body"] == (
        "            x = self.a\n"
        "            y = x + n\n"
        "            pool = make(\n"
        "                a=x,\n"
        "                b=y,\n"
        "            )\n"
    )
    assert ex["call"] == "            pool = self._build_pool(n=n)\n"
    assert ex["return_text"] == "        return pool"
    assert ex["signature"] == "    def _build_pool(self, *, n):\n"


def test_infer_extract_function_no_return_text_when_body_is_whole_helper(
    repo: Path,
) -> None:
    """When the helper body reproduces the source block with no trailing return, return_text
    is None (the block is a side-effecting statement sequence, not a value producer)."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self.log(1)\n"
                "        self.log(2)\n"
                "        self.post()\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self._emit()\n"
                "        self.post()\n"
                "\n"
                "    def _emit(self):\n"
                "        self.log(1)\n"
                "        self.log(2)\n"
            )
        },
    )
    _commit(repo, "extract _emit from run")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["name"] == "_emit"
    assert ex["return_text"] is None
    assert ex["call"] == "        self._emit()\n"


def test_infer_extract_function_rejects_edited_body(repo: Path) -> None:
    """A helper whose body was edited (not a verbatim cut) does not infer an extract_function,
    so the residual surfaces the bundled change instead of a false pass."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self.log(1)\n"
                "        self.post()\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self._emit()\n"
                "        self.post()\n"
                "\n"
                "    def _emit(self):\n"
                "        self.log(2)\n"
            )
        },
    )
    _commit(repo, "extract _emit but change the arg")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.extract_functions == []
    assert recipe.supported is False


def test_emitted_script_passes_on_extract_function(repo: Path, tmp_path: Path) -> None:
    """The recipe for an extract_function reproduces the commit byte-for-byte (bare repo, no
    formatter) so build_repro returns an empty residual."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            x = self.a\n"
                "            pool = make(\n"
                "                a=x,\n"
                "            )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(n=n)\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        x = self.a\n"
                "        pool = make(\n"
                "            a=x,\n"
                "        )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    commit = _commit(repo, "extract _build_pool from dispatch")
    recipe = infer_recipe(commit, str(repo))
    residual = build_repro(recipe, repo_root=str(repo)).run()
    assert residual == "", residual
    assert "extract_function" in recipe_to_script(recipe, "extract")
