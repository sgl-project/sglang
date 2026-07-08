import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generate_testlib import (  # noqa: F401
    _commit,
    _free_function_move_with_module_level_caller,
    _git,
    _method_onto_class,
    _write,
)
from mechanical_refactor_generate_proof import (
    infer_recipe,
    recipe_to_script,
)


def test_infer_recipe_method_onto_class(repo: Path) -> None:
    """A method move onto a class infers the move, the call-site lowering, and the orphaned
    local import removal."""
    _method_onto_class(repo)
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert [
        (m["name"], m["src"], m["dst"], m["into_class"], m["dedent"])
        for m in recipe.moves
    ] == [("foo", "model.py", "comp.py", "C", 0)]
    assert recipe.lowerings == [
        {"name": "foo", "owner": "M", "path": "caller.py", "kind": "lower"}
    ]
    assert recipe.import_removals == [
        {"path": "caller.py", "text": "from model import M", "in_function": "run"}
    ]
    assert recipe.import_additions == []


def test_infer_recipe_free_function_move_uses_requalify(repo: Path) -> None:
    """A move to a module-level free function dedents and requalifies the call site
    (drops the qualifier), rather than lowering a receiver."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(x):\n"
                "        return x + 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "util.py": "import os\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from model import M\n"
                "\n"
                "        return M.foo(9)\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "util.py": "import os\n\n\ndef foo(x):\n    return x + 1\n",
            "caller.py": ("class K:\n    def run(self):\n        return foo(9)\n"),
        },
    )
    _commit(repo, "move foo to util as a free function")
    recipe = infer_recipe("HEAD", str(repo))
    assert [(m["name"], m["into_class"], m["dedent"]) for m in recipe.moves] == [
        ("foo", None, 4)
    ]
    assert recipe.lowerings == [
        {"name": "foo", "owner": "M", "path": "caller.py", "kind": "requalify"}
    ]


def test_infer_recipe_excludes_the_moved_bodys_own_call(repo: Path) -> None:
    """A same-named call on a different receiver inside the moved body is not a caller
    lowering (only `M.foo(...)` is, not `worker.foo(...)`)."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self, x):\n"
                "        worker.foo(x)\n"
                "        return x\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self, x):\n"
                "        worker.foo(x)\n"
                "        return x\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.lowerings == []


def test_infer_recipe_skips_nested_functions(repo: Path) -> None:
    """A def nested inside a moved method is not inferred as its own move."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    def wrap(self):\n"
                "        def inner(z):\n"
                "            return z\n"
                "        return inner\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def wrap(self):\n"
                "        def inner(z):\n"
                "            return z\n"
                "        return inner\n"
            ),
        },
    )
    _commit(repo, "move wrap onto C")
    recipe = infer_recipe("HEAD", str(repo))
    names = [m["name"] for m in recipe.moves]
    assert names == ["wrap"]
    assert any("inner" in n for n in recipe.notes)


def test_infer_recipe_free_function_source_move_repaths_caller(repo: Path) -> None:
    """A free function moved to an existing module becomes a move_symbol with the call left
    bare; a caller's function-scoped import is repathed."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from model import resolve\n"
                "\n"
                "        return resolve(self.m)\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "import os\n\n\ndef resolve(m):\n    return m\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from util import resolve\n"
                "\n"
                "        return resolve(self.m)\n"
            ),
        },
    )
    _commit(repo, "move resolve to util")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert [(m["name"], m["src"], m["dst"], m["into_class"]) for m in recipe.moves] == [
        ("resolve", "model.py", "util.py", None)
    ]
    assert recipe.lowerings == []
    assert recipe.repaths == [
        {
            "path": "caller.py",
            "old_module": "model",
            "new_module": "util",
            "name": "resolve",
        }
    ]


def test_infer_recipe_survives_a_non_python_file_in_the_commit(repo: Path) -> None:
    """A commit also touching a .md file infers the move and notes the non-Python path."""
    _write(
        repo,
        **{
            "model.py": "def foo():\n    return 1\n\n\ndef keep():\n    return 0\n",
            "util.py": "x = 1\n",
            "README.md": "hello\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "x = 1\n\n\ndef foo():\n    return 1\n",
            "README.md": "hello world, this is plain markdown text\n",
        },
    )
    commit = _commit(repo, "move foo and touch docs")

    recipe = infer_recipe(commit, str(repo))

    assert [mv["name"] for mv in recipe.moves] == ["foo"]
    assert any("README.md" in note for note in recipe.notes)


def test_infer_recipe_records_the_source_class_for_disambiguation(repo: Path) -> None:
    """A method move carries from_class so the cut cannot hit a same-named other method."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
                "\n"
                "\n"
                "class Other:\n"
                "    def foo(self, x):\n"
                "        return x + 2\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    pass\n"
                "\n"
                "\n"
                "class Other:\n"
                "    def foo(self, x):\n"
                "        return x + 2\n"
            ),
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
            ),
        },
    )
    commit = _commit(repo, "move M.foo onto C")

    recipe = infer_recipe(commit, str(repo))

    assert [mv["from_class"] for mv in recipe.moves] == ["M"]
    script = recipe_to_script(recipe, "move M.foo onto C")
    assert "from_class='M'" in script
