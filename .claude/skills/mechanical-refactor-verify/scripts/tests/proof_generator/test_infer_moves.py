import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import (  # noqa: F401
    _commit,
    _free_function_move_with_module_level_caller,
    _git,
    _method_onto_class,
    _write,
)
from mechanical_refactor_proof_generator import (
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


def test_infer_recipe_module_level_def_shadowed_by_method_name(repo: Path) -> None:
    """A column-0 cut resolves to the module-level def even when a method shares its name."""
    _write(
        repo,
        **{
            "model.py": (
                "def foo(*, x):\n"
                "    return x + 1\n"
                "\n"
                "\n"
                "class M:\n"
                "    def foo(self):\n"
                "        return foo(x=self.x)\n"
            ),
            "util.py": "def keep():\n    return 1\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": (
                "from util import foo\n"
                "\n"
                "\n"
                "class M:\n"
                "    def foo(self):\n"
                "        return foo(x=self.x)\n"
            ),
            "util.py": (
                "def keep():\n"
                "    return 1\n"
                "\n"
                "\n"
                "def foo(*, x):\n"
                "    return x + 1\n"
            ),
        },
    )
    commit = _commit(repo, "move module-level foo to util")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["foo"]
    assert recipe.moves[0]["from_class"] is None
    assert recipe.moves[0]["into_class"] is None


def test_infer_recipe_class_move_between_existing_files(repo: Path) -> None:
    """A top-level class relocated to an existing module moves whole; its methods do not."""
    _write(
        repo,
        **{
            "model.py": (
                "class Payload:\n"
                "    def get(self):\n"
                "        return 1\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 2\n"
            ),
            "comp.py": "def keep():\n    return 3\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def stay():\n    return 2\n",
            "comp.py": (
                "def keep():\n"
                "    return 3\n"
                "\n"
                "\n"
                "class Payload:\n"
                "    def get(self):\n"
                "        return 1\n"
            ),
        },
    )
    commit = _commit(repo, "move Payload to comp")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["Payload"]
    assert recipe.moves[0]["from_class"] is None
    assert recipe.moves[0]["into_class"] is None


def test_infer_recipe_move_leaving_a_forwarding_delegate(repo: Path) -> None:
    """A same-named stub re-added to the source infers leave_delegate on the move."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n" "    def work(self, x):\n" "        return x + 1\n"
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
                "    def work(self, x):\n"
                "        return self.comp.work(x)\n"
            ),
            "comp.py": (
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def work(self, x):\n"
                "        return x + 1\n"
            ),
        },
    )
    commit = _commit(repo, "move M.work onto C, leaving a delegate")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [mv["name"] for mv in recipe.moves] == ["work"]
    assert recipe.moves[0]["dst"] == "comp.py"
    assert recipe.moves[0]["leave_delegate"] == "comp"
    assert recipe.moves[0]["delegate_name"] is None
    script = recipe_to_script(recipe, "move with delegate")
    assert "leave_delegate='comp'" in script


def test_infer_recipe_constant_relocated_with_the_move(repo: Path) -> None:
    """A module constant that vanished from the source and appeared in the existing
    destination becomes a move_assign."""
    _write(
        repo,
        **{
            "model.py": (
                "RATIO = 3\n"
                "\n"
                "\n"
                "def work(x):\n"
                "    return x * RATIO\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 1\n"
            ),
            "comp.py": "import os\n\n\ndef keep():\n    return 2\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def stay():\n    return 1\n",
            "comp.py": (
                "import os\n"
                "\n"
                "RATIO = 3\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return 2\n"
                "\n"
                "\n"
                "def work(x):\n"
                "    return x * RATIO\n"
            ),
        },
    )
    commit = _commit(repo, "move work + RATIO to comp")

    recipe = infer_recipe(commit, str(repo))

    assert recipe.supported
    assert [am["name"] for am in recipe.assign_moves] == ["RATIO"]
    script = recipe_to_script(recipe, "move with constant")
    assert "move_assign" in script
