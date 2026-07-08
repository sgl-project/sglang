import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mechanical_refactor_generate_proof import (
    infer_recipe,
    recipe_to_script,
)

from generate_testlib import (  # noqa: F401
    _commit,
    _free_function_move_with_module_level_caller,
    _git,
    _method_onto_class,
    _write,
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


def test_infer_recipe_infers_added_module_imports(repo: Path) -> None:
    """An import the destination module gains (the moved code needs it) is inferred."""
    _write(
        repo,
        **{
            "model.py": (
                "import gc\n"
                "\n"
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
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
                "import gc\n"
                "\n"
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert {"path": "comp.py", "text": "import gc"} in recipe.import_additions


def test_infer_recipe_new_file_extract_from_class_method_unsupported(
    repo: Path,
) -> None:
    """A method still inside the class cut straight into a new module cannot be cut as a
    top-level symbol, so the extract is reported unsupported (prep must lift it out first).
    """
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        return 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def other(self):\n        return 0\n",
            "newmod.py": "def foo():\n    return 1\n",
        },
    )
    _commit(repo, "extract foo to a new module")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported is False
    assert any("not all top-level" in note for note in recipe.notes)


def test_infer_recipe_new_file_extract_from_staged_tail(repo: Path) -> None:
    """A staged trailing block (scaffolding + def at the source tail) cut into a new file
    infers an extract_to_new_module, prepending the future import."""
    _write(
        repo,
        **{
            "model.py": (
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
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "class M:\n    def keep(self):\n        return 1\n",
            "newmod.py": (
                "from __future__ import annotations\n"
                "\n"
                "import logging\n"
                "\n"
                "logger = logging.getLogger(__name__)\n"
                "\n"
                "\n"
                "def foo(x):\n"
                "    return x + 1\n"
            ),
        },
    )
    _commit(repo, "extract foo to a new module")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.moves == []
    assert recipe.extracts == [
        {
            "src": "model.py",
            "dst": "newmod.py",
            "symbols": ["foo"],
            "future_import": True,
        }
    ]


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


def test_infer_recipe_module_level_import_repoint_realised_by_diff(repo: Path) -> None:
    """A module-level consumer whose import is repointed old -> new yields a remove of the old
    name and an add of the new -- not a reliance on the formatter pruning a duplicate.
    """
    _free_function_move_with_module_level_caller(repo)
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.repaths == []
    assert {
        "path": "caller.py",
        "module": "model",
        "name": "resolve",
        "asname": None,
    } in recipe.module_import_removals
    assert {"path": "caller.py", "text": "from util import resolve"} in (
        recipe.import_additions
    )


def test_infer_recipe_removes_an_import_the_source_no_longer_uses(repo: Path) -> None:
    """When the moved body took the source's only use of an import, the source's lost name is
    realised as a removal (deterministic, not left to the formatter)."""
    _write(
        repo,
        **{
            "model.py": (
                "import gc\n"
                "\n"
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
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
                "import gc\n"
                "\n"
                "class C:\n"
                "    def keep(self):\n"
                "        return 1\n"
                "\n"
                "    def foo(self):\n"
                "        gc.collect()\n"
                "        return 1\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")
    recipe = infer_recipe("HEAD", str(repo))
    assert {
        "path": "model.py",
        "module": None,
        "name": "gc",
        "asname": None,
    } in recipe.module_import_removals


def test_infer_recipe_scattered_new_module_extract(repo: Path) -> None:
    """Scattered top-level defs cut into a new module (no staged trailing block) infer a scatter
    extract with the authored header and target order, not UNSUPPORTED."""
    _write(
        repo,
        **{
            "common.py": (
                "import os\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return 0\n"
                "\n"
                "\n"
                "def beta():\n"
                "    return 2\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 9\n"
                "\n"
                "\n"
                "def alpha():\n"
                "    return 1\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "common.py": (
                "import os\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return 0\n"
                "\n"
                "\n"
                "def stay():\n"
                "    return 9\n"
            ),
            "alloc.py": (
                "from __future__ import annotations\n"
                "\n"
                "import logging\n"
                "\n"
                "logger = logging.getLogger(__name__)\n"
                "\n"
                "\n"
                "def alpha():\n"
                "    return 1\n"
                "\n"
                "\n"
                "def beta():\n"
                "    return 2\n"
            ),
        },
    )
    _commit(repo, "extract alpha, beta to alloc.py")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.extracts == []
    assert recipe.moves == []
    assert len(recipe.scatter_extracts) == 1
    sx = recipe.scatter_extracts[0]
    assert sx["src"] == "common.py" and sx["dst"] == "alloc.py"
    assert sorted(sx["symbols"]) == ["alpha", "beta"]
    assert sx["order"] == ["alpha", "beta"]
    assert sx["header"].startswith("from __future__ import annotations\n")
    assert "logger = logging.getLogger(__name__)" in sx["header"]
    assert sx["drop_assigns"] == []
    script = recipe_to_script(recipe, "extract alpha, beta to alloc.py")
    assert "extract_symbols_to_new_module" in script


def test_infer_recipe_scatter_extract_drops_relocated_constant(repo: Path) -> None:
    """A module-level constant relocated into the new module is inferred as a drop_assign so the
    scatter extract removes it from the source too; a constant the source keeps is not.
    """
    _write(
        repo,
        **{
            "common.py": (
                "from u import is_hip\n"
                "\n"
                "_IS_HIP = is_hip()\n"
                "logger = 1\n"
                "\n"
                "\n"
                "def moved():\n"
                "    return _IS_HIP\n"
                "\n"
                "\n"
                "def keep():\n"
                "    return logger\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "common.py": ("logger = 1\n\n\ndef keep():\n    return logger\n"),
            "alloc.py": (
                "from __future__ import annotations\n"
                "\n"
                "from u import is_hip\n"
                "\n"
                "_IS_HIP = is_hip()\n"
                "\n"
                "\n"
                "def moved():\n"
                "    return _IS_HIP\n"
            ),
        },
    )
    _commit(repo, "extract moved to alloc.py")
    recipe = infer_recipe("HEAD", str(repo))
    assert len(recipe.scatter_extracts) == 1
    assert recipe.scatter_extracts[0]["drop_assigns"] == ["_IS_HIP"]


def test_infer_recipe_adds_wholly_new_module_import_verbatim(repo: Path) -> None:
    """An import gained from a module not present in base is captured as the target's verbatim
    statement (so an exploded/magic-comma wrapping is reproduced, not collapsed per-name).
    """
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef solve(x):\n    return x\n",
            "util.py": "import os\n",
            "caller.py": (
                "from model import solve\n\n\ndef run():\n    return solve(1)\n"
            ),
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n",
            "util.py": "import os\n\n\ndef solve(x):\n    return x\n",
            "caller.py": (
                "from util import (\n    solve,\n)\n\n\ndef run():\n    return solve(1)\n"
            ),
        },
    )
    _commit(repo, "move solve to util")
    recipe = infer_recipe("HEAD", str(repo))
    caller_adds = [
        a["text"] for a in recipe.import_additions if a["path"] == "caller.py"
    ]
    assert "from util import (\n    solve,\n)" in caller_adds
    assert {
        "path": "caller.py",
        "module": "model",
        "name": "solve",
        "asname": None,
    } in recipe.module_import_removals


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
