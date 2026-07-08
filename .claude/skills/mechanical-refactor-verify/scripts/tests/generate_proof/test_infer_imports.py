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
