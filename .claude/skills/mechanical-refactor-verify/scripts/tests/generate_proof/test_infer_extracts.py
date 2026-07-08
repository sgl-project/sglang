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
