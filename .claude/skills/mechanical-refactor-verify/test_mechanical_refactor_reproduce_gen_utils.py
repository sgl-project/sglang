import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mechanical_refactor_reproduce_gen_utils import (
    infer_recipe,
    recipe_to_script,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, **files: str | None) -> None:
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _git(root, "config", "commit.gpgsign", "false")
    return root


def _method_onto_class(repo: Path) -> None:
    """Stage a base + a 'move foo from M onto C, lower the caller' commit."""
    _write(
        repo,
        **{
            "model.py": (
                "class M:\n"
                "    @staticmethod\n"
                "    def foo(self, x):\n"
                "        return x + 1\n"
                "\n"
                "    def other(self):\n"
                "        return 0\n"
            ),
            "comp.py": "class C:\n    def keep(self):\n        return 1\n",
            "caller.py": (
                "class K:\n"
                "    def run(self):\n"
                "        from model import M\n"
                "\n"
                "        return M.foo(self.c, 9)\n"
            ),
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
                "        return x + 1\n"
            ),
            "caller.py": (
                "class K:\n    def run(self):\n        return self.c.foo(9)\n"
            ),
        },
    )
    _commit(repo, "move foo onto C")


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
    """A method still inside the class cut straight into a new module is not a staged trailing
    block, so the extract is reported unsupported (prep should inline it at the tail first).
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
    assert any("staged trailing block" in note for note in recipe.notes)


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


def test_recipe_to_script_is_self_contained_and_ordered(repo: Path) -> None:
    """The emitted script imports only the reproduce util and lowers before moving."""
    _method_onto_class(repo)
    script = recipe_to_script(infer_recipe("HEAD", str(repo)), "move foo onto C")
    assert "from mechanical_refactor_reproduce_utils import Repro" in script
    assert script.index("lower_call_sites") < script.index("move_symbol")
    assert "r.run()" in script
    # importing nothing else from the skill keeps the script auditable in isolation
    assert "mechanical_refactor_verify_utils" not in script
    assert "mechanical_refactor_reproduce_gen_utils" not in script
