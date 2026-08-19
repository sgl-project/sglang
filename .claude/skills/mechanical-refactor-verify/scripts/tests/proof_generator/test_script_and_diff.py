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


def test_recipe_to_script_is_self_contained_and_ordered(repo: Path) -> None:
    """The emitted script imports only the reproduce util and lowers before moving."""
    _method_onto_class(repo)
    script = recipe_to_script(infer_recipe("HEAD", str(repo)), "move foo onto C")
    assert "from mechanical_refactor_reproduction_utils import Repro" in script
    assert script.index("lower_call_sites") < script.index("move_symbol")
    assert "residual = r.run()" in script
    assert "sys.exit(1 if residual else 0)" in script
    # importing nothing else from the skill keeps the script auditable in isolation
    assert "mechanical_refactor_verify_utils" not in script
    assert "mechanical_refactor_proof_generator" not in script


def test_recipe_to_script_orders_import_ops_after_moves(repo: Path) -> None:
    """The emitted script applies module-level import add/remove AFTER the move, matching
    build_repro's run order so the script and the in-process verdict cannot diverge."""
    _free_function_move_with_module_level_caller(repo)
    script = recipe_to_script(infer_recipe("HEAD", str(repo)), "move resolve to util")
    assert script.index("move_symbol") < script.index("remove_imported_name")
    assert script.index("move_symbol") < script.index("add_import")


def _emit_runnable_script(repo: Path, out: Path, commit: str, subject: str) -> Path:
    """Write the emitted script plus its util dependency into a proof-folder layout."""
    scripts_dir = out / "repro_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    utils_src = Path(__file__).resolve().parents[2] / (
        "mechanical_refactor_reproduction_utils.py"
    )
    (out / "mechanical_refactor_reproduction_utils.py").write_text(
        utils_src.read_text()
    )
    script = recipe_to_script(infer_recipe(commit, str(repo)), subject)
    script_path = scripts_dir / f"{commit[:9]}.py"
    script_path.write_text(script)
    return script_path


def test_emitted_script_exits_zero_on_faithful_commit(
    repo: Path, tmp_path: Path
) -> None:
    """Running the emitted script on a clean move exits 0 and prints the PASS verdict."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
            "caller.py": (
                "from model import resolve\n\n\ndef run(m):\n    return resolve(m)\n"
            ),
        },
    )
    _commit(repo, "base")
    # The after-state is the primitives' exact output (this bare repo has no formatter
    # to absorb the cut's leftover blank lines, unlike a pre-commit-clean real repo).
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": "import os\n\ndef resolve(m):\n    return m\n",
            "caller.py": (
                "from util import resolve\n\n\ndef run(m):\n    return resolve(m)\n"
            ),
        },
    )
    commit = _commit(repo, "move resolve to util")
    script_path = _emit_runnable_script(repo, tmp_path / "out", commit, "move")

    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_emitted_script_exits_nonzero_on_bundled_change(
    repo: Path, tmp_path: Path
) -> None:
    """A commit bundling a non-move change makes the emitted script exit non-zero."""
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\ndef resolve(m):\n    return m\n",
            "util.py": "import os\n",
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 99\n",
            "util.py": "import os\n\n\ndef resolve(m):\n    return m\n",
        },
    )
    commit = _commit(repo, "move resolve AND change keep")
    script_path = _emit_runnable_script(repo, tmp_path / "out", commit, "dirty move")

    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "RESIDUAL" in result.stdout


def test_emitted_script_passes_on_move_above_typechecking_guard(
    repo: Path, tmp_path: Path
) -> None:
    """A module-level def relocated to just above an ``if TYPE_CHECKING:`` guard reproduces
    via an inferred after= anchor and the emitted script exits 0."""
    _write(
        repo,
        **{
            "model.py": (
                "def keep():\n    return 0\n\n\ndef helper(x):\n    return x + 1\n"
            ),
            "util.py": (
                "from u import is_hip\n"
                "\n"
                "_is_hip = is_hip()\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from m import Thing\n"
            ),
        },
    )
    _commit(repo, "base")
    # After-state = the primitive's exact output (bare repo, no formatter to absorb blanks).
    _write(
        repo,
        **{
            "model.py": "def keep():\n    return 0\n\n\n",
            "util.py": (
                "from u import is_hip\n"
                "\n"
                "_is_hip = is_hip()\n"
                "\n"
                "def helper(x):\n"
                "    return x + 1\n"
                "\n"
                "if TYPE_CHECKING:\n"
                "    from m import Thing\n"
            ),
        },
    )
    commit = _commit(repo, "move helper above the TYPE_CHECKING guard")
    script_path = _emit_runnable_script(
        repo, tmp_path / "out", commit, "after-anchor move"
    )

    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=repo, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
    assert "after='_is_hip'" in script_path.read_text()


def test_per_file_diff_keeps_content_lines_starting_with_plus_signs(repo: Path) -> None:
    """An added content line beginning with '++' is collected, not mistaken for a header."""
    from mechanical_refactor_proof_generator import _per_file_diff

    _write(repo, **{"notes.py": "a = 1\n"})
    _commit(repo, "base")
    _write(repo, **{"notes.py": 'a = 1\nb = "++x"\n'})
    commit = _commit(repo, "add plus-plus line")

    files = _per_file_diff(commit, str(repo))

    assert files["notes.py"]["added"] == ['b = "++x"']
