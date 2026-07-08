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
    assert "r.run()" in script
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


def test_per_file_diff_keeps_content_lines_starting_with_plus_signs(repo: Path) -> None:
    """An added content line beginning with '++' is collected, not mistaken for a header."""
    from mechanical_refactor_proof_generator import _per_file_diff

    _write(repo, **{"notes.py": "a = 1\n"})
    _commit(repo, "base")
    _write(repo, **{"notes.py": 'a = 1\nb = "++x"\n'})
    commit = _commit(repo, "add plus-plus line")

    files = _per_file_diff(commit, str(repo))

    assert files["notes.py"]["added"] == ['b = "++x"']
