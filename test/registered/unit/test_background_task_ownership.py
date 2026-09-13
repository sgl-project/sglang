"""Ratchet against unowned fire-and-forget asyncio tasks.

``asyncio`` only keeps weak references to scheduled tasks.  A bare
``create_task(...)`` expression can therefore disappear before the coroutine
finishes.  These modules previously contained the production failures tracked
by #34434; keep every future task attached to an explicit lifecycle owner.
"""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWNERSHIP_RATCHET_PATHS = (
    "python/sglang/multimodal_gen/runtime/entrypoints/openai/mesh_api.py",
    "python/sglang/srt/managers/multi_tokenizer_mixin.py",
    "python/sglang/srt/managers/tokenizer_manager.py",
)


def _bare_task_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr in {"create_task", "ensure_future"}
    ]


class TestBackgroundTaskOwnership(CustomTestCase):
    def test_fire_and_forget_tasks_have_lifecycle_owners(self):
        offenders = {
            relative_path: lines
            for relative_path in _OWNERSHIP_RATCHET_PATHS
            if (lines := _bare_task_calls(_REPO_ROOT / relative_path))
        }

        self.assertFalse(
            offenders,
            "Bare task creation leaves only asyncio's weak reference; retain each "
            f"task until completion: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
