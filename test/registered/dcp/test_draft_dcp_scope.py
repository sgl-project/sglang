"""Draft init in the scheduler runs under draft_dcp_context."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang.srt
from sglang.srt import runtime_context as rc
from sglang.srt.distributed.parallel_state import patch_decode_context_parallel_group
from sglang.srt.speculative.spec_utils import draft_dcp_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SCHEDULER = Path(next(iter(sglang.srt.__path__))) / "managers" / "scheduler.py"

_SCOPED_METHODS = (
    "maybe_init_draft_worker",
    "init_memory_pools",
    "init_all_attention_backends",
    "init_all_cuda_graphs",
)


def _scheduler_methods():
    tree = ast.parse(_SCHEDULER.read_text(), filename=str(_SCHEDULER))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    return {
        n.name: n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _scoped_line_ranges(fn):
    ranges = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "draft_dcp_context"
            ):
                ranges.append((node.body[0].lineno, node.end_lineno))
    return ranges


def _draft_init_uses(fn):
    uses = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "draft_worker"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        ):
            uses.append((node.lineno, f"self.draft_worker.{node.attr}"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DraftWorkerClass"
        ):
            uses.append((node.lineno, "DraftWorkerClass(...)"))
    return uses


class TestDraftDcpScope(CustomTestCase):
    def test_scheduler_draft_init_runs_under_the_scope(self):
        methods = _scheduler_methods()
        unscoped = []
        for name in _SCOPED_METHODS:
            self.assertIn(name, methods, f"Scheduler.{name} was renamed or removed")
            fn = methods[name]
            ranges = _scoped_line_ranges(fn)
            uses = _draft_init_uses(fn)
            self.assertTrue(uses, f"{name}: found no draft init to check")
            unscoped += [
                f"{name}:{lineno} {what}"
                for lineno, what in uses
                if not any(lo <= lineno <= hi for lo, hi in ranges)
            ]
        self.assertEqual(
            unscoped,
            [],
            "draft init outside draft_dcp_context:\n  " + "\n  ".join(unscoped),
        )

    def test_the_scope_removes_the_dcp_group(self):
        """The draft must reach the state a process without DCP is in, so that
        every dcp_enabled guard short-circuits and attn_dcp_* derives to 1/0.
        Neutralizing attn_dcp_* alone would leave the group in place: the guards
        would still fire and the draft would take DCP paths it cannot run."""
        target_group = SimpleNamespace(world_size=4, rank_in_group=3)
        with patch_decode_context_parallel_group(target_group):
            self.assertTrue(rc.get_parallel().dcp_enabled)
            self.assertEqual(rc.get_parallel().attn_dcp_size, 4)
            self.assertEqual(rc.get_parallel().attn_dcp_rank, 3)

            with draft_dcp_context():
                parallel = rc.get_parallel()
                self.assertFalse(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_rank, 0)

            self.assertTrue(rc.get_parallel().dcp_enabled)
            self.assertEqual(rc.get_parallel().attn_dcp_size, 4)


if __name__ == "__main__":
    unittest.main()
