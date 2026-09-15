"""`entrypoints/worker_launch.py` is a light-import copy of the launch
preparation and scheduler spawn loop in `entrypoints/engine.py`, used by
pre-spawn (SGLANG_PRESPAWN_WORKERS=1). These tests fail when the copy drifts
from the engine, and when the copy grows an import of the heavy modules it
exists to avoid.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import ast
import inspect
import subprocess
import sys
import textwrap
import unittest

from sglang.srt.entrypoints import engine, worker_launch
from sglang.srt.entrypoints.worker_launch import _calculate_rank_ranges
from sglang.test.test_utils import CustomTestCase

# Module-level helpers copied verbatim (same name in both modules).
MIRRORED_HELPERS = (
    "SchedulerInitResult",
    "_set_envs_and_config",
    "_log_legacy_kernel_cache_dirs",
    "_scheduler_died_error",
    "_wait_for_scheduler_ready",
    "_calculate_rank_ranges",
    "_compute_parallelism_ranks",
)

# Modules the light launch path must never import (their import is what
# pre-spawn overlaps with the workers' own init).
HEAVY_MODULES = (
    "sglang.srt.entrypoints.engine",
    "sglang.srt.entrypoints.http_server",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.data_parallel_controller",
    "sglang.srt.managers.detokenizer_manager",
)


def _from_marker(src: str, marker: str) -> str:
    """The source from the first line containing `marker` on, dedented."""
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    return textwrap.dedent("\n".join(lines[start:]))


def _statement_lines(func) -> list:
    """Stripped physical lines of a function's body statements (docstring
    excluded, comments and blank lines dropped)."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    body = tree.body[0].body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    out = []
    for stmt in body:
        for line in ast.get_source_segment(src, stmt).splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


class TestMirrorsEngine(CustomTestCase):
    def test_helpers_identical(self):
        for name in MIRRORED_HELPERS:
            self.assertEqual(
                inspect.getsource(getattr(engine, name)),
                inspect.getsource(getattr(worker_launch, name)),
                f"{name}: worker_launch.py no longer matches engine.py; "
                "update the copy",
            )

    def test_spawn_loop_identical(self):
        marker = "scheduler_procs = []"
        self.assertEqual(
            _from_marker(
                inspect.getsource(engine.Engine._launch_scheduler_processes), marker
            ),
            _from_marker(
                inspect.getsource(worker_launch.launch_scheduler_processes), marker
            ),
            "launch_scheduler_processes no longer matches "
            "Engine._launch_scheduler_processes; update the copy",
        )

    def test_preparation_statements_are_in_the_engine(self):
        """Every statement pre-spawn runs before spawning is (still) a statement
        of `Engine._launch_subprocesses`, so a renamed or removed step in the
        engine shows up here. (A step added to the engine is not detected.)"""
        launch = {
            line.strip()
            for line in inspect.getsource(
                engine.Engine._launch_subprocesses
            ).splitlines()
        }
        for func in (worker_launch.prepare_launch, worker_launch.allocate_port_args):
            for stmt_line in _statement_lines(func):
                if stmt_line.startswith("return"):
                    continue
                self.assertIn(
                    stmt_line,
                    launch,
                    f"{func.__name__}: {stmt_line!r} is not in "
                    "Engine._launch_subprocesses any more; update worker_launch.py",
                )


class TestStaysLight(CustomTestCase):
    def test_light_modules_do_not_import_the_server_stack(self):
        code = (
            "import sys\n"
            "import sglang.srt.entrypoints.worker_launch\n"
            "import sglang.srt.entrypoints.prespawn\n"
            "import sglang.srt.managers.process_entry\n"
            "print(','.join(m for m in %r if m in sys.modules))\n" % (HEAVY_MODULES,)
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(out, "", f"heavy modules imported: {out}")


class TestRankRanges(CustomTestCase):
    def test_single_node(self):
        pp, tp, pp_per_node, tp_per_node = _calculate_rank_ranges(
            nnodes=1, pp_size=1, tp_size=4, node_rank=0
        )
        self.assertEqual((list(pp), list(tp)), ([0], [0, 1, 2, 3]))
        self.assertEqual((pp_per_node, tp_per_node), (1, 4))

    def test_tp_group_spanning_two_nodes(self):
        _, tp, _, tp_per_node = _calculate_rank_ranges(
            nnodes=2, pp_size=1, tp_size=8, node_rank=1
        )
        self.assertEqual(list(tp), [4, 5, 6, 7])
        self.assertEqual(tp_per_node, 4)

    def test_one_pp_stage_per_node(self):
        pp, tp, pp_per_node, tp_per_node = _calculate_rank_ranges(
            nnodes=2, pp_size=2, tp_size=4, node_rank=1
        )
        self.assertEqual((list(pp), list(tp)), ([1], [0, 1, 2, 3]))
        self.assertEqual((pp_per_node, tp_per_node), (1, 4))


if __name__ == "__main__":
    unittest.main()
