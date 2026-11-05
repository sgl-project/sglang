import ast
import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang
from sglang.srt.managers.scheduler_components import request_retirer
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).parent / "srt"
_STEPS = ("release_kv_cache", "release_aborted_request")
# mem_cache defines the steps; the retirer is where they belong.
_SKIPPED = ("mem_cache/", "managers/scheduler_components/request_retirer.py")

# The functions outside `RequestRetirer` that still run a release step
# themselves. It only shrinks: a new exit goes through `RequestRetirer`.
_STEP_CALLERS = {
    "disaggregation/prefill.py:PrefillBootstrapQueue._abort",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue",
    "managers/schedule_batch.py:release_req",
    "managers/scheduler.py:Scheduler.process_pending_chunked_abort",
    "managers/scheduler.py:Scheduler.release_aborted_request",
    "managers/scheduler.py:Scheduler.retire_unadmitted_request",
    "managers/scheduler_components/dynamic_chunk_sizer.py:DynamicChunkSizer._profile_prefill_latency",
}


def _is_step(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in _STEPS
    return isinstance(node.func, ast.Attribute) and node.func.attr in _STEPS


def _step_callers() -> set:
    callers = set()
    for path in _SRT.rglob("*.py"):
        where = path.relative_to(_SRT).as_posix()
        if where.startswith(_SKIPPED):
            continue
        body = ast.parse(path.read_text()).body
        methods = [
            (f"{c.name}.", f)
            for c in body
            if isinstance(c, ast.ClassDef)
            for f in c.body
        ]
        for prefix, fn in methods + [("", f) for f in body]:
            if isinstance(fn, ast.FunctionDef) and any(map(_is_step, ast.walk(fn))):
                callers.add(f"{where}:{prefix}{fn.name}")
    return callers


class TestReleaseStepsLiveInRequestRetirer(CustomTestCase):
    def test_only_allowlisted_functions_run_a_release_step(self):
        self.assertEqual(_step_callers(), _STEP_CALLERS)


class TestReleaseAborted(CustomTestCase):
    def test_drops_cache_side_state_before_an_uncached_release(self):
        # A PD transfer or bootstrap failure, or a checksum mismatch, must also
        # cancel the attempt's cache work and free its streaming session.
        calls = []
        tree_cache = MagicMock()
        tree_cache.finish.side_effect = lambda handle, outcome: calls.append(outcome)
        req = SimpleNamespace(
            cache_request_handle=object(),
            session=SimpleNamespace(
                streaming=True, abort_req=lambda: calls.append("session")
            ),
            release_mm_inputs_on_abort=lambda: calls.append("mm"),
            kv=SimpleNamespace(
                holds_kv=True, holds_mamba=False, retraction_backup=None
            ),
        )

        with patch.object(
            request_retirer,
            "release_kv_cache",
            side_effect=lambda req, cache, *, checkpoint: calls.append(checkpoint),
        ):
            release_only_retirer(tree_cache).release_aborted(req)

        self.assertEqual(calls, ["mm", CacheRequestOutcome.ABORT, "session", False])


if __name__ == "__main__":
    unittest.main()
