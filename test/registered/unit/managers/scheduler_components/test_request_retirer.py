import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).parent / "srt"
_STEPS = ("release_kv_cache", "release_aborted_request")
# mem_cache defines the steps; the retirer is where they belong.
_SKIPPED = ("mem_cache/", "managers/scheduler_components/request_retirer.py")

# The functions outside `RequestRetirer` that still run a release step
# themselves. It only shrinks: a new exit goes through `RequestRetirer`.
_STEP_CALLERS = {
    "disaggregation/decode.py:DecodePreallocQueue.pop_preallocated",
    "disaggregation/decode.py:DecodeTransferQueue._release_request",
    "disaggregation/decode.py:SchedulerDisaggregationDecodeMixin._handle_kv_checksum_mismatch",
    "disaggregation/prefill.py:PrefillBootstrapQueue._abort",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin._retire_aborted_prefill_result",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.process_disagg_prefill_transfer_queue",
    "disaggregation/prefill.py:SchedulerDisaggregationPrefillMixin.release_aborted_prefill_waiting_req",
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


if __name__ == "__main__":
    unittest.main()
