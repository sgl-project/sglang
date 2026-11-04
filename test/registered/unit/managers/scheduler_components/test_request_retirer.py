import ast
import pathlib
import unittest
import warnings

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).parent / "srt"
_RETIRER = "sglang.srt.managers.scheduler_components.request_retirer"
# mem_cache defines `release_kv_cache` and a cache-level `release_aborted_request`
# of its own; only the callers above the cache are scanned.
_SKIPPED_DIRS = ("mem_cache",)

# The functions outside `RequestRetirer` that still run a cleanup step
# themselves. It only shrinks: a new exit goes through `RequestRetirer`.
_STEP_CALLERS = {
    # The PD decode prealloc and transfer queues.
    "sglang.srt.disaggregation.decode.DecodePreallocQueue.pop_preallocated",
    "sglang.srt.disaggregation.decode.DecodeTransferQueue._release_request",
    "sglang.srt.disaggregation.decode.SchedulerDisaggregationDecodeMixin._handle_kv_checksum_mismatch",
    "sglang.srt.disaggregation.decode_hicache_mixin.DecodeHiCacheTransferMixin._clean_hicache_prefetch_resources",
    # The PD prefill transfer outcomes and the waiting-queue abort's PD part.
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin._retire_aborted_prefill_result",
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure",
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue",
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.process_disagg_prefill_transfer_queue",
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.release_aborted_prefill_waiting_req",
    # Finishes in output processing.
    "sglang.srt.dllm.mixin.scheduler.SchedulerDllmMixin._finish_dllm_request_if_needed",
    "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_finish_state_updated_req",
    "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor._handle_sampling_mask_abort",
    "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prebuilt",
    "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prefill",
    # Retraction, the chunked abort, and the delegate their callers still use.
    "sglang.srt.managers.schedule_batch.release_req",
    "sglang.srt.managers.scheduler.Scheduler.process_pending_chunked_abort",
    "sglang.srt.managers.scheduler.Scheduler.release_aborted_request",
    # Startup profiling of a dummy request, not a request exit.
    "sglang.srt.managers.scheduler_components.dynamic_chunk_sizer.DynamicChunkSizer._profile_prefill_latency",
}


def _is_cleanup_step(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "release_kv_cache"
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr in ("release_kv_cache", "release_aborted_request"):
        return True
    owner = func.value
    return func.attr == "finish" and (
        (isinstance(owner, ast.Attribute) and owner.attr == "tree_cache")
        or (isinstance(owner, ast.Name) and owner.id == "tree_cache")
    )


def _step_callers() -> set:
    callers = set()

    def visit(node, module, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, module, scope + [child.name])
                continue
            if isinstance(child, ast.Call) and _is_cleanup_step(child):
                callers.add(".".join([module, *scope]))
            visit(child, module, scope)

    for path in sorted(_SRT.rglob("*.py")):
        relative = path.relative_to(_SRT)
        if relative.parts[0] in _SKIPPED_DIRS:
            continue
        module = ".".join(["sglang", "srt", *relative.with_suffix("").parts])
        if module == _RETIRER:
            continue
        with warnings.catch_warnings():
            # Some modules hold invalid escapes in non-raw strings.
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text())
        visit(tree, module, [])
    return callers


class TestCleanupStepsLiveInRequestRetirer(CustomTestCase):
    def test_only_allowlisted_functions_run_a_cleanup_step(self):
        self.assertEqual(_step_callers(), _STEP_CALLERS)


if __name__ == "__main__":
    unittest.main()
