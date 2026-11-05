"""Exercise upstream-only request-list and HiSparse compatibility changes."""

from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


def load(path, cls, methods, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node.bases = []
    node.body = [
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    for func in node.body:
        func.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls]


class IndexedQueue:
    def __init__(self):
        self.requests = []

    def __iter__(self):
        return iter(self.requests)

    def join(self, requests):
        self.requests.extend(requests)


class FinishedAbort:
    pass


class TestMainCompatibility(CustomTestCase):
    def test_chunked_abort_waits_in_indexed_queue_and_cleans_hisparse_after_drain(self):
        events = []
        prefill = ROOT / "python/sglang/srt/disaggregation/prefill.py"
        scheduler = ROOT / "python/sglang/srt/managers/scheduler.py"
        if "_kvserve_deferred_hisparse_cleanup" not in scheduler.read_text():
            self.skipTest("upstream-only compatibility path")
        namespace = dict(
            logger=Mock(),
            get_parallel=lambda: types.SimpleNamespace(tp_rank=0),
            FINISH_ABORT=FinishedAbort,
            FINISH_LENGTH=lambda **kwargs: object(),
            DisaggregationMode=types.SimpleNamespace(PREFILL="prefill"),
            prepare_abort=lambda *args, **kwargs: None,
            maybe_release_metadata_buffer=lambda *args: events.append("metadata"),
            release_kv_cache=lambda *args, **kwargs: events.append("pages"),
            make_abort_req=lambda *args, **kwargs: object(),
        )
        mixin = load(
            prefill,
            "SchedulerDisaggregationPrefillMixin",
            {
                "defer_aborted_kvserve_source_release",
                "_retire_aborted_prefill_result",
                "handle_inflight_transfer_failure",
            },
            namespace,
        )
        scheduler_class = load(
            scheduler, "Scheduler", {"process_pending_chunked_abort"}, namespace
        )
        owner = object.__new__(mixin)
        owner.process_pending_chunked_abort = types.MethodType(
            scheduler_class.process_pending_chunked_abort, owner
        )
        owner.disagg_prefill_transfer_queue = IndexedQueue()
        owner.prefill_launched_reqs = types.SimpleNamespace(leave=Mock())
        owner.clear_pending_chunk_send = Mock()
        owner.release_aborted_request = lambda req: events.append("request")
        owner.req_to_metadata_buffer_idx_allocator = object()
        owner.tree_cache = object()
        owner.enable_hisparse = True
        owner.hisparse_coordinator = types.SimpleNamespace(
            request_finished=lambda req: events.append("hisparse")
        )
        owner.beam_coordinator = types.SimpleNamespace(retire_group=Mock())
        owner.metrics_reporter = types.SimpleNamespace(enable_metrics=False)
        owner.disaggregation_mode = "prefill"
        owner.send_abort_output = Mock()
        sender = types.SimpleNamespace(
            kv_mgr=types.SimpleNamespace(kvserve_enabled=True),
            abort=Mock(),
            is_source_release_safe=lambda: False,
            failure_exception=Mock(side_effect=RuntimeError("failed")),
        )
        reason = types.SimpleNamespace(
            message="abort", status_code=400, to_json=lambda: {}
        )
        req = types.SimpleNamespace(
            rid="r",
            bootstrap_room=1,
            disagg_kv_sender=sender,
            discard_output_reason=reason,
            finished_reason=FinishedAbort(),
            time_stats=types.SimpleNamespace(
                trace_ctx=types.SimpleNamespace(abort=Mock())
            ),
            clear_requested_finish=Mock(),
            kv=types.SimpleNamespace(holds_kv=True, holds_mamba=False),
            metadata_buffer_index=1,
            pending_bootstrap=True,
            to_finish=None,
        )
        owner.chunked_req = owner._pending_chunked_abort_req = req

        owner.process_pending_chunked_abort()
        self.assertEqual(owner.disagg_prefill_transfer_queue.requests, [req])
        self.assertTrue(req._kvserve_deferred_hisparse_cleanup)
        self.assertEqual(events, [])

        # A late overlapping forward result must not become a second releaser.
        self.assertFalse(owner._retire_aborted_prefill_result(req))
        self.assertEqual(owner.disagg_prefill_transfer_queue.requests, [req])
        owner.prefill_launched_reqs.leave.assert_called_once_with([req])

        # Once the source gate drains, the normal failed-transfer handler owns
        # cleanup; the upstream HiSparse mapping must be dropped before pages.
        owner.handle_inflight_transfer_failure(req)
        self.assertEqual(events, ["hisparse", "pages", "request"])
        self.assertFalse(req._kvserve_deferred_hisparse_cleanup)


if __name__ == "__main__":
    unittest.main()
