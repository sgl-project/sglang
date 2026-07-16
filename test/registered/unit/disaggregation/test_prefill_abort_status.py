import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fastapi

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    is_client_closed_abort,
    prepare_client_closed_abort,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillAbortStatus(unittest.TestCase):
    @staticmethod
    def _make_req():
        req = SimpleNamespace(
            return_logprob=False,
            finished_reason=None,
            rid="cancelled-rid",
            bootstrap_room=7,
            req_pool_idx=None,
            kv=None,
            mamba_pool_idx=None,
            metadata_buffer_index=-1,
            disagg_kv_sender=MagicMock(),
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
        )
        req.disagg_kv_sender.failure_exception.side_effect = RuntimeError(
            "Aborted by AbortReq."
        )
        prepare_client_closed_abort(req)
        return req

    @staticmethod
    def _make_scheduler():
        return SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0),
            tree_cache=MagicMock(),
            output_streamer=MagicMock(),
            metrics_reporter=SimpleNamespace(enable_metrics=True),
            metrics_collector=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            enable_hicache_storage=False,
        )

    def test_explicit_client_abort_has_terminal_http_status(self):
        class Req:
            return_logprob = False
            finished_reason = None

        req = Req()
        prepare_client_closed_abort(req)

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 499)
        self.assertEqual(req.finished_reason.err_type, "RequestCancelled")
        self.assertEqual(req.finished_reason.message, "Aborted by AbortReq.")
        self.assertTrue(is_client_closed_abort(req))

    def test_bootstrap_failure_preserves_client_abort(self):
        req = self._make_req()
        scheduler = self._make_scheduler()

        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertTrue(is_client_closed_abort(req))
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        scheduler.metrics_collector.increment_bootstrap_failed_reqs.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    def test_transfer_failure_preserves_client_abort(self, release_kv_cache):
        req = self._make_req()
        scheduler = self._make_scheduler()

        SchedulerDisaggregationPrefillMixin.handle_inflight_transfer_failure(
            scheduler, req
        )

        self.assertTrue(is_client_closed_abort(req))
        release_kv_cache.assert_called_once_with(req, scheduler.tree_cache)
        scheduler.metrics_collector.increment_transfer_failed_reqs.assert_not_called()


class TestTokenizerAbortStatus(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_client_abort_is_returned_as_http_499(self):
        rid = "cancelled-rid"
        manager = SimpleNamespace(
            rid_to_state={rid: object()},
            enable_lora=False,
        )
        state = SimpleNamespace(
            obj=SimpleNamespace(rid=rid, lora_path=None),
        )
        out = {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": "Aborted by AbortReq.",
                    "status_code": 499,
                    "err_type": "RequestCancelled",
                }
            }
        }

        with self.assertRaises(fastapi.HTTPException) as exc:
            await TokenizerManager._handle_abort_finish_reason(
                manager, out, state, is_stream=False
            )

        self.assertEqual(exc.exception.status_code, 499)
        self.assertNotIn(rid, manager.rid_to_state)


if __name__ == "__main__":
    unittest.main()
