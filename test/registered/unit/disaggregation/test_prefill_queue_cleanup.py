import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillBootstrapQueueCleanup(CustomTestCase):
    @patch("sglang.srt.disaggregation.prefill.prepare_abort")
    def test_oversized_request_releases_hicache_prefetch(self, mock_prepare_abort):
        req = SimpleNamespace(
            rid="oversized",
            origin_input_ids=[1, 2, 3],
            return_logprob=False,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
        )
        scheduler = SimpleNamespace(
            enable_hicache_storage=True,
            tree_cache=MagicMock(),
            output_streamer=MagicMock(),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler = scheduler
        queue.max_total_num_tokens = 2

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))

        message = "Request oversized exceeds the maximum number of tokens: 3 > 2"
        scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
        req.time_stats.trace_ctx.abort.assert_called_once_with(
            abort_info={"reason": message}
        )
        mock_prepare_abort.assert_called_once_with(
            req, message, status_code=HTTPStatus.BAD_REQUEST
        )
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_request_at_capacity_keeps_hicache_prefetch(self):
        req = SimpleNamespace(rid="at-capacity", origin_input_ids=[1, 2])
        scheduler = SimpleNamespace(
            enable_hicache_storage=True,
            tree_cache=MagicMock(),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler = scheduler
        queue.max_total_num_tokens = 2

        self.assertFalse(queue._check_if_req_exceed_kv_capacity(req))
        scheduler.tree_cache.release_aborted_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
