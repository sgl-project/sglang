import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillCapacity(CustomTestCase):
    @patch.object(PrefillBootstrapQueue, "_init_kv_manager")
    @patch("sglang.srt.disaggregation.prefill.prepare_abort")
    def test_rejects_against_scheduler_input_limit(
        self, mock_prepare_abort, mock_init_kv_manager
    ):
        scheduler = MagicMock()

        queue = PrefillBootstrapQueue(
            token_to_kv_pool=MagicMock(),
            draft_token_to_kv_pool=None,
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            metadata_buffers=MagicMock(),
            tp_rank=0,
            tp_size=1,
            gpu_id=0,
            bootstrap_port=12345,
            gloo_group=MagicMock(),
            max_req_input_len=1024,
            scheduler=scheduler,
            pp_rank=0,
            pp_size=1,
            transfer_backend=TransferBackend.NIXL,
        )

        self.assertEqual(queue.max_req_input_len, 1024)

        req = MagicMock()
        req.rid = "at-input-limit"
        req.origin_input_ids = [0] * 1024
        req.return_logprob = False

        self.assertTrue(queue._check_if_req_exceed_kv_capacity(req))
        mock_prepare_abort.assert_called_once()
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

        scheduler.output_streamer.reset_mock()
        aborted_req = MagicMock()
        aborted_req.rid = "already-aborted"
        aborted_req.origin_input_ids = [0]
        aborted_req.return_logprob = False
        aborted_req.to_finish = FINISH_ABORT(
            "Input length exceeds the scheduler limit",
            HTTPStatus.BAD_REQUEST,
            "BadRequestError",
        )
        aborted_req.finished_reason = None
        queue.create_sender = MagicMock()

        queue.add(aborted_req, num_kv_heads=1)

        queue.create_sender.assert_not_called()
        self.assertEqual(queue.queue, [])
        self.assertIsNone(aborted_req.to_finish)
        self.assertIsInstance(aborted_req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [aborted_req], False
        )


if __name__ == "__main__":
    unittest.main()
