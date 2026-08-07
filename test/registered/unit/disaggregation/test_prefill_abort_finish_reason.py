import asyncio
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(rid, *, to_finish=None, finished_reason=None):
    return SimpleNamespace(
        rid=rid,
        pending_bootstrap=False,
        to_finish=to_finish,
        finished_reason=finished_reason,
        bootstrap_host=FAKE_BOOTSTRAP_HOST,
        return_logprob=False,
        metadata_buffer_index=-1,
        disagg_kv_sender=SimpleNamespace(kv_mgr=None, clear=MagicMock()),
        time_stats=MagicMock(),
    )


def _make_scheduler(req):
    scheduler = SchedulerDisaggregationPrefillMixin.__new__(
        SchedulerDisaggregationPrefillMixin
    )
    scheduler.disagg_prefill_inflight_queue = [req]
    scheduler.attn_cp_cpu_group = MagicMock()
    scheduler.attn_tp_cpu_group = MagicMock()
    scheduler.tree_cache = MagicMock()
    scheduler.output_streamer = MagicMock()
    scheduler.req_to_metadata_buffer_idx_allocator = MagicMock()
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    return scheduler


class TestPrefillAbortFinishReason(CustomTestCase):
    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_pending_abort_promoted_on_transfer_success(
        self, mock_poll, mock_release
    ):
        # A request aborted mid-transfer (e.g., input over max_req_input_len)
        # carries its FINISH_ABORT in to_finish. KVPoll.Success must promote it
        # instead of overwriting it with a spurious FINISH_LENGTH.
        req = _make_req(
            "abort-pending",
            to_finish=FINISH_ABORT(
                "input too long", HTTPStatus.BAD_REQUEST, "BadRequestError"
            ),
        )
        scheduler = _make_scheduler(req)
        mock_poll.return_value = [KVPoll.Success]

        done = scheduler.process_disagg_prefill_inflight_queue()

        self.assertEqual(done, [req])
        self.assertIsNone(req.to_finish)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(req.finished_reason.err_type, "BadRequestError")
        finish_json = req.finished_reason.to_json()
        self.assertEqual(finish_json["type"], "abort")
        self.assertEqual(finish_json["status_code"], HTTPStatus.BAD_REQUEST)

        # The streamed request carries the abort finish_reason on the wire.
        scheduler.output_streamer.stream_output.assert_called_once()
        streamed_reqs = scheduler.output_streamer.stream_output.call_args.args[0]
        self.assertIs(streamed_reqs[0], req)
        self.assertEqual(streamed_reqs[0].finished_reason.to_json()["type"], "abort")
        self.assertEqual(
            streamed_reqs[0].finished_reason.to_json()["status_code"],
            HTTPStatus.BAD_REQUEST,
        )

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_normal_success_still_finish_length_zero(
        self, mock_poll, mock_release
    ):
        # A transfer that completes without any pending abort keeps the legacy
        # behavior: FINISH_LENGTH(length=0).
        req = _make_req("normal-success")
        scheduler = _make_scheduler(req)
        mock_poll.return_value = [KVPoll.Success]

        done = scheduler.process_disagg_prefill_inflight_queue()

        self.assertEqual(done, [req])
        self.assertIsNone(req.to_finish)
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        self.assertEqual(req.finished_reason.length, 0)

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache")
    @patch("sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group")
    def test_existing_finish_abort_not_overwritten(self, mock_poll, mock_release):
        # A request that already finished with FINISH_ABORT (written directly to
        # finished_reason) must not be replaced by FINISH_LENGTH.
        req = _make_req(
            "abort-existing",
            finished_reason=FINISH_ABORT(
                "already aborted", HTTPStatus.BAD_REQUEST, "BadRequestError"
            ),
        )
        scheduler = _make_scheduler(req)
        mock_poll.return_value = [KVPoll.Success]

        done = scheduler.process_disagg_prefill_inflight_queue()

        self.assertEqual(done, [req])
        self.assertIsNone(req.to_finish)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.message, "already aborted")
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)

    def test_abort_handler_raises_value_error_for_non_stream(self):
        # The promoted FINISH_ABORT serializes to the wire finish_reason that
        # TokenizerManager._handle_abort_finish_reason rejects for non-streaming
        # requests, so the caller surfaces HTTP 400 instead of HTTP 200.
        req = _make_req(
            "abort-wire",
            to_finish=FINISH_ABORT(
                "input too long", HTTPStatus.BAD_REQUEST, "BadRequestError"
            ),
        )
        out = {"meta_info": {"finish_reason": req.to_finish.to_json()}}
        tm = TokenizerManager.__new__(TokenizerManager)

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(tm._handle_abort_finish_reason(out, MagicMock(), is_stream=False))

        self.assertIn("input too long", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
