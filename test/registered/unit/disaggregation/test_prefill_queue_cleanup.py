import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class RejectingGrammar:
    def __init__(self):
        self.finished = False

    def accept_token(self, token_id):
        raise ValueError(f"token {token_id} rejected")


class FakeSender:
    def __init__(self):
        self.clear_called = False

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None

    def get_transfer_metric(self):
        return None


class TestPrefillQueueCleanup(CustomTestCase):
    def _run_grammar_failure(self, terminal_poll):
        sender = FakeSender()
        req = SimpleNamespace(
            rid="grammar-failure",
            bootstrap_room=7,
            bootstrap_host="decode.example.com",
            disagg_kv_sender=sender,
            finished_reason=None,
            to_finish=None,
            grammar=RejectingGrammar(),
            inflight_middle_chunks=0,
            metadata_buffer_index=-1,
            output_ids=[],
            pending_bootstrap=False,
            return_logprob=False,
            time_stats=MagicMock(),
        )
        req.finished = lambda: req.finished_reason is not None

        batch = SimpleNamespace(
            reqs=[req],
            spec_info=None,
            prefill_stats=MagicMock(),
            dp_cooperation_info=MagicMock(),
        )
        result = SimpleNamespace(
            logits_output=MagicMock(),
            next_token_ids=SimpleNamespace(tolist=lambda: [17]),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            can_run_cuda_graph=False,
        )
        scheduler = SimpleNamespace(
            attn_cp_cpu_group=MagicMock(),
            attn_tp_cpu_group=MagicMock(),
            batch_result_processor=MagicMock(),
            disagg_prefill_inflight_queue=[],
            metrics_reporter=MagicMock(enable_metrics=False),
            output_streamer=MagicMock(),
            ps=SimpleNamespace(tp_rank=0),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            send_kv_chunk=MagicMock(),
            spec_algorithm=MagicMock(),
            tree_cache=MagicMock(),
        )
        scheduler.spec_algorithm.is_eagle.return_value = False

        with (
            patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req"),
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=[[KVPoll.Transferring], [terminal_poll]],
            ),
        ):
            SchedulerDisaggregationPrefillMixin.process_batch_result_disagg_prefill(
                scheduler, batch, result
            )

            self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
            self.assertIsInstance(req.finished_reason, FINISH_ABORT)
            grammar_abort = req.finished_reason
            release.assert_not_called()

            done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )

            self.assertEqual(done, [])
            self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
            release.assert_not_called()
            scheduler.output_streamer.stream_output.assert_called_once_with(
                [], False, None
            )
            scheduler.output_streamer.stream_output.reset_mock()

            done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )

            self.assertEqual(done, [req])
            self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
            self.assertIs(req.finished_reason, grammar_abort)
            release.assert_called_once_with(req, scheduler.tree_cache)
            scheduler.output_streamer.stream_output.assert_called_once_with(
                [req], False, None
            )

        return sender

    def test_grammar_failure_preserved_after_successful_transfer(self):
        sender = self._run_grammar_failure(KVPoll.Success)
        self.assertTrue(sender.clear_called)

    def test_grammar_failure_preserved_after_failed_transfer(self):
        sender = self._run_grammar_failure(KVPoll.Failed)
        self.assertFalse(sender.clear_called)

    def test_deferred_abort_does_not_suppress_transfer_result(self):
        req = SimpleNamespace(
            rid="deferred-abort",
            bootstrap_room=7,
            bootstrap_host="decode.example.com",
            disagg_kv_sender=FakeSender(),
            finished_reason=None,
            to_finish=FINISH_ABORT("externally aborted"),
            metadata_buffer_index=-1,
            pending_bootstrap=False,
            return_logprob=False,
            time_stats=MagicMock(),
        )
        scheduler = SimpleNamespace(
            attn_cp_cpu_group=MagicMock(),
            attn_tp_cpu_group=MagicMock(),
            disagg_prefill_inflight_queue=[req],
            metrics_reporter=MagicMock(enable_metrics=False),
            output_streamer=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            tree_cache=MagicMock(),
        )

        with (
            patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Success],
            ),
        ):
            done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )

        self.assertEqual(done, [req])
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)
        release.assert_called_once_with(req, scheduler.tree_cache)


if __name__ == "__main__":
    unittest.main()
