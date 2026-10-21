"""Cancellation and intake rejection must return feature-owned resources."""

import unittest
from contextlib import ExitStack
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.deferred_draft_prefill import DeferredDraftPrefillPool
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeferredDraftPrefillAbortLifecycle(unittest.TestCase):
    def scheduler(self):
        scheduler = object.__new__(Scheduler)
        scheduler.chunked_req = scheduler.mm_receiver = scheduler.dllm_config = None
        scheduler.waiting_queue = []
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.enable_hisparse = False
        scheduler.tree_cache = Mock()
        scheduler.beam_coordinator = Mock()
        scheduler.ipc_channels = Mock()
        scheduler.grammar_manager = Mock()
        scheduler.collect_inflight_reqs = lambda: set()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        return scheduler

    def request(self, rid, session=None):
        req = Req(rid, "", [1, 2], SamplingParams(max_new_tokens=8), session=session)
        req.time_stats.trace_ctx = Mock()
        return req

    def test_repeated_waiting_abort_does_not_exhaust_hidden_admission(self):
        scheduler = self.scheduler()
        pool = DeferredDraftPrefillPool(
            1, 8, 4, "torch.bfloat16", "cpu", allocate_storage=False
        )
        scheduler.deferred_draft_prefill_pool = pool
        with (
            patch(
                "sglang.srt.managers.scheduler.get_serving",
                return_value=SimpleNamespace(weight_version=None),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_host_receive_threshold=1
                ),
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache") as release,
        ):
            for i in range(3):
                req = self.request(f"cancel-{i}")
                # This is the post-transfer/pre-draft-init ownership boundary.
                self.assertTrue(pool.try_alloc(req))
                pool.mark_received(req, 2)
                req.pd_deferred_draft_prefill = True
                scheduler.waiting_queue = [req]
                scheduler.abort_request(AbortReq(rid=req.rid))
                self.assertEqual(scheduler.waiting_queue, [])
                self.assertEqual(pool.available_slots, 1)
                self.assertFalse(req.pd_deferred_draft_prefill)
                self.assertIsInstance(req.finished_reason, FINISH_ABORT)
            self.assertEqual(release.call_count, 3)

    def test_disabled_feature_keeps_waiting_abort_cleanup(self):
        for has_attribute in (False, True):
            with self.subTest(has_attribute=has_attribute):
                scheduler = self.scheduler()
                if has_attribute:
                    scheduler.deferred_draft_prefill_pool = None
                req = self.request("ordinary")
                scheduler.waiting_queue = [req]
                with (
                    patch(
                        "sglang.srt.managers.scheduler.get_serving",
                        return_value=SimpleNamespace(weight_version=None),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler.get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_decode_host_receive_threshold=1
                        ),
                    ),
                    patch("sglang.srt.managers.scheduler.release_kv_cache") as release,
                ):
                    scheduler.abort_request(AbortReq(rid=req.rid))
                release.assert_called_once_with(
                    req, scheduler.tree_cache, checkpoint=True
                )
                self.assertEqual(scheduler.waiting_queue, [])

    def test_validation_rejection_retires_session_and_beam_ownership(self):
        scheduler = self.scheduler()
        session = SimpleNamespace(
            streaming=True, close_on_finish=False, _inflight=False
        )
        req = self.request("session-reject", session=session)

        def create_req(*args, **kwargs):
            session._inflight = True
            return req

        def abort_req():
            session._inflight = False

        session.create_req = Mock(side_effect=create_req)
        session.abort_req = Mock(side_effect=abort_req)
        scheduler.session_controller = {"session": session}
        scheduler.enable_session_radix_cache = False
        scheduler.tokenizer = None
        scheduler.model_config = SimpleNamespace(vocab_size=32, hf_eos_token_id=0)
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.output_streamer = Mock()
        recv_req = SimpleNamespace(
            session_params=SimpleNamespace(id="session"),
            session_id=None,
            bootstrap_port=8998,
            mm_inputs=None,
            input_embeds=None,
            lora_id=None,
            custom_logit_processor=None,
        )
        with (
            ExitStack() as cleanup,
            patch(
                "sglang.srt.managers.scheduler.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_defer_draft_prefill=True,
                    disaggregation_draft_prefill_max_input_tokens=8,
                ),
            ),
        ):
            scheduler._handle_generate_request(recv_req, cleanup)
        session.create_req.assert_called_once()
        session.abort_req.assert_called_once()
        self.assertFalse(session._inflight)
        self.assertIsNone(req.session)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        scheduler.beam_coordinator.retire_group.assert_called_once_with(req)
        scheduler.tree_cache.finish.assert_called_once()
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)


if __name__ == "__main__":
    unittest.main()
