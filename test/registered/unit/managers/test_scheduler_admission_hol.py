"""Regression tests for #22831: head-of-line blocking in the prefill
admission loop of Scheduler._get_new_batch_prefill_raw.

Fix invariant: when add_one_req returns NO_TOKEN for the head-of-queue
req but the adder still has KV slack (rem_total_tokens > 0), the
scheduler must clear batch_is_full and continue walking the waiting
queue instead of breaking. Under true saturation (rem_total_tokens == 0)
the original break path is preserved.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestSchedulerAdmissionHOL(CustomTestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.enable_lora = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_dynamic_chunking = False
        scheduler.enable_overlap = False
        scheduler.enable_priority_scheduling = False
        scheduler.is_mixed_chunk = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.chunked_req = None
        scheduler.truncation_align_size = 0
        scheduler.chunked_prefill_size = 4096
        scheduler.page_size = 1
        scheduler.new_token_ratio = 1.0
        scheduler.max_prefill_tokens = 16384
        scheduler.max_prefill_bs = 128
        scheduler.max_running_requests = 128
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.dllm_config = None
        scheduler.prefill_delayer = None
        scheduler.spec_algorithm = MagicMock()
        scheduler.model_config = MagicMock()

        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False

        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.running_batch.return_logprob = False

        scheduler.tree_cache = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.req_to_token_pool = MagicMock()
        scheduler.policy = MagicMock()
        scheduler.server_args = MagicMock()
        scheduler.server_args.prefill_max_requests = 128

        scheduler.get_num_allocatable_reqs = MagicMock(return_value=128)
        scheduler._get_num_pending_tokens = MagicMock(return_value=0)
        scheduler._add_request_to_queue = MagicMock()

        return scheduler

    def _make_req(self, rid: str, input_len: int):
        req = MagicMock()
        req.rid = rid
        req.origin_input_ids = [0] * input_len
        req.lora_id = None
        req.mamba_pool_idx = None
        return req

    def _make_adder(self, rem_total_tokens, failing_req, failing_result):
        """Build a fake PrefillAdder where `failing_req` returns
        `failing_result` and every other req is admitted."""
        fake_adder = MagicMock()
        fake_adder.can_run_list = []
        fake_adder.preempt_list = []
        fake_adder.new_chunked_req = None
        fake_adder.rem_total_tokens = rem_total_tokens

        def add_one_req(req, **_):
            if req is failing_req:
                return failing_result
            fake_adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        fake_adder.add_one_req.side_effect = add_one_req
        return fake_adder

    def _run_admission(self, scheduler, fake_adder):
        """Call _get_new_batch_prefill_raw with the batch-construction
        tail patched out, so only the admission loop is exercised."""
        with patch(
            "sglang.srt.managers.scheduler.PrefillAdder", return_value=fake_adder
        ), patch(
            "sglang.srt.managers.scheduler.ScheduleBatch"
        ) as mock_batch_cls, patch(
            "sglang.srt.managers.scheduler.PrefillStats"
        ), patch(
            "sglang.srt.managers.scheduler.set_time_batch"
        ):
            mock_batch_cls.init_new.return_value = MagicMock()
            Scheduler._get_new_batch_prefill_raw(
                scheduler, prefill_delayer_single_pass=None
            )

    def test_continue_on_no_token_when_slack_remains(self):
        """NO_TOKEN on an oversized head req + remaining KV slack should
        clear batch_is_full and continue to the next req, not break."""
        scheduler = self._new_scheduler()
        large = self._make_req("large", 11000)
        small = self._make_req("small", 50)
        scheduler.waiting_queue = [large, small]

        fake_adder = self._make_adder(
            rem_total_tokens=8000,
            failing_req=large,
            failing_result=AddReqResult.NO_TOKEN,
        )

        self._run_admission(scheduler, fake_adder)

        self.assertIn(
            small,
            fake_adder.can_run_list,
            "small req behind an oversized NO_TOKEN head must still be admitted "
            "when adder reports KV slack remains",
        )
        self.assertFalse(
            scheduler.running_batch.batch_is_full,
            "batch_is_full must not be sticky-set when the skip guard fires",
        )

    def test_break_preserved_under_true_saturation(self):
        """NO_TOKEN with rem_total_tokens == 0 must still break
        (original path), so the admission loop doesn't churn under
        true KV saturation."""
        scheduler = self._new_scheduler()
        large = self._make_req("large", 11000)
        small = self._make_req("small", 50)
        scheduler.waiting_queue = [large, small]

        fake_adder = self._make_adder(
            rem_total_tokens=0,
            failing_req=large,
            failing_result=AddReqResult.NO_TOKEN,
        )

        self._run_admission(scheduler, fake_adder)

        self.assertNotIn(
            small,
            fake_adder.can_run_list,
            "under true saturation (rem_total_tokens == 0) the original break "
            "path must fire; small reqs should not be admitted",
        )
        self.assertTrue(
            scheduler.running_batch.batch_is_full,
            "batch_is_full must be set under true saturation",
        )

    def test_other_result_breaks_without_guard(self):
        """The NO_TOKEN-specific guard must not leak to other non-CONTINUE
        results (e.g. AddReqResult.OTHER from max-concurrency limits)."""
        scheduler = self._new_scheduler()
        head = self._make_req("head", 100)
        tail = self._make_req("tail", 50)
        scheduler.waiting_queue = [head, tail]

        fake_adder = self._make_adder(
            rem_total_tokens=8000,
            failing_req=head,
            failing_result=AddReqResult.OTHER,
        )

        self._run_admission(scheduler, fake_adder)

        self.assertNotIn(
            tail,
            fake_adder.can_run_list,
            "AddReqResult.OTHER must still break the admission loop; the "
            "HOL guard is NO_TOKEN-specific",
        )


if __name__ == "__main__":
    unittest.main()
