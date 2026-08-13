import inspect
import unittest
from unittest.mock import ANY, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

FORBIDDEN_TOKENS = ("self.running_batch", "self.last_batch", "self.cur_batch")

DECISION_METHODS = (
    Scheduler.get_next_batch_to_run,
    Scheduler.get_new_batch_prefill,
    Scheduler._get_new_batch_prefill_raw,
    Scheduler._abort_on_running_timeout,
    Scheduler.is_disable_overlap_for_batch,
    SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run,
    SchedulerDisaggregationPrefillMixin.process_prefill_chunk,
    SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch,
    SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run,
)


class TestDecisionMethodsHaveNoHiddenBatchChannel(unittest.TestCase):
    def test_decision_methods_take_batches_as_params_not_self(self):
        """The batch decision tree must receive running/last batch as params, never via self.*."""
        for method in DECISION_METHODS:
            source = inspect.getsource(inspect.unwrap(method))
            self.assertIn(
                f"def {method.__name__}",
                source,
                msg=f"failed to read the real source of {method.__qualname__}",
            )
            for token in FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    source,
                    msg=(
                        f"{method.__qualname__} references {token}; pass the batch "
                        "explicitly and return it via NextBatchPlan instead."
                    ),
                )


class TestConditionalAggregationBatchDecision(unittest.TestCase):
    def test_conditional_aggregation_uses_standard_scheduler(self):
        scheduler = MagicMock()
        scheduler.conditional_agg_enabled = True
        scheduler.chunked_req = None
        scheduler.waiting_queue = []
        scheduler._conditional_local_chunk_is_stalled.return_value = False
        scheduler.get_new_prebuilt_batch.return_value = None
        expected_plan = MagicMock()
        scheduler.get_next_batch_to_run.return_value = expected_plan
        running_batch = MagicMock()
        last_batch = MagicMock()

        plan = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            scheduler, running_batch, last_batch
        )

        self.assertIs(plan, expected_plan)
        scheduler.get_next_batch_to_run.assert_called_once_with(
            running_batch=running_batch, last_batch=last_batch
        )

    def test_chunked_local_prefill_defers_prebuilt_absorption(self):
        scheduler = MagicMock()
        scheduler.conditional_agg_enabled = True
        scheduler.chunked_req = MagicMock()
        scheduler.waiting_queue = []
        scheduler._conditional_local_chunk_is_stalled.return_value = False
        expected_plan = MagicMock()
        scheduler.get_next_batch_to_run.return_value = expected_plan

        plan = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            scheduler, MagicMock(), MagicMock()
        )

        self.assertIs(plan, expected_plan)
        scheduler.get_new_prebuilt_batch.assert_not_called()

    def test_conditional_local_wait_rechecks_full_batch_after_capacity_increases(self):
        scheduler = MagicMock()
        scheduler.conditional_agg_enabled = True
        scheduler.chunked_req = None
        scheduler.waiting_queue = [MagicMock()]
        scheduler._conditional_local_chunk_is_stalled.return_value = False
        scheduler._conditional_agg_full_capacity = None
        scheduler.token_to_kv_pool_allocator.available_size.side_effect = [
            27_008,
            70_848,
        ]
        scheduler.req_to_token_pool.available_size.return_value = 54
        scheduler.get_new_prebuilt_batch.return_value = None
        expected_plan = MagicMock()
        scheduler.get_next_batch_to_run.return_value = expected_plan
        running_batch = MagicMock()
        running_batch.batch_is_full = True
        last_batch = MagicMock()

        first_plan = (
            SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                scheduler, running_batch, last_batch
            )
        )

        self.assertIs(first_plan, expected_plan)
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(scheduler._conditional_agg_full_capacity, (27_008, 54))

        second_plan = (
            SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                scheduler, running_batch, last_batch
            )
        )

        self.assertIs(second_plan, expected_plan)
        self.assertFalse(running_batch.batch_is_full)
        self.assertIsNone(scheduler._conditional_agg_full_capacity)
        scheduler.get_next_batch_to_run.assert_called_with(
            running_batch=running_batch, last_batch=last_batch
        )

    def test_conditional_local_chunk_defers_remote_preallocation(self):
        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.conditional_agg_enabled = True
        scheduler.chunked_req = MagicMock()
        scheduler.disagg_decode_prebuilt_queue = []
        scheduler.polling_count = 0
        scheduler.polling_interval = 1
        transferred_req = MagicMock()
        scheduler.disagg_decode_transfer_queue.pop_transferred.return_value = [
            transferred_req
        ]

        with patch("sglang.srt.disaggregation.decode.get_disagg") as get_disagg:
            get_disagg.return_value.disaggregation_decode_enable_offload_kvcache = False
            SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        scheduler.disagg_decode_prealloc_queue.resume_retracted_reqs.assert_not_called()
        scheduler.disagg_decode_prealloc_queue.pop_preallocated.assert_not_called()
        scheduler.disagg_decode_transfer_queue.pop_transferred.assert_called_once_with()
        self.assertEqual(
            scheduler.disagg_decode_prebuilt_queue,
            [transferred_req],
        )

    def test_conditional_local_chunk_waits_for_capacity(self):
        scheduler = MagicMock()
        scheduler.conditional_agg_enabled = True
        scheduler.chunked_req.full_untruncated_fill_ids = list(range(32_768))
        scheduler.chunked_req.prefix_indices = list(range(16_384))
        scheduler.max_prefill_tokens = 16_384
        scheduler.chunked_prefill_size = 16_384
        scheduler.token_to_kv_pool_allocator.page_size = 64
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
        scheduler._conditional_local_chunk_is_stalled.return_value = True
        scheduler.get_new_prebuilt_batch.return_value = None
        expected_plan = MagicMock()
        scheduler.get_next_batch_to_run.return_value = expected_plan
        running_batch = MagicMock()
        running_batch.is_empty.return_value = True

        plan = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            scheduler, running_batch, MagicMock()
        )

        self.assertIs(plan, expected_plan)
        scheduler.get_new_prebuilt_batch.assert_called_once_with(running_batch)
        scheduler.get_next_batch_to_run.assert_called_once_with(
            running_batch=running_batch,
            last_batch=ANY,
            allow_prefill=False,
        )

        self.assertTrue(
            SchedulerDisaggregationDecodeMixin._conditional_local_chunk_is_stalled(
                scheduler
            )
        )


if __name__ == "__main__":
    unittest.main()
