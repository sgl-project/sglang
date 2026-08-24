import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def test_batched_middle_chunks_leave_previous_running_batch(self):
        chunked_reqs = [
            MagicMock(name="chunked_req_0"),
            MagicMock(name="chunked_req_1"),
        ]
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        last_batch.chunked_req = None
        last_batch.requeue_chunked_reqs = chunked_reqs
        last_batch.batch_size.side_effect = [2, 0]
        running_batch = SimpleNamespace(batch_is_full=True)

        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            SimpleNamespace(chunked_req=None),
            last_batch=last_batch,
            running_batch=running_batch,
        )

        excluded = last_batch.filter_batch.call_args.kwargs["chunked_req_to_exclude"]
        self.assertEqual(set(excluded), set(chunked_reqs))
        self.assertFalse(running_batch.batch_is_full)

    def test_requeued_batched_chunk_keeps_committed_chunk_cache_prefix(self):
        req = MagicMock(pp_batched_chunk_requeued=True)
        scheduler = SimpleNamespace(
            pp_batch_independent_chunks=True,
            tree_cache=MagicMock(),
        )

        Scheduler._init_waiting_req_next_round(scheduler, req)

        req.init_next_round_input.assert_called_once_with()

    def test_requeued_batched_chunk_reuses_request_pool_slot(self):
        scheduler = SimpleNamespace(pp_batch_independent_chunks=True)
        req = SimpleNamespace(
            pp_batched_chunk_requeued=True,
            req_pool_idx=7,
        )

        self.assertTrue(Scheduler._pp_batched_chunk_reuses_req_slot(scheduler, req))
        req.req_pool_idx = None
        self.assertFalse(Scheduler._pp_batched_chunk_reuses_req_slot(scheduler, req))

    def test_requeued_chunks_bypass_slotless_requests_before_pool_saturation(self):
        new_reqs = [
            MagicMock(
                rid=f"new-{i}",
                pp_batched_chunk_requeued=False,
                req_pool_idx=None,
            )
            for i in range(16)
        ]
        reusable_0 = MagicMock(
            rid="reusable-0", pp_batched_chunk_requeued=True, req_pool_idx=3
        )
        reusable_1 = MagicMock(
            rid="reusable-1", pp_batched_chunk_requeued=True, req_pool_idx=9
        )
        waiting_queue = new_reqs + [reusable_0, reusable_1]
        scheduler = SimpleNamespace(pp_batch_independent_chunks=True)
        scheduler._pp_batched_chunk_reuses_req_slot = (
            lambda req: Scheduler._pp_batched_chunk_reuses_req_slot(scheduler, req)
        )

        for num_allocatable_reqs in (0, 1, 15):
            with self.subTest(num_allocatable_reqs=num_allocatable_reqs):
                ordered = Scheduler._pp_order_waiting_queue_for_req_slots(
                    scheduler,
                    waiting_queue,
                    num_allocatable_reqs=num_allocatable_reqs,
                )
                self.assertEqual(ordered, [reusable_0, reusable_1] + new_reqs)

        self.assertEqual(waiting_queue, new_reqs + [reusable_0, reusable_1])
        self.assertIs(
            Scheduler._pp_order_waiting_queue_for_req_slots(
                scheduler, waiting_queue, num_allocatable_reqs=16
            ),
            waiting_queue,
        )


if __name__ == "__main__":
    unittest.main()
