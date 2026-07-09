import unittest
from collections import deque
from typing import List, Optional
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cpu_ci(est_time=9, suite="base-c-test-cpu")


class TestSchedulerPauseGeneration(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.chunked_req = None
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.running_batch.batch_is_full = False
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.protected_size.return_value = 0
        scheduler.req_to_token_pool = MagicMock()
        scheduler.hisparse_coordinator = MagicMock()
        scheduler.result_queue = deque()
        # Support _kv_snap diagnostic logging in patched schedulers
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.return_value = 1000
        scheduler.max_total_num_tokens = 1000
        scheduler._get_token_info = MagicMock(
            return_value=PoolStats(
                full_num_used=0,
                full_token_usage=0,
                full_available_size=1000,
                full_evictable_size=0,
            )
        )
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.hisparse_coordinator = None
        scheduler.server_args = MagicMock()
        scheduler.waiting_queue = []
        # pause_generation zeros gen_throughput and flushes KV events.
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
        scheduler.kv_events_publisher = MagicMock()
        return scheduler

    def _make_req(self, rid: str, finished: bool = False) -> Req:
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        if finished:
            req.finished_reason = MagicMock()
        return req

    def _make_batch(
        self,
        scheduler: Scheduler,
        reqs: List[Req],
        forward_mode: Optional[ForwardMode] = None,
        with_tensors: bool = False,
    ) -> ScheduleBatch:
        batch = ScheduleBatch(reqs=reqs)
        batch.device = "cpu"
        batch.forward_mode = forward_mode
        batch.req_to_token_pool = scheduler.req_to_token_pool
        batch.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        batch.tree_cache = scheduler.tree_cache
        batch.hisparse_coordinator = None
        batch.model_config = MagicMock(is_encoder_decoder=False)
        batch.sampling_info = MagicMock()
        batch.spec_info = None
        batch.multimodal_inputs = None
        if with_tensors:
            batch_size = len(reqs)
            batch.req_pool_indices = torch.arange(batch_size, dtype=torch.int64)
            batch.req_pool_indices_cpu = torch.arange(batch_size, dtype=torch.int64)
            batch.seq_lens = torch.full((batch_size,), 4, dtype=torch.int64)
            batch.orig_seq_lens = torch.full((batch_size,), 4, dtype=torch.int32)
            batch.seq_lens_cpu = torch.full((batch_size,), 4, dtype=torch.int64)
            batch.input_ids = None
        return batch

    def _spy_requeue(self, scheduler: Scheduler) -> List[dict]:
        requeue_log: List[dict] = []

        def record(req):
            requeue_log.append({"req": req, "is_retracted": req.is_retracted})

        scheduler._add_request_to_queue = MagicMock(side_effect=record)
        return requeue_log

    def test_inplace_only_sets_flag(self):
        """in_place pause should only set _engine_paused and return."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.cur_batch_for_debug = MagicMock()
        scheduler.chunked_req = MagicMock()

        original_last_batch = scheduler.last_batch
        original_cur_batch = scheduler.cur_batch_for_debug
        original_chunked_req = scheduler.chunked_req

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.last_batch, original_last_batch)
        self.assertIs(scheduler.cur_batch_for_debug, original_cur_batch)
        self.assertIs(scheduler.chunked_req, original_chunked_req)

    def test_inplace_does_not_drain_overlap_queue(self):
        """in_place should not process the overlap result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        scheduler.last_batch = MagicMock()
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.result_queue), 1)

    def test_inplace_does_not_merge_batch(self):
        """in_place should not filter or merge last_batch into running_batch."""
        scheduler = self._new_scheduler()
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        scheduler.last_batch = last_batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        last_batch.filter_batch.assert_not_called()
        scheduler.running_batch.merge_batch.assert_not_called()

    def test_abort_mode_rejected_at_scheduler(self):
        """abort mode must be rejected by the scheduler-side assert."""
        scheduler = self._new_scheduler()

        with self.assertRaises(AssertionError):
            scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

    def test_default_mode_rejected_at_scheduler(self):
        """bare PauseGenerationReqInput defaults to abort and must be rejected."""
        scheduler = self._new_scheduler()

        with self.assertRaises(AssertionError):
            scheduler.pause_generation(PauseGenerationReqInput())

    def test_retract_clears_last_batch_state(self):
        """retract mode should clear last_batch and cur_batch_for_debug."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = False
        scheduler.cur_batch_for_debug = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        self.assertIsNone(scheduler.last_batch)
        self.assertIsNone(scheduler.cur_batch_for_debug)

    def test_retract_requeues_running_then_last_fold_in(self):
        """retract requeues running reqs first, then last extend reqs, all released."""
        scheduler = self._new_scheduler()
        run_req_a = self._make_req("run-a")
        run_req_b = self._make_req("run-b")
        last_req = self._make_req("last")
        scheduler.running_batch = self._make_batch(
            scheduler, reqs=[run_req_a, run_req_b], with_tensors=True
        )
        scheduler.running_batch.batch_is_full = True
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        scheduler.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(
            [entry["req"] for entry in requeue_log], [run_req_a, run_req_b, last_req]
        )
        self.assertTrue(all(entry["is_retracted"] for entry in requeue_log))
        self.assertEqual(
            [req.retraction_count for req in (run_req_a, run_req_b, last_req)],
            [1, 1, 1],
        )
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler.last_batch)

    def test_retract_with_empty_running_uses_last_batch_reqs(self):
        """retract with empty running batch releases and requeues the last extend reqs."""
        scheduler = self._new_scheduler()
        last_req = self._make_req("last")
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        scheduler.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual([entry["req"] for entry in requeue_log], [last_req])
        self.assertTrue(requeue_log[0]["is_retracted"])
        self.assertEqual(last_req.retraction_count, 1)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertIsNone(scheduler.chunked_req)

    def test_retract_releases_through_batch_side_hisparse_coordinator(self):
        """retract with empty running + last fold-in must not touch the scheduler-owned hisparse coordinator."""
        scheduler = self._new_scheduler()
        scheduler.hisparse_coordinator = MagicMock()
        last_req = self._make_req("last")
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.hisparse_coordinator.retract_req.assert_not_called()
        self.assertEqual([entry["req"] for entry in requeue_log], [last_req])

    def test_retract_disagg_prefill_excludes_last_batch(self):
        """retract under disagg prefill must not release or requeue last extend reqs."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        run_req = self._make_req("run")
        last_req = self._make_req("last")
        scheduler.running_batch = self._make_batch(
            scheduler, reqs=[run_req], with_tensors=True
        )
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual([entry["req"] for entry in requeue_log], [run_req])
        self.assertEqual(run_req.retraction_count, 1)
        self.assertEqual(last_req.retraction_count, 0)
        self.assertFalse(last_req.is_retracted)

    def test_retract_decode_last_batch_only_retracts_running(self):
        """retract with a decode last batch only releases and requeues running reqs."""
        scheduler = self._new_scheduler()
        run_req = self._make_req("run")
        running = self._make_batch(
            scheduler,
            reqs=[run_req],
            forward_mode=ForwardMode.DECODE,
            with_tensors=True,
        )
        scheduler.running_batch = running
        scheduler.last_batch = running
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual([entry["req"] for entry in requeue_log], [run_req])
        self.assertEqual(run_req.retraction_count, 1)
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_retract_partial_finished_running_batch(self):
        """retract with mixed finished/unfinished reqs only releases the unfinished ones."""
        scheduler = self._new_scheduler()
        req_unfinished_a = self._make_req("unfinished-a")
        req_finished = self._make_req("finished", finished=True)
        req_unfinished_b = self._make_req("unfinished-b")
        scheduler.running_batch = self._make_batch(
            scheduler,
            reqs=[req_unfinished_a, req_finished, req_unfinished_b],
            with_tensors=True,
        )
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(
            [entry["req"] for entry in requeue_log],
            [req_unfinished_a, req_unfinished_b],
        )
        self.assertEqual(req_unfinished_a.retraction_count, 1)
        self.assertEqual(req_unfinished_b.retraction_count, 1)
        self.assertEqual(req_finished.retraction_count, 0)
        self.assertFalse(req_finished.is_retracted)
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_retract_empty_post_fold_keeps_chunked_req(self):
        """retract with nothing to retract leaves chunked_req and batch_is_full alone."""
        scheduler = self._new_scheduler()
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        chunked_req = MagicMock()
        scheduler.chunked_req = chunked_req
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertIs(scheduler.chunked_req, chunked_req)
        self.assertTrue(scheduler.running_batch.batch_is_full)

    def test_retract_all_finished_clears_fields_without_requeue(self):
        """retract with only finished reqs clears fields but releases nothing."""
        scheduler = self._new_scheduler()
        req_finished_a = self._make_req("finished-a", finished=True)
        req_finished_b = self._make_req("finished-b", finished=True)
        scheduler.running_batch = self._make_batch(
            scheduler, reqs=[req_finished_a, req_finished_b]
        )
        scheduler.running_batch.batch_is_full = True
        scheduler.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertEqual(req_finished_a.retraction_count, 0)
        self.assertEqual(req_finished_b.retraction_count, 0)
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertIsNone(scheduler.chunked_req)

    def test_retract_drain_happens_once_before_release(self):
        """retract with overlap drains the result_queue once before releasing reqs."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        last_req = self._make_req("last")
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        event_log: List[str] = []
        scheduler.process_batch_result = MagicMock(
            side_effect=lambda *args, **kwargs: event_log.append("drain")
        )
        scheduler._add_request_to_queue = MagicMock(
            side_effect=lambda req: event_log.append(
                "requeue-released" if req.is_retracted else "requeue-unreleased"
            )
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(event_log, ["drain", "requeue-released"])
        self.assertEqual(len(scheduler.result_queue), 0)

    def test_retract_fold_in_releases_via_scheduler_hisparse_coordinator(self):
        """retract of a folded-in last extend batch must release through the scheduler-owned hisparse coordinator."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.waiting_queue = []
        scheduler._add_request_to_queue = MagicMock()
        scheduler.server_args = MagicMock()

        req = MagicMock()
        req.finished.return_value = False
        req.req_pool_idx = None
        last_batch = MagicMock()
        last_batch.forward_mode.is_extend.return_value = True
        last_batch.is_empty.return_value = False
        last_batch.reqs = [req]
        scheduler.last_batch = last_batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(req)
        self.assertEqual(
            [call.args[0] for call in scheduler._add_request_to_queue.call_args_list],
            [req],
        )

    def test_retract_empty_running_batch_requeues_nothing(self):
        """retract with empty running_batch must not release or requeue any request."""
        scheduler = self._new_scheduler()
        original_reqs = scheduler.running_batch.reqs

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.waiting_queue), 0)
        self.assertIs(scheduler.running_batch.reqs, original_reqs)

    def test_retract_empty_clears_chunked_req_and_batch_is_full(self):
        """retract with everything empty must still clear chunked_req and batch_is_full."""
        scheduler = self._new_scheduler()
        scheduler.waiting_queue = []
        scheduler.chunked_req = MagicMock()
        scheduler.running_batch.batch_is_full = True

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertIsNone(scheduler.chunked_req)
        self.assertFalse(scheduler.running_batch.batch_is_full)

    def test_retract_drains_overlap_queue(self):
        """retract with overlap enabled should drain the result_queue."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        mock_batch = MagicMock()
        mock_batch.forward_mode.is_extend.return_value = False
        scheduler.last_batch = mock_batch
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        scheduler.process_batch_result = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 0)


if __name__ == "__main__":
    unittest.main()
