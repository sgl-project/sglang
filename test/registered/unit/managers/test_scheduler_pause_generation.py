import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch
from weakref import WeakKeyDictionary

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class _PendingCopy:
    def __init__(self):
        self.done = False

    def synchronize(self):
        self.done = True


def _all_reduce_with_peer(ops, peer_queue_sizes, peer_pending):
    # One simulated peer rank: its (write, backup) ack counts before the drain and
    # its pending flag after it.
    def all_reduce(tensor, op, group):
        ops.append(op)
        if op == torch.distributed.ReduceOp.MIN:
            torch.minimum(
                tensor, torch.tensor(peer_queue_sizes, dtype=tensor.dtype), out=tensor
            )
        else:
            tensor.clamp_(min=peer_pending)

    return all_reduce


class TestSchedulerPauseGeneration(CustomTestCase):
    def setUp(self):
        # The scheduler runs after its process publishes; retraction reads the
        # disaggregation and schedule bags rather than the record it is handed.
        from sglang.srt.server_args import ServerArgs

        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

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
        scheduler.disaggregation_mode = DisaggregationMode.NULL
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
        scheduler.decode_offload_manager = None
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

    def _decode_req_with_pending_offload(
        self, scheduler: Scheduler, tp_world_size: int = 1, offload: bool = True
    ) -> Tuple[Req, List[Tuple[str, bool]]]:
        req = Req(
            rid="run",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3]),
            sampling_params=SamplingParams(),
        )
        req.output_ids.extend([4, 5, 6])
        req.kv.req_pool_idx = 0
        req.kv.kv_committed_len = 5
        req.kv.kv_allocated_len = 6

        copy = _PendingCopy()
        controller = MagicMock(ack_write_queue=[])
        controller.ack_backup_queue.qsize.return_value = 0

        def write(device_indices, node_id):
            controller.ack_write_queue.append(HiCacheAck(None, copy, [node_id]))
            return torch.arange(len(device_indices))

        controller.write.side_effect = write

        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8).unsqueeze(0)
        )
        manager.page_size = 1
        manager.offload_stride = 1
        manager.request_counter = 0
        manager.tp_world_size = tp_world_size
        manager.tp_group = None
        manager.cache_controller = controller
        manager.decode_host_mem_pool = MagicMock()
        manager.ongoing_offload = {}
        manager.ongoing_backup = {}
        manager.offloaded_state = WeakKeyDictionary()
        manager.offload_inflight = WeakKeyDictionary()
        if offload:
            self.assertTrue(manager.offload_kv_cache(req))
        scheduler.decode_offload_manager = manager

        frees: List[Tuple[str, bool]] = []

        def free_kv_row(released_req, **kwargs):
            frees.append((released_req.rid, copy.done))
            released_req.kv.req_pool_idx = None
            released_req.kv.mark_kv_released()

        scheduler.tree_cache.cache_finished_req.side_effect = free_kv_row
        return req, frees

    def _retract_by_pause(self, scheduler: Scheduler, req: Req):
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler.running_batch = self._make_batch(
            scheduler, reqs=[req], with_tensors=True
        )
        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

    def _retract_by_kv_full(self, scheduler: Scheduler, req: Req):
        batch = self._make_batch(
            scheduler, reqs=[req], forward_mode=ForwardMode.DECODE, with_tensors=True
        )
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.token_to_kv_pool_allocator.page_size = 1
        scheduler.token_to_kv_pool_allocator.check_decode_capacity.return_value = False
        scheduler.tree_cache.req_to_token_pool.mamba_allocator = None
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.ipc_channels = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.update_running_batch(batch)

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

    def test_paused_engine_accounting_uses_current_scheduler_state(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle = MagicMock()

        for is_idle in (True, False):
            with self.subTest(is_idle=is_idle):
                scheduler.is_fully_idle.return_value = is_idle
                scheduler.metrics_reporter.reset_mock()

                scheduler._record_scheduler_state_for_paused_engine()

                if is_idle:
                    scheduler.metrics_reporter.record_scheduler_idle.assert_called_once_with()
                    scheduler.metrics_reporter.record_scheduler_active.assert_not_called()
                else:
                    scheduler.metrics_reporter.record_scheduler_active.assert_called_once_with()
                    scheduler.metrics_reporter.record_scheduler_idle.assert_not_called()

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

    def test_retract_fold_in_releases_via_scheduler_hisparse_coordinator(self):
        """retract of a folded-in last extend batch must release through the scheduler-owned hisparse coordinator."""
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

        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(last_req)
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

    def test_retract_empty_post_fold_clears_chunked_req_and_batch_is_full(self):
        """retract with nothing to retract still clears chunked_req and batch_is_full."""
        scheduler = self._new_scheduler()
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        scheduler.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertIsNone(scheduler.chunked_req)
        self.assertFalse(scheduler.running_batch.batch_is_full)

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

    def test_retract_empty_running_batch_requeues_nothing(self):
        """retract with empty running_batch must not release or requeue any request."""
        scheduler = self._new_scheduler()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler._engine_paused)
        self.assertEqual(len(scheduler.waiting_queue), 0)
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_retract_disagg_prefill_keeps_live_chunked_req(self):
        """disagg-PREFILL retract must leave a live mid-chunk chunked_req untouched."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._add_request_to_queue = MagicMock()
        scheduler.last_batch = None

        chunked_req = MagicMock()
        chunked_req.finished.return_value = False
        scheduler.chunked_req = chunked_req

        with patch("sglang.srt.managers.scheduler.retract_all") as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        mock_retract_all.assert_not_called()
        scheduler._add_request_to_queue.assert_not_called()
        self.assertIs(scheduler.chunked_req, chunked_req)

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

    def test_pd_decode_retract_requeues_for_rebootstrap(self):
        """PD decode retract should rebootstrap instead of resuming stale CPU KV."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.last_batch = None
        scheduler._add_request_to_queue = MagicMock()
        scheduler.disagg_decode_prealloc_queue = MagicMock()

        req = SimpleNamespace(
            finished=lambda: False,
            output_ids=[10, 11, 12],
            time_stats=MagicMock(),
        )
        scheduler.running_batch.reqs = [req]

        with patch("sglang.srt.managers.scheduler.retract_all") as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler._add_request_to_queue.assert_not_called()
        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.assert_called_once_with(
            req
        )
        self.assertEqual(req.output_ids, [10, 11])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 12)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        # Rebootstrap recomputes the KV from the prefill, so the retract must skip
        # the device->host KV offload rather than offload-then-delete it.
        mock_retract_all.assert_called_once()
        self.assertEqual(mock_retract_all.call_args.kwargs["offload_kv"], False)

    def test_pd_decode_retract_completes_offload_before_free(self):
        """PD decode retract must not free KV an offload copy is still reading, and
        must not leave offload acks that keep the paused engine from going idle."""
        scheduler = self._new_scheduler()
        req, frees = self._decode_req_with_pending_offload(scheduler)

        self._retract_by_pause(scheduler, req)

        self.assertEqual(frees, [("run", True)])
        self.assertEqual(scheduler.decode_offload_manager.ongoing_offload, {})

    def test_kv_full_retract_completes_offload_before_free(self):
        """A KV-full retract that aborts the last decode request must not free KV an
        offload copy is still reading."""
        scheduler = self._new_scheduler()
        req, frees = self._decode_req_with_pending_offload(scheduler)

        self._retract_by_kv_full(scheduler, req)

        self.assertEqual(frees, [("run", True)])

    def test_retraction_refused_while_any_rank_tracks_an_offload_copy(self):
        """Retraction must be refused on every rank, before any KV is freed, while any
        rank still tracks a device-to-host offload copy after the drain."""
        min_op, max_op = torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX
        for case, entrypoint, tp_size, offload, peer_acks, peer_pending in (
            # Nothing is tracked here at the call site; only the peer has a copy.
            ("peer_only", self._retract_by_kv_full, 2, False, [0, 0], 1),
            # This rank has an ack the peer lacks, so the MIN drain leaves its copy.
            ("asymmetric_acks", self._retract_by_pause, 2, True, [0, 0], 0),
            # A tracked copy whose ack is missing; checked locally at TP1.
            ("tracked_without_ack", self._retract_by_pause, 1, True, None, None),
        ):
            with self.subTest(case=case):
                scheduler = self._new_scheduler()
                req, frees = self._decode_req_with_pending_offload(
                    scheduler, tp_world_size=tp_size, offload=offload
                )
                manager = scheduler.decode_offload_manager
                if case == "tracked_without_ack":
                    manager.cache_controller.ack_write_queue.clear()
                    manager.ongoing_offload.clear()
                ops = []
                all_reduce = _all_reduce_with_peer(ops, peer_acks, peer_pending)
                with patch.object(
                    torch.distributed, "all_reduce", side_effect=all_reduce
                ):
                    if case == "asymmetric_acks":
                        # The per-iteration drain only waits for the peer: it neither
                        # fails closed nor adds a collective.
                        manager.check_offload_progress()
                        self.assertEqual(ops, [min_op])
                        ops.clear()
                    with self.assertRaisesRegex(RuntimeError, "refusing to retract"):
                        entrypoint(scheduler, req)

                self.assertEqual(frees, [])
                if tp_size > 1:
                    # A rank that skips the MAX leaves the other ranks waiting in it.
                    self.assertIn(max_op, ops)

    def test_pending_storage_backup_ack_does_not_block_retraction(self):
        """A storage-backup ack left queued by the MIN drain must not block retraction:
        backups read only host memory."""
        scheduler = self._new_scheduler()
        req, frees = self._decode_req_with_pending_offload(scheduler, tp_world_size=2)
        # Only this rank has the ack, so the MIN over backup counts leaves it queued.
        controller = scheduler.decode_offload_manager.cache_controller
        controller.ack_backup_queue.qsize.return_value = 1
        all_reduce = _all_reduce_with_peer([], [1, 0], 0)
        with patch.object(torch.distributed, "all_reduce", side_effect=all_reduce):
            self._retract_by_pause(scheduler, req)

        self.assertEqual(frees, [("run", True)])

    def test_pd_decode_continue_releases_held_rebootstrap(self):
        """continue_generation must enqueue staged rebootstrap reqs on resume."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        scheduler._engine_paused = True

        scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=False)
        )

        scheduler.disagg_decode_prealloc_queue.enqueue_held_rebootstrap.assert_called_once_with()
        self.assertFalse(scheduler._engine_paused)


if __name__ == "__main__":
    unittest.main()
