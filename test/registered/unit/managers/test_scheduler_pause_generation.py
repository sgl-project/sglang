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

from sglang.srt.beam_search.batch_tail import append_beam_tail
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.managers.hisparse_coordinator import HiSparseAct, HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
)
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import RetractionBackup
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class _FakeReq(SimpleNamespace):
    """A request stand-in compared and hashed by identity, like Req."""

    owner_list = None
    __eq__ = object.__eq__
    __hash__ = object.__hash__


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
        scheduler.pause_state = PauseState()
        scheduler.enable_overlap = False
        scheduler.enable_skip_finishing_decode = False
        scheduler.enable_overlap_mlx = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
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
        # Static pools: no unified KV storage behind the allocator.
        scheduler.token_to_kv_pool_allocator.storage = None
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
        scheduler.disagg_prefill_bootstrap_queue = None
        scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        scheduler.disagg_decode_prealloc_queue = None
        scheduler.disagg_decode_transfer_queue = None
        scheduler.dllm_config = None
        scheduler.uocr_runtime = None
        scheduler.server_args = MagicMock()
        scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
        # pause_generation zeros gen_throughput and flushes KV events.
        scheduler.metrics_reporter = MagicMock()
        scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
        scheduler.invariant_checker = MagicMock()
        scheduler.kv_events_publisher = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.output_streamer = MagicMock()
        scheduler.send_abort_output = MagicMock()
        scheduler.init_request_retirer()
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

        def record(req, is_retracted=False):
            requeue_log.append({"req": req, "is_retracted": req.is_retracted})
            # The queue door takes every request out of its list.
            if req.owner_list is not None:
                req.owner_list.leave([req])

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
        # Committed == allocated: without spec decoding, release_kv_cache
        # asserts there is no overallocated tail.
        req.kv.kv_committed_len = 5
        req.kv.kv_allocated_len = 5

        copy = _PendingCopy()
        controller = MagicMock(ack_write_queue=[])
        controller.ack_backup_queue.qsize.return_value = 0

        def write(device_indices, node_id, extra_pools=()):
            controller.ack_write_queue.append(HiCacheAck(None, copy, [node_id]))
            return torch.arange(len(device_indices))

        controller.write.side_effect = write

        manager = object.__new__(DecodeKVCacheOffloadManager)
        # No unified window sub-pool, so offloads carry no extra SWA pool transfers.
        manager.kv_cache = None
        manager._unified_window = False
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

        def record_free(kv_info, ranges):
            frees.append((req.rid, copy.done))

        # release_kv_cache: an unclaimed row is freed through free_kv_row, then
        # the req_to_token row is given back (a real pool clears req_pool_idx).
        scheduler.tree_cache.claim_kv_row.return_value = False
        scheduler.tree_cache.free_kv_row.side_effect = record_free
        scheduler.token_to_kv_pool_allocator.page_size = 1
        scheduler.tree_cache.token_to_kv_pool_allocator = (
            scheduler.token_to_kv_pool_allocator
        )
        scheduler.tree_cache.logical_page_size = 1
        scheduler.tree_cache.req_to_token_pool.free.side_effect = lambda released_req: (
            setattr(released_req.kv, "req_pool_idx", None)
        )
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
        scheduler.token_to_kv_pool_allocator.pools_share_memory = False
        scheduler.token_to_kv_pool_allocator.check_decode_capacity.return_value = False
        scheduler.tree_cache.req_to_token_pool.mamba_allocator = None
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.ipc_channels = MagicMock()
        scheduler.rust_server = None
        scheduler.output_streamer = MagicMock()
        scheduler.update_running_batch(batch)

    def paused_time_seen_while_draining(self, scheduler, mode, now):
        # Records the pause time busy accounting sees, up to `now`, while the held
        # result drains.
        seen = []
        scheduler.enable_overlap = True
        scheduler.last_batch = MagicMock()
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])
        scheduler.process_batch_result = lambda batch, result: seen.append(
            scheduler.pause_state.consume_paused_time(0.0, now)
        )
        with patch(
            "sglang.srt.managers.scheduler_components.pause_state.time.monotonic",
            return_value=now,
        ):
            scheduler.pause_generation(PauseGenerationReqInput(mode=mode))
        return seen

    def test_first_retract_starts_the_pause_after_draining(self):
        scheduler = self._new_scheduler()
        self.assertEqual(
            self.paused_time_seen_while_draining(scheduler, "retract", 3.5), [0.0]
        )
        # The window opened at 3.5.
        self.assertEqual(scheduler.pause_state.consume_paused_time(0.0, 4.0), 0.5)

    def test_retract_after_in_place_keeps_the_first_pause_start(self):
        scheduler = self._new_scheduler()
        with patch(
            "sglang.srt.managers.scheduler_components.pause_state.time.monotonic",
            return_value=0.125,
        ):
            scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))
        self.assertEqual(
            self.paused_time_seen_while_draining(scheduler, "retract", 10.125), [10.0]
        )
        # The window still opens at 0.125.
        self.assertEqual(scheduler.pause_state.consume_paused_time(0.0, 1.125), 1.0)

    def test_inplace_only_sets_flag(self):
        """An in-place pause sets the flag and starts the pause window; it leaves the
        batches and the chunked request alone."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.cur_batch_for_debug = MagicMock()
        scheduler.chunked_reqs.join([MagicMock(owner_list=None)])

        original_last_batch = scheduler.last_batch
        original_cur_batch = scheduler.cur_batch_for_debug
        original_chunked_req = scheduler.chunked_req

        scheduler.enable_overlap = True
        scheduler.last_batch.forward_mode.is_extend.return_value = True
        scheduler.result_queue = deque([(MagicMock(), MagicMock())])

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler.pause_state.paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.last_batch, original_last_batch)
        self.assertIs(scheduler.cur_batch_for_debug, original_cur_batch)
        self.assertIs(scheduler.chunked_req, original_chunked_req)
        # The held result stays queued and nothing is merged.
        self.assertEqual(len(scheduler.result_queue), 1)
        original_last_batch.filter_batch.assert_not_called()
        scheduler.running_batch.merge_batch.assert_not_called()

    def test_cache_flush_while_paused_is_published_before_resume(self):
        """A cache flush during a retract pause (the weight-update window) must be
        published by the paused loop, not held until continue_generation."""
        scheduler = self._new_scheduler()
        scheduler.tree_cache = RadixCache.create_simulated(enable_kv_cache_events=True)
        scheduler.tree_cache.take_events()  # Drop the construction-time clear.
        published = []
        scheduler.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=None,
            tree_cache=scheduler.tree_cache,
            send_metrics_from_scheduler=None,
            max_running_requests=1,
            max_total_num_tokens=1,
            get_stats=lambda: None,
        )
        scheduler.kv_events_publisher.enable_kv_cache_events = True
        scheduler.kv_events_publisher.kv_event_publisher = SimpleNamespace(
            publish=published.append
        )
        scheduler.grammar_manager = MagicMock(grammar_queue=[])
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.draft_worker = None
        scheduler.request_receiver = object.__new__(SchedulerRequestReceiver)
        scheduler.deferred_responses = DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=None,
            send_abort=None,
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
        self.assertTrue(scheduler.flush_cache(empty_cache=False))
        scheduler._record_scheduler_state_for_paused_engine()

        self.assertEqual(
            [type(e).__name__ for batch in published for e in batch.events],
            ["AllBlocksCleared"],
            "flush_cache while paused was not published by the paused loop",
        )

    def test_abort_mode_rejected_at_scheduler(self):
        """abort mode must be rejected by the scheduler-side assert."""
        scheduler = self._new_scheduler()

        with self.assertRaises(AssertionError):
            scheduler.pause_generation(PauseGenerationReqInput(mode="abort"))

    def test_retract_clears_last_batch_state(self):
        """retract mode should clear last_batch and cur_batch_for_debug."""
        scheduler = self._new_scheduler()
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = False
        scheduler.cur_batch_for_debug = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler.pause_state.paused)
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
        scheduler.chunked_reqs.join([MagicMock(owner_list=None)])
        scheduler.running_reqs.join([run_req_a, run_req_b, last_req])
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(
            [entry["req"] for entry in requeue_log], [run_req_a, run_req_b, last_req]
        )
        self.assertEqual(len(scheduler.running_reqs), 0)
        self.assertTrue(all(entry["is_retracted"] for entry in requeue_log))
        self.assertEqual(
            [req.retraction_count for req in (run_req_a, run_req_b, last_req)],
            [1, 1, 1],
        )
        self.assertEqual(scheduler.running_batch.reqs, [])
        self.assertFalse(scheduler.running_batch.batch_is_full)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler.last_batch)

    def test_retract_after_dp_conversion_releases_each_req_once(self):
        """retract after DP attention converted the running batch to extend
        (running_batch is last_batch) releases each request once."""
        scheduler = self._new_scheduler()
        run_req_a = self._make_req("run-a")
        run_req_b = self._make_req("run-b")
        converted = self._make_batch(
            scheduler,
            reqs=[run_req_a, run_req_b],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        scheduler.running_batch = converted
        scheduler.last_batch = converted
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(
            [entry["req"] for entry in requeue_log], [run_req_a, run_req_b]
        )
        self.assertEqual(
            [req.retraction_count for req in (run_req_a, run_req_b)], [1, 1]
        )

    def test_retract_fold_in_releases_via_scheduler_hisparse_coordinator(self):
        """retract of a folded-in last extend batch must release through the scheduler-owned hisparse coordinator."""
        scheduler = self._new_scheduler()
        scheduler.hisparse_coordinator = MagicMock()
        scheduler.hisparse_coordinator.ack_staging_queue = IndexedReqList(
            ListKind.HISPARSE_STAGING
        )
        scheduler.init_request_retirer()
        last_req = self._make_req("last")
        last_req.kv.req_pool_idx = 0
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        scheduler.last_batch = self._make_batch(
            scheduler,
            reqs=[last_req],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        requeue_log = self._spy_requeue(scheduler)

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
            side_effect=lambda req, *_, **__: setattr(req.kv, "req_pool_idx", None),
        ):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.hisparse_coordinator.release_request.assert_called_once_with(last_req)
        self.assertEqual([entry["req"] for entry in requeue_log], [last_req])

    def test_retract_takes_hisparse_staging_requests(self):
        scheduler = self._new_scheduler()
        # `older` was prefilled a step earlier and still waits for its host
        # copy; `newer` is the last prefill batch, staged too.
        older = self._make_req("older")
        newer = self._make_req("newer")
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.ack_staging_queue = IndexedReqList(
            ListKind.HISPARSE_STAGING, req_of=lambda act: act.req
        )
        coordinator.write_staging_stream = MagicMock()
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((2, 8), dtype=torch.int64)
        )
        coordinator.token_to_kv_pool_allocator = MagicMock()
        coordinator.mem_pool_host = MagicMock()
        coordinator.mem_pool_host.allocated_host_indices.return_value = torch.empty(0)
        coordinator.req_to_host_pool = torch.full((2, 8), -1, dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.zeros(2, dtype=torch.int64)
        coordinator._skip_first_backup = {}
        coordinator.spec_swap = MagicMock()
        for slot, req in enumerate((older, newer)):
            req.kv.req_pool_idx = slot
            req.extend_end = 3
            req.hisparse_staging = True
            coordinator.ack_staging_queue.join([HiSparseAct(None, MagicMock(), req)])
        scheduler.hisparse_coordinator = coordinator
        scheduler.init_request_retirer()
        scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=True)
        prefill = self._make_batch(
            scheduler,
            reqs=[newer],
            forward_mode=ForwardMode.EXTEND,
            with_tensors=True,
        )
        scheduler.last_batch = prefill
        requeue_log = self._spy_requeue(scheduler)

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
            side_effect=lambda req, *_, **__: setattr(req.kv, "req_pool_idx", None),
        ):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual([entry["req"] for entry in requeue_log], [newer, older])
        self.assertEqual(len(coordinator.ack_staging_queue), 0)
        self.assertEqual(coordinator.write_staging_stream.synchronize.call_count, 2)
        for req in (older, newer):
            self.assertEqual(req.retraction_count, 1)
            self.assertFalse(req.hisparse_staging)
            self.assertFalse(req.kv.holds_kv)

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
        scheduler.chunked_reqs.join([MagicMock(owner_list=None)])
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertIsNone(scheduler.chunked_req)
        self.assertFalse(scheduler.running_batch.batch_is_full)

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

        def process_result(batch, result):
            self.assertFalse(scheduler.result_queue)
            event_log.append("drain")

        scheduler.process_batch_result = MagicMock(side_effect=process_result)
        scheduler._add_request_to_queue = MagicMock(
            side_effect=lambda req, **_: event_log.append(
                "requeue-released" if req.is_retracted else "requeue-unreleased"
            )
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(event_log, ["drain", "requeue-released"])
        self.assertEqual(len(scheduler.result_queue), 0)

    def test_retract_disagg_prefill_keeps_live_chunked_req(self):
        """disagg-PREFILL retract must leave a live mid-chunk chunked_req untouched."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler._add_request_to_queue = MagicMock()
        scheduler.last_batch = None

        chunked_req = MagicMock(owner_list=None)
        chunked_req.finished.return_value = False
        scheduler.chunked_reqs.join([chunked_req])

        with patch("sglang.srt.managers.scheduler.retract_all") as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        mock_retract_all.assert_not_called()
        scheduler._add_request_to_queue.assert_not_called()
        self.assertIs(scheduler.chunked_req, chunked_req)

    def test_retract_drains_overlap_queue(self):
        """A retract pause processes the one held result and leaves the PD prefill
        chunk preparation (`prefill_chunk_processed`) to the loop."""
        for mode in (
            DisaggregationMode.NULL,
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            with self.subTest(mode=mode):
                scheduler = self._new_scheduler()
                scheduler.enable_overlap = True
                scheduler.disaggregation_mode = mode
                batch = ScheduleBatch(reqs=[], forward_mode=ForwardMode.EXTEND)
                result = SimpleNamespace(delay_sample_func=None)
                scheduler.last_batch = batch
                scheduler.result_queue = deque([(batch, result)])
                scheduler.process_batch_result = MagicMock()

                scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

                scheduler.process_batch_result.assert_called_once_with(batch, result)
                self.assertFalse(scheduler.result_queue)
                self.assertFalse(batch.prefill_chunk_processed)

    def test_pd_decode_retract_includes_transferred_waiting_reqs(self):
        """A transferred request waiting for a running slot holds a request row
        and KV; retract must rebuild it too, so a paused flush cannot free it."""
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = MagicMock()
        held = IndexedReqList(ListKind.PD_HELD)
        scheduler.disagg_decode_prealloc_queue.hold_rebootstrap.side_effect = (
            lambda req: move([req], held)
        )

        def _req(rid, req_pool_idx):
            return _FakeReq(
                rid=rid,
                finished=lambda: False,
                output_ids=[7],
                time_stats=MagicMock(),
                kv=SimpleNamespace(holds_kv=req_pool_idx is not None),
            )

        running = _req("running", 0)
        staged = _req("staged", 1)
        unstaged = _req("unstaged", None)
        scheduler.running_batch.reqs = [running]
        scheduler.waiting_queue.join([staged, unstaged])

        with patch("sglang.srt.managers.scheduler.retract_all") as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(mock_retract_all.call_args.kwargs["reqs"], [running, staged])
        # Rebootstrap recomputes the KV from the prefill, so the retract skips the
        # device->host KV offload rather than offload-then-delete it.
        self.assertEqual(mock_retract_all.call_args.kwargs["offload_kv"], False)
        self.assertEqual(list(scheduler.waiting_queue), [unstaged])
        self.assertEqual(list(held), [running, staged])

    def test_retract_aborts_requests_the_recompute_cannot_rebuild(self):
        for decode in (False, True):
            with self.subTest(decode=decode):
                scheduler = self._new_scheduler()
                scheduler.disagg_decode_prealloc_queue = MagicMock()
                requeue_log = self._spy_requeue(scheduler)
                if decode:
                    scheduler.disaggregation_mode = DisaggregationMode.DECODE
                    get_context().override("test", disaggregation_mode="decode")
                plain, sent, unsent = map(self._make_req, ("plain", "sent", "unsent"))
                for req in (plain, sent, unsent):
                    req.output_ids.append(7)
                sent.input_embeds = unsent.input_embeds = [[0.0]] * 3
                sent.send_token_offset = 1
                scheduler.running_batch = self._make_batch(
                    scheduler, reqs=[plain, sent, unsent], with_tensors=True
                )

                scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

                aborted = [
                    c.args[1] for c in scheduler.send_abort_output.call_args_list
                ]
                # An aborted request drops its cache-side state like a queued one.
                finished = [
                    c.args[0] for c in scheduler.tree_cache.finish.call_args_list
                ]
                self.assertEqual(finished, [r.cache_request_handle for r in aborted])
                if decode:
                    queue = scheduler.disagg_decode_prealloc_queue
                    held = [c.args[0] for c in queue.hold_rebootstrap.call_args_list]
                    self.assertEqual((aborted, held), ([sent, unsent], [plain]))
                else:
                    requeued = [entry["req"] for entry in requeue_log]
                    self.assertEqual((aborted, requeued), ([sent], [plain, unsent]))

    def test_retract_aborts_beam_groups_and_drops_the_beam_tail(self):
        scheduler = self._new_scheduler()
        requeue_log = self._spy_requeue(scheduler)
        plain, leader = self._make_req("plain"), self._make_req("leader")
        member_rows = torch.tensor([10, 11])
        group = SimpleNamespace(
            leader=leader,
            prompt_len=3,
            member_rows=member_rows,
            member_rows_cpu=member_rows,
            all_rows=torch.tensor([1, 10, 11]),
            num_member_rows=2,
            retired=False,
        )
        leader.beam_group = group
        batch = self._make_batch(scheduler, reqs=[plain, leader], with_tensors=True)
        append_beam_tail(batch)
        scheduler.running_batch = batch

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        aborted = [c.args[1] for c in scheduler.send_abort_output.call_args_list]
        self.assertEqual(aborted, [leader])
        self.assertEqual([entry["req"] for entry in requeue_log], [plain])
        storage = scheduler.tree_cache.token_to_kv_pool_allocator.storage
        storage.release_rows.assert_called_once_with([10, 11])
        self.assertIsNone(group.member_rows)
        self.assertIsNone(batch.beam_tail)
        # The first filter after resume finds no tail to strip.
        batch.filter_batch()

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
        scheduler.pause_state = PauseState()
        scheduler.pause_state.pause()

        scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=False)
        )

        scheduler.disagg_decode_prealloc_queue.enqueue_held_rebootstrap.assert_called_once_with()
        self.assertFalse(scheduler.pause_state.paused)

    def test_pd_decode_abort_finishes_a_held_rebootstrap(self):
        scheduler = self._new_scheduler()
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc_queue.scheduler = scheduler
        prealloc_queue.queue = IndexedReqList(ListKind.PD_PREALLOC)
        prealloc_queue.retracted_queue = IndexedReqList(ListKind.PD_RETRACTED)
        prealloc_queue.held_rebootstrap_reqs = IndexedReqList(
            ListKind.PD_HELD, abort=prealloc_queue._abort_held
        )
        prealloc_queue.add = MagicMock()
        scheduler.disagg_decode_prealloc_queue = prealloc_queue
        transfer_queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer_queue.queue = IndexedReqList(ListKind.PD_RECEIVING)
        transfer_queue._deferred_releases = IndexedReqList(ListKind.PD_RELEASING)
        scheduler.disagg_decode_transfer_queue = transfer_queue
        scheduler.deferred_responses = MagicMock()
        scheduler.mm_receiver = None
        scheduler.grammar_manager = MagicMock()
        scheduler.send_abort_output = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.init_request_retirer()
        scheduler.enable_continuous_input_polling = False
        scheduler.pause_state = PauseState()
        scheduler.pause_state.pause()
        aborted, kept = self._make_req("aborted"), self._make_req("kept")
        prealloc_queue.hold_rebootstrap(aborted)
        prealloc_queue.hold_rebootstrap(kept)

        scheduler.abort_request(AbortReq(rid="aborted"))

        self.assertIsInstance(aborted.finished_reason, FINISH_ABORT)
        self.assertIsNone(kept.finished_reason)
        self.assertEqual(list(prealloc_queue.held_rebootstrap_reqs), [kept])
        scheduler.tree_cache.finish.assert_called_once()
        ((abort, sent_req), _) = scheduler.send_abort_output.call_args
        self.assertEqual((abort.rid, sent_req), ("aborted", aborted))

        scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=False)
        )
        prealloc_queue.add.assert_called_once_with(kept, is_rebootstrap=True)

    def test_pd_decode_abort_finishes_a_retracted_request(self):
        """A retracted request aborted while queued must leave finished, with its
        cache-side state and KV backup released and one abort sent."""
        scheduler = self._new_scheduler()
        scheduler.send_abort_output = MagicMock()
        scheduler.beam_coordinator = MagicMock()
        scheduler.init_request_retirer()
        prealloc_queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc_queue.scheduler = scheduler
        prealloc_queue.retracted_queue = IndexedReqList(
            ListKind.PD_RETRACTED, abort=prealloc_queue._abort_retracted
        )
        req = self._make_req("aborted")
        req.kv.retraction_backup = RetractionBackup(cpu_tensors=object())
        prealloc_queue.retracted_queue.join([req])

        prealloc_queue.retracted_queue.abort(AbortReq(rid="aborted"))

        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.tree_cache.finish.assert_called_once_with(
            req.cache_request_handle, CacheRequestOutcome.ABORT
        )
        self.assertIsNone(req.kv.retraction_backup)
        scheduler.send_abort_output.assert_called_once()

    def test_pp_retract_waits_for_in_flight_then_retracts_every_micro_batch(self):
        class StopLoop(Exception):
            pass

        scheduler = self._new_scheduler()
        req_a = self._make_req("a")
        req_b = self._make_req("b")
        mb_a = self._make_batch(
            scheduler, [req_a], forward_mode=ForwardMode.DECODE, with_tensors=True
        )
        mb_b = self._make_batch(
            scheduler, [req_b], forward_mode=ForwardMode.DECODE, with_tensors=True
        )
        scheduler.running_reqs.join([req_a, req_b])
        requeue_log = self._spy_requeue(scheduler)
        # Micro-batch a's result is processed; b's is still in flight.
        scheduler.pp_loop_size = 2
        scheduler.running_mbs = [mb_a, mb_b]
        scheduler.mbs = [mb_a, mb_b]
        scheduler.last_mbs = [mb_a, None]
        scheduler.mb_metadata = [None, None]
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        # A pause, then a later input, arrive while b is in flight.
        pause = PauseGenerationReqInput(mode="retract")
        later = MagicMock()
        inputs = iter([[pause, later]])
        scheduler.request_receiver = SimpleNamespace(
            recv_requests=lambda local_reqs=None: next(inputs, [])
        )
        scheduler._poll_timeout_aborts = MagicMock(return_value=[])
        events = []

        def process_input_requests(reqs):
            for req in reqs:
                if isinstance(req, PauseGenerationReqInput):
                    scheduler.pause_generation(req)
                events.append(
                    (
                        "pause" if req is pause else "later",
                        any(mb is not None for mb in scheduler.mbs),
                        [entry["req"] for entry in requeue_log],
                    )
                )

        scheduler.process_input_requests = process_input_requests
        scheduler.init_pp_loop_state = MagicMock()
        scheduler.get_next_batch_to_run = MagicMock(
            side_effect=lambda running_batch, last_batch: SimpleNamespace(
                running_batch=running_batch, batch_to_run=None
            )
        )
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = MagicMock(
            return_value=(None, None, MagicMock())
        )
        scheduler._pp_commit_proxy_send_work = MagicMock()
        scheduler._pp_process_batch_result = MagicMock(
            side_effect=lambda batch, result: events.append(("result", batch))
        )
        scheduler.on_idle = MagicMock()
        # Three passes over the two slots, then leave the endless loop.
        scheduler.increment_scheduler_iterations = MagicMock(
            side_effect=[None, None, None, StopLoop]
        )
        pp_view = MagicMock(
            pp_size=2,
            pp_async_batch_depth=0,
            pp_rank=0,
            attn_tp_rank=0,
            attn_cp_rank=0,
        )
        with (
            patch("sglang.srt.managers.scheduler.get_parallel", return_value=pp_view),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=pp_view,
            ),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin._pp_exchange_outputs_before_forward",
                return_value=False,
            ),
            self.assertRaises(StopLoop),
        ):
            SchedulerPPMixin.event_loop_pp.__wrapped__(scheduler)

        # b's result drains first; the pause then sees no micro-batch in flight
        # and retracts both micro-batches; the later input runs after it.
        self.assertEqual(
            events,
            [
                ("result", mb_b),
                ("pause", False, [req_a, req_b]),
                ("later", False, [req_a, req_b]),
            ],
        )
        scheduler.get_next_batch_to_run.assert_not_called()
        self.assertTrue(scheduler.pause_state.paused)
        self.assertEqual(scheduler.last_mbs, [None, None])
        self.assertEqual([mb.reqs for mb in scheduler.running_mbs], [[], []])
        self.assertEqual(len(scheduler.running_reqs), 0)

    def test_pp_retract_takes_a_request_in_two_sources_once(self):
        scheduler = self._new_scheduler()
        running = self._make_req("running")
        chunked = self._make_req("chunked")
        chunked.kv.req_pool_idx = 0
        # The chunked request's last chunk ran in the second micro-batch.
        scheduler.chunked_reqs.join([chunked])
        scheduler.pp_loop_size = 2
        scheduler.running_mbs = [
            self._make_batch(
                scheduler, [running], forward_mode=ForwardMode.DECODE, with_tensors=True
            ),
            ScheduleBatch(reqs=[]),
        ]
        scheduler.last_mbs = [
            None,
            self._make_batch(
                scheduler, [chunked], forward_mode=ForwardMode.EXTEND, with_tensors=True
            ),
        ]
        scheduler.running_reqs.join([running])
        requeue_log = self._spy_requeue(scheduler)

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
            side_effect=lambda req, *_, **__: setattr(req.kv, "req_pool_idx", None),
        ):
            scheduler._pp_retract_microbatches()

        self.assertEqual([entry["req"] for entry in requeue_log], [running, chunked])
        self.assertEqual(chunked.retraction_count, 1)
        self.assertIsNone(scheduler.chunked_req)

    def test_dllm_retract_takes_every_row_holder_and_restarts_it(self):
        scheduler = self._new_scheduler()
        dllm_config = DllmConfig(
            algorithm="LowConfidence",
            algorithm_config={},
            block_size=4,
            mask_id=99,
            max_running_requests=4,
        )
        scheduler.dllm_config = dllm_config
        scheduler.dllm_manager = DllmManager(dllm_config=dllm_config, abort=MagicMock())
        decoding, prefilling, done = (
            Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", [1, 2, 3, 4, 5]),
                sampling_params=SamplingParams(),
                dllm_config=dllm_config,
            )
            for rid in ("decoding", "prefilling", "done")
        )
        # A prompt shorter than a block, with output past it.
        decoding.origin_input_ids = array("q", [1, 2])
        decoding.output_ids = array("q", [3, 4, 5])
        # Left out of the last round, mid-decode with an unresolved FDFO block.
        decoding.kv.req_pool_idx = 0
        decoding.dllm_initialized = True
        decoding.dllm_phase = DllmReqPhase.STAGING_DECODE
        decoding.dllm_incomplete_ids = array("q", [7, 99, 99, 99])
        decoding.dllm_algo_state = object()
        prefilling.kv.req_pool_idx = 1
        prefilling.dllm_phase = DllmReqPhase.STAGING_PREFILL
        # Finished and released by the last round's result.
        done.finished_reason = FINISH_LENGTH(length=0)
        done.dllm_phase = DllmReqPhase.STAGING_PREFILL
        scheduler.dllm_manager.waiting_queue.join([decoding, prefilling, done])
        scheduler.dllm_manager.add_staging_reqs([prefilling, done])
        scheduler.running_batch = self._make_batch(scheduler, [])
        scheduler.last_batch = self._make_batch(
            scheduler, [prefilling, done], forward_mode=ForwardMode.DLLM_EXTEND
        )
        requeue_log = self._spy_requeue(scheduler)

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
            side_effect=lambda req, *_, **__: setattr(req.kv, "req_pool_idx", None),
        ):
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(
            [entry["req"] for entry in requeue_log], [prefilling, decoding]
        )
        self.assertEqual(len(scheduler.dllm_manager.waiting_queue), 0)
        self.assertFalse(scheduler.dllm_manager.any_staging_reqs())
        self.assertFalse(decoding.kv.holds_kv)
        self.assertEqual(decoding.dllm_phase, DllmReqPhase.INCOMING_PREFILL)
        self.assertFalse(decoding.dllm_initialized)
        self.assertEqual(len(decoding.dllm_incomplete_ids), 0)
        self.assertIsNone(decoding.dllm_algo_state)


if __name__ == "__main__":
    unittest.main()
