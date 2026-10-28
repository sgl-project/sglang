import unittest
from array import array
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_lists import (
    make_request_retirer,
    partial_scheduler,
    request_registry_of,
    step_planner_of,
)
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
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
    PendingSteps,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.managers.scheduler_components.step_counters import StepCounters
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class _PendingCopy:
    def __init__(self):
        self.done = False

    def synchronize(self):
        self.done = True


class _Parts(SimpleNamespace):
    @property
    def request_retirer(self):
        # Over the pools and HiSparse as each test last set them.
        return make_request_retirer(
            kv=SimpleNamespace(
                tree_cache=self.kv.tree_cache,
                req_to_token_pool=self.kv.req_to_token_pool,
                token_to_kv_pool_allocator=self.kv.token_to_kv_pool_allocator,
            ),
            hisparse_coordinator=self.mode_owners.hisparse_coordinator,
        )


class _Hub:
    def __init__(self, scheduler):
        self._scheduler = scheduler

    @property
    def request_registry(self):
        # Over the lists and owners as each test last set them.
        return request_registry_of(self._scheduler)

    @property
    def step_planner(self):
        # Over the scheduler's parts as each test last set them.
        return step_planner_of(self._scheduler)


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
        scheduler = partial_scheduler()
        scheduler.parts = _Parts(**vars(scheduler.parts))
        scheduler.hub = _Hub(scheduler)
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.counters = LaunchCounters()
        scheduler.parts.flags = EngineFlags()
        scheduler.parts.lists = CoreLists()
        scheduler.parts.flags._engine_paused = False
        scheduler.enable_overlap = False
        scheduler.parts.flags.enable_skip_finishing_decode = False
        scheduler.parts.counters.cur_batch_for_debug = None
        scheduler.parts.lists.chunked_req = None
        scheduler.parts.lists.decoding = []
        scheduler.parts.pending = PendingSteps()
        scheduler.flat_slot = ScheduleSlot()
        scheduler.parts.kv.tree_cache = MagicMock()
        scheduler.parts.kv.tree_cache.protected_size.return_value = 0
        scheduler.parts.kv.req_to_token_pool = MagicMock()
        scheduler.parts.mode_owners.hisparse_coordinator = MagicMock()
        scheduler.mode.disaggregation_mode = DisaggregationMode.NULL
        # Support _kv_snap diagnostic logging in patched schedulers
        scheduler.parts.kv.token_to_kv_pool_allocator = MagicMock()
        scheduler.parts.kv.token_to_kv_pool_allocator.available_size.return_value = 1000
        scheduler.parts.capacity.max_total_num_tokens = 1000
        scheduler._get_token_info = MagicMock(
            return_value=PoolStats(
                full_num_used=0,
                full_token_usage=0,
                full_available_size=1000,
                full_evictable_size=0,
            )
        )
        scheduler.mode.disaggregation_mode = DisaggregationMode.NULL
        scheduler.parts.mode_owners.disagg_prefill = None
        scheduler.parts.mode_owners.disagg_decode = None
        scheduler.parts.mode_owners.dllm = None
        scheduler.parts.grammar_manager = SimpleNamespace(
            grammar_queue=[], abort_requests=None
        )
        scheduler.parts.mode_owners.hisparse_coordinator = None
        scheduler.parts.decode_offload_manager = None
        scheduler.parts.server_args = MagicMock()
        scheduler.parts.lists.waiting_queue = []
        # pause_generation zeros gen_throughput and flushes KV events.
        scheduler.hub.metrics_reporter = MagicMock()
        scheduler.parts.metrics_sink = MagicMock()
        scheduler.parts.metrics_sink.current_scheduler_metrics_enabled = False
        scheduler.parts.kv_events_publisher = MagicMock()
        scheduler.parts.step_counters = StepCounters()
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
        batch.req_to_token_pool = scheduler.parts.kv.req_to_token_pool
        batch.token_to_kv_pool_allocator = scheduler.parts.kv.token_to_kv_pool_allocator
        batch.tree_cache = scheduler.parts.kv.tree_cache
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

    def _push_pending(self, scheduler: Scheduler, step, result=None) -> None:
        scheduler.parts.pending.push(
            PendingStep(
                step=step,
                result=result,
                launch_ts=0.0,
                after_idle_gap=False,
            )
        )

    def _spy_requeue(self, scheduler: Scheduler) -> List[dict]:
        requeue_log: List[dict] = []

        def record(req):
            requeue_log.append({"req": req, "is_retracted": req.is_retracted})

        scheduler.hub.request_intake = SimpleNamespace(
            add_request_to_queue=MagicMock(side_effect=record)
        )
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
        req.disagg = DisaggReqState()
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
        # Not a UnifiedSWAKVPool, so offloads carry no extra SWA pool transfers.
        manager.kv_cache = None
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
        if offload:
            self.assertTrue(manager.offload_kv_cache(req))
        scheduler.parts.decode_offload_manager = manager
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=MagicMock(), decode_offload_manager=manager
        )

        frees: List[Tuple[str, bool]] = []

        def record_free(kv_info, ranges):
            frees.append((req.rid, copy.done))

        # release_kv_cache: an unclaimed row is freed through free_kv_row, then
        # the req_to_token row is given back (a real pool clears req_pool_idx).
        scheduler.parts.kv.tree_cache.claim_kv_row.return_value = False
        scheduler.parts.kv.tree_cache.free_kv_row.side_effect = record_free
        scheduler.parts.kv.token_to_kv_pool_allocator.page_size = 1
        scheduler.parts.kv.tree_cache.token_to_kv_pool_allocator = (
            scheduler.parts.kv.token_to_kv_pool_allocator
        )
        scheduler.parts.kv.tree_cache.req_to_token_pool.free.side_effect = (
            lambda released_req: setattr(released_req.kv, "req_pool_idx", None)
        )
        return req, frees

    def _retract_by_pause(self, scheduler: Scheduler, req: Req):
        scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.parts.lists.decoding = [req]
        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

    def _retract_by_kv_full(self, scheduler: Scheduler, req: Req):
        scheduler.parts.lists.decoding = [req]
        scheduler.parts.spec_algorithm = SpeculativeAlgorithm.NONE
        scheduler.parts.counters.forward_ct = 1
        scheduler.parts.kv.token_to_kv_pool_allocator.page_size = 1
        scheduler.parts.kv.token_to_kv_pool_allocator.check_decode_capacity.return_value = False
        scheduler.parts.kv.tree_cache.req_to_token_pool.mamba_allocator = None
        scheduler.parts.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.parts.beam_coordinator = MagicMock()
        scheduler.hub.step_planner.get_decode_step(scheduler.flat_slot)

    def drain_with_pause_start_seen(self, scheduler, mode, now):
        # Records the pause start that busy accounting sees while the pending step commits.
        seen = []
        step_counters = scheduler.parts.step_counters
        scheduler.parts.pending.push(MagicMock())
        scheduler.commit_step = lambda record: seen.append(
            step_counters._pause_started_at
        )
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=now):
            scheduler.pause_generation(PauseGenerationReqInput(mode=mode))
        return seen

    def test_first_retract_starts_the_pause_after_draining(self):
        scheduler = self._new_scheduler()
        self.assertEqual(
            self.drain_with_pause_start_seen(scheduler, "retract", 3.5), [None]
        )
        self.assertEqual(scheduler.parts.step_counters._pause_started_at, 3.5)

    def test_retract_after_in_place_keeps_the_first_pause_start(self):
        scheduler = self._new_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=0.125):
            scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))
        self.assertEqual(
            self.drain_with_pause_start_seen(scheduler, "retract", 10.125), [0.125]
        )
        self.assertEqual(scheduler.parts.step_counters._pause_started_at, 0.125)

    def test_inplace_only_sets_flag(self):
        """in_place pause should only set _engine_paused and return."""
        scheduler = self._new_scheduler()
        scheduler.parts.counters.cur_batch_for_debug = MagicMock()
        scheduler.parts.lists.chunked_req = MagicMock()
        running = [self._make_req("run")]
        scheduler.parts.lists.decoding = running

        original_cur_batch = scheduler.parts.counters.cur_batch_for_debug
        original_chunked_req = scheduler.parts.lists.chunked_req

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler.parts.flags._engine_paused)
        # All state must be preserved — no mutation
        self.assertIs(scheduler.parts.lists.decoding, running)
        self.assertIs(scheduler.parts.counters.cur_batch_for_debug, original_cur_batch)
        self.assertIs(scheduler.parts.lists.chunked_req, original_chunked_req)

    def test_inplace_does_not_commit_pending_steps(self):
        """in_place should leave the pending steps to the event loop."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        self._push_pending(
            scheduler, ScheduleBatch(reqs=[], forward_mode=ForwardMode.DECODE)
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="in_place"))

        self.assertTrue(scheduler.parts.flags._engine_paused)
        self.assertEqual(len(scheduler.parts.pending), 1)

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

    def test_retract_clears_the_launched_step(self):
        """retract mode should clear cur_batch_for_debug."""
        scheduler = self._new_scheduler()
        scheduler.parts.counters.cur_batch_for_debug = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler.parts.flags._engine_paused)
        self.assertIsNone(scheduler.parts.counters.cur_batch_for_debug)

    def test_retract_requeues_decoding_requests_in_order(self):
        """retract releases and requeues the decoding requests in list order,
        including one whose prefill launched last."""
        scheduler = self._new_scheduler()
        run_req_a = self._make_req("run-a")
        run_req_b = self._make_req("run-b")
        last_req = self._make_req("last")
        scheduler.parts.lists.decoding = [run_req_a, run_req_b, last_req]
        scheduler.parts.lists.chunked_req = MagicMock()
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
        self.assertEqual(scheduler.parts.lists.decoding, [])
        self.assertIsNone(scheduler.parts.lists.chunked_req)

    def test_retract_releases_via_scheduler_hisparse_coordinator(self):
        """retract must release through the scheduler-owned hisparse coordinator."""
        scheduler = self._new_scheduler()
        scheduler.parts.mode_owners.hisparse_coordinator = MagicMock()
        scheduler.hisparse_coordinator.lists.return_value = []
        last_req = self._make_req("last")
        scheduler.parts.lists.decoding = [last_req]
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(last_req)
        self.assertEqual([entry["req"] for entry in requeue_log], [last_req])

    def test_retract_partial_finished_decoding(self):
        """retract with mixed finished/unfinished reqs only releases the unfinished ones."""
        scheduler = self._new_scheduler()
        req_unfinished_a = self._make_req("unfinished-a")
        req_finished = self._make_req("finished", finished=True)
        req_unfinished_b = self._make_req("unfinished-b")
        scheduler.parts.lists.decoding = [
            req_unfinished_a,
            req_finished,
            req_unfinished_b,
        ]
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
        self.assertEqual(scheduler.parts.lists.decoding, [])

    def test_retract_empty_clears_chunked_req(self):
        """retract with nothing to retract still clears chunked_req."""
        scheduler = self._new_scheduler()
        scheduler.parts.lists.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertIsNone(scheduler.parts.lists.chunked_req)

    def test_retract_all_finished_clears_fields_without_requeue(self):
        """retract with only finished reqs clears fields but releases nothing."""
        scheduler = self._new_scheduler()
        req_finished_a = self._make_req("finished-a", finished=True)
        req_finished_b = self._make_req("finished-b", finished=True)
        scheduler.parts.lists.decoding = [req_finished_a, req_finished_b]
        scheduler.parts.lists.chunked_req = MagicMock()
        requeue_log = self._spy_requeue(scheduler)

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(requeue_log, [])
        self.assertEqual(req_finished_a.retraction_count, 0)
        self.assertEqual(req_finished_b.retraction_count, 0)
        self.assertEqual(scheduler.parts.lists.decoding, [])
        self.assertIsNone(scheduler.parts.lists.chunked_req)

    def test_retract_commit_happens_once_before_release(self):
        """retract with overlap commits the pending step once before releasing reqs."""
        scheduler = self._new_scheduler()
        scheduler.enable_overlap = True
        last_req = self._make_req("last")
        scheduler.parts.lists.decoding = [last_req]
        self._push_pending(
            scheduler,
            self._make_batch(
                scheduler,
                reqs=[last_req],
                forward_mode=ForwardMode.EXTEND,
                with_tensors=True,
            ),
        )
        event_log: List[str] = []

        def process_result(batch, result):
            self.assertFalse(scheduler.parts.pending)
            event_log.append("drain")

        scheduler.process_batch_result = MagicMock(side_effect=process_result)
        scheduler.parts.step_counters = MagicMock()
        scheduler.hub.request_intake = SimpleNamespace(
            add_request_to_queue=MagicMock(
                side_effect=lambda req: event_log.append(
                    "requeue-released" if req.is_retracted else "requeue-unreleased"
                )
            )
        )

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(event_log, ["drain", "requeue-released"])
        self.assertEqual(len(scheduler.parts.pending), 0)

    def test_retract_nothing_decoding_requeues_nothing(self):
        """retract with nothing decoding must not release or requeue any request."""
        scheduler = self._new_scheduler()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertTrue(scheduler.parts.flags._engine_paused)
        self.assertEqual(len(scheduler.parts.lists.waiting_queue), 0)
        self.assertEqual(scheduler.parts.lists.decoding, [])

    def test_retract_disagg_prefill_keeps_live_chunked_req(self):
        """disagg-PREFILL retract must leave a live mid-chunk chunked_req untouched."""
        scheduler = self._new_scheduler()
        scheduler.mode.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.parts.mode_owners.disagg_prefill = prefill_owner_shell()
        scheduler.hub.request_intake = SimpleNamespace(add_request_to_queue=MagicMock())

        chunked_req = MagicMock()
        chunked_req.finished.return_value = False
        scheduler.parts.lists.chunked_req = chunked_req

        with patch(
            "sglang.srt.managers.scheduler_components.step_planner.retract_all"
        ) as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        mock_retract_all.assert_not_called()
        scheduler.hub.request_intake.add_request_to_queue.assert_not_called()
        self.assertIs(scheduler.parts.lists.chunked_req, chunked_req)

    def test_retract_commits_pending_steps(self):
        """retract commits the pending step before releasing, in every mode."""
        for mode in (
            DisaggregationMode.NULL,
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            with self.subTest(mode=mode):
                scheduler = self._new_scheduler()
                scheduler.enable_overlap = True
                scheduler.mode.disaggregation_mode = mode
                if mode == DisaggregationMode.PREFILL:
                    scheduler.parts.mode_owners.disagg_prefill = prefill_owner_shell()
                if mode == DisaggregationMode.DECODE:
                    scheduler.parts.mode_owners.disagg_decode = decode_owner_shell()
                batch = ScheduleBatch(reqs=[], forward_mode=ForwardMode.EXTEND)
                result = SimpleNamespace(delay_sample_func=None)
                self._push_pending(scheduler, batch, result)
                scheduler.process_batch_result = MagicMock()
                scheduler.parts.step_counters = MagicMock()

                scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

                scheduler.process_batch_result.assert_called_once_with(batch, result)
                self.assertFalse(scheduler.parts.pending)

    def test_pd_decode_retract_requeues_for_rebootstrap(self):
        """PD decode retract should rebootstrap instead of resuming stale CPU KV."""
        scheduler = self._new_scheduler()
        scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.hub.request_intake = SimpleNamespace(add_request_to_queue=MagicMock())
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=MagicMock()
        )

        req = SimpleNamespace(
            finished=lambda: False,
            output_ids=[10, 11, 12],
            time_stats=MagicMock(),
            disagg=DisaggReqState(),
        )
        scheduler.parts.lists.decoding = [req]

        with patch(
            "sglang.srt.managers.scheduler_components.step_planner.retract_all"
        ) as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.hub.request_intake.add_request_to_queue.assert_not_called()
        scheduler.disagg_decode.prealloc_queue.hold_rebootstrap.assert_called_once_with(
            req
        )
        self.assertEqual(req.output_ids, [10, 11])
        self.assertEqual(req.disagg.rebootstrap_forced_output_id, 12)
        self.assertTrue(req.disagg.rebootstrap_in_progress)
        # Rebootstrap recomputes the KV from the prefill, so the retract must skip
        # the device->host KV offload rather than offload-then-delete it.
        mock_retract_all.assert_called_once()
        self.assertEqual(mock_retract_all.call_args.kwargs["offload_kv"], False)

    def test_pd_decode_retract_includes_ready_reqs(self):
        """A transferred request waiting for a running slot holds a request row
        and KV; retract must rebuild it too, so a paused flush cannot free it."""
        scheduler = self._new_scheduler()
        scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=MagicMock()
        )

        def _req(rid):
            return SimpleNamespace(
                rid=rid,
                finished=lambda: False,
                output_ids=[7],
                time_stats=MagicMock(),
                kv=SimpleNamespace(holds_kv=True),
                disagg=DisaggReqState(),
            )

        running, staged = _req("running"), _req("staged")
        scheduler.parts.lists.decoding = [running]
        scheduler.disagg_decode.ready = [staged]

        with patch(
            "sglang.srt.managers.scheduler_components.step_planner.retract_all"
        ) as mock_retract_all:
            scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(mock_retract_all.call_args.kwargs["reqs"], [running, staged])
        self.assertEqual(scheduler.disagg_decode.ready, [])
        held = [
            c.args[0]
            for c in scheduler.disagg_decode.prealloc_queue.hold_rebootstrap.call_args_list
        ]
        self.assertEqual(held, [running, staged])

    def test_pd_decode_retract_completes_offload_before_free(self):
        """PD decode retract must not free KV an offload copy is still reading, and
        must not leave offload acks that keep the paused engine from going idle."""
        scheduler = self._new_scheduler()
        req, frees = self._decode_req_with_pending_offload(scheduler)

        self._retract_by_pause(scheduler, req)

        self.assertEqual(frees, [("run", True)])
        self.assertEqual(scheduler.parts.decode_offload_manager.ongoing_offload, {})

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
                manager = scheduler.parts.decode_offload_manager
                if case == "tracked_without_ack":
                    manager.cache_controller.ack_write_queue.clear()
                ops = []
                all_reduce = _all_reduce_with_peer(ops, peer_acks, peer_pending)
                with patch.object(
                    torch.distributed, "all_reduce", side_effect=all_reduce
                ):
                    if case == "asymmetric_acks":
                        # The per-iteration drain only waits for the peer: it neither
                        # fails closed nor adds a collective.
                        scheduler.disagg_decode.check_offload_progress()
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
        controller = scheduler.parts.decode_offload_manager.cache_controller
        controller.ack_backup_queue.qsize.return_value = 1
        all_reduce = _all_reduce_with_peer([], [1, 0], 0)
        with patch.object(torch.distributed, "all_reduce", side_effect=all_reduce):
            self._retract_by_pause(scheduler, req)

        self.assertEqual(frees, [("run", True)])

    def test_pd_decode_continue_releases_held_rebootstrap(self):
        """continue_generation must enqueue staged rebootstrap reqs on resume."""
        scheduler = self._new_scheduler()
        scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=MagicMock()
        )
        scheduler.parts.flags._engine_paused = True

        scheduler.continue_generation(
            ContinueGenerationReqInput(torch_empty_cache=False)
        )

        scheduler.disagg_decode.prealloc_queue.enqueue_held_rebootstrap.assert_called_once_with()
        self.assertFalse(scheduler.parts.flags._engine_paused)


if __name__ == "__main__":
    unittest.main()
