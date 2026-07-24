import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    PauseGenerationReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.forward_resource_lease import (
    ForwardResourceLease,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.chunk_cache import (
    ChunkCache,
    PureSWAChunkCache,
    SWAChunkCache,
)
from sglang.srt.mem_cache.radix_cache import RadixCache

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestForwardResourceLease(CustomTestCase):
    def _make_lease(self):
        calls = MagicMock()
        full_done = MagicMock()
        full_done.synchronize.side_effect = calls.full_synchronize
        generation = [SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 7)]
        req_pool = SimpleNamespace(req_generation=generation)
        allocator = SimpleNamespace(
            free_group_begin=calls.free_group_begin,
            free_group_end=calls.free_group_end,
        )

        def release(req, *, is_insert):
            calls.release(req, is_insert=is_insert)
            req.req_pool_idx = None

        lease = ForwardResourceLease(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            wait_for_read_done=calls.wait_for_read_done,
            release_finished_req=release,
        )
        return lease, calls, req_pool, full_done

    def test_mapping_fence_does_not_retire_before_full_forward(self):
        lease, calls, _, full_done = self._make_lease()
        req = SimpleNamespace(req_pool_idx=1)
        epoch = lease.arm_after_launch([req], full_done_event=full_done)

        self.assertTrue(lease.try_defer_finished_req(req, is_insert=True))
        self.assertEqual(req.req_pool_idx, 1)
        self.assertEqual(lease.num_pending_retirements, 1)

        lease.wait_mapping_read_done()

        self.assertEqual(req.req_pool_idx, 1)
        calls.release.assert_not_called()
        full_done.synchronize.assert_not_called()

        lease.mark_forward_completed(epoch)

        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(
            [entry[0] for entry in calls.mock_calls],
            [
                "wait_for_read_done",
                "free_group_begin",
                "release",
                "free_group_end",
            ],
        )

    def test_request_outside_active_lease_fences_and_retires_immediately(self):
        lease, calls, _, full_done = self._make_lease()
        active_req = SimpleNamespace(req_pool_idx=1)
        other_req = SimpleNamespace(req_pool_idx=0)
        lease.arm_after_launch([active_req], full_done_event=full_done)

        self.assertFalse(lease.try_defer_finished_req(other_req, is_insert=True))
        self.assertFalse(lease.read_pending)
        self.assertEqual(lease.num_pending_retirements, 0)
        calls.wait_for_read_done.assert_called_once_with()

        # Consuming the mapping fence must not discard the physical-resource
        # guard for another request in the same result batch.
        self.assertTrue(lease.try_defer_finished_req(active_req, is_insert=False))
        self.assertEqual(lease.num_pending_retirements, 1)

    def test_generation_change_is_detected_before_retirement(self):
        lease, _, req_pool, full_done = self._make_lease()
        req = SimpleNamespace(req_pool_idx=1)
        epoch = lease.arm_after_launch([req], full_done_event=full_done)
        self.assertTrue(lease.try_defer_finished_req(req, is_insert=False))
        lease.wait_read_done()
        req_pool.req_generation[1] = SimpleNamespace(item=lambda: 8)

        with self.assertRaisesRegex(AssertionError, "was reused"):
            lease.mark_forward_completed(epoch)

    def test_all_generations_are_validated_before_any_retirement(self):
        calls = MagicMock()
        req_pool = SimpleNamespace(
            req_generation=[
                SimpleNamespace(item=lambda: 0),
                SimpleNamespace(item=lambda: 7),
                SimpleNamespace(item=lambda: 9),
            ]
        )
        lease = ForwardResourceLease(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=calls.free_group_begin,
                free_group_end=calls.free_group_end,
            ),
            wait_for_read_done=calls.wait_for_read_done,
            release_finished_req=calls.release,
        )
        first = SimpleNamespace(req_pool_idx=1)
        second = SimpleNamespace(req_pool_idx=2)
        full_done = MagicMock()
        epoch = lease.arm_after_launch(
            [first, second],
            full_done_event=full_done,
        )
        self.assertTrue(lease.try_defer_finished_req(first, is_insert=True))
        self.assertTrue(lease.try_defer_finished_req(second, is_insert=True))
        lease.wait_read_done()
        req_pool.req_generation[2] = SimpleNamespace(item=lambda: 10)

        with self.assertRaisesRegex(AssertionError, "was reused"):
            lease.mark_forward_completed(epoch)

        calls.free_group_begin.assert_not_called()
        calls.release.assert_not_called()

    def test_missing_request_row_consumes_fence(self):
        lease, calls, _, full_done = self._make_lease()
        active_req = SimpleNamespace(req_pool_idx=1)
        lease.arm_after_launch([active_req], full_done_event=full_done)
        active_req.req_pool_idx = None

        self.assertFalse(lease.try_defer_finished_req(active_req, is_insert=True))
        calls.wait_for_read_done.assert_called_once_with()

    def test_pending_epoch_does_not_block_arming_next_forward(self):
        lease, calls, _, first_done = self._make_lease()
        first_req = SimpleNamespace(req_pool_idx=1)
        first_epoch = lease.arm_after_launch([first_req], full_done_event=first_done)
        self.assertTrue(lease.try_defer_finished_req(first_req, is_insert=True))
        lease.wait_read_done()

        second_done = MagicMock()
        lease.arm_after_launch([], full_done_event=second_done)

        self.assertEqual(lease.num_pending_retirements, 1)
        lease.mark_forward_completed(first_epoch)
        self.assertIsNone(first_req.req_pool_idx)
        second_done.synchronize.assert_not_called()

    def test_control_path_synchronizes_before_release(self):
        lease, calls, _, full_done = self._make_lease()
        req = SimpleNamespace(req_pool_idx=1)
        lease.arm_after_launch([req], full_done_event=full_done)
        self.assertTrue(lease.try_defer_finished_req(req, is_insert=True))

        lease.synchronize_all_and_drain()
        lease.synchronize_all_and_drain()

        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(
            [entry[0] for entry in full_done.mock_calls],
            ["synchronize"],
        )
        self.assertEqual(
            [entry[0] for entry in calls.mock_calls],
            [
                "wait_for_read_done",
                "full_synchronize",
                "free_group_begin",
                "release",
                "free_group_end",
            ],
        )


class TestDisaggDecodeResourceLeaseLoop(CustomTestCase):
    def _run_overlap_iterations(
        self,
        *,
        use_lease: bool,
        mutate_result: bool = False,
        second_recv=None,
        defer_result_release: bool = False,
        has_copy_done: bool = True,
    ):
        num_iterations = 3 if defer_result_release else 2
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.request_receiver = MagicMock()
        recv_reqs = [[] for _ in range(num_iterations)]
        recv_reqs[1] = [] if second_recv is None else second_recv
        scheduler.request_receiver.recv_requests.side_effect = [
            *recv_reqs,
            StopIteration,
        ]
        scheduler.process_input_requests = MagicMock()
        scheduler._engine_paused = False
        scheduler.running_batch = MagicMock(name="initial_running_batch")
        scheduler.chunked_req = None

        leased_req = SimpleNamespace(req_pool_idx=0)
        batches = [MagicMock(name=f"batch_{i}") for i in range(num_iterations)]
        for batch in batches:
            batch.reqs = [leased_req]
        calls = MagicMock()
        copy_done_events = []
        for _ in range(num_iterations):
            if has_copy_done:
                event = MagicMock()
                event.synchronize.side_effect = calls.copy_done_synchronize
                copy_done_events.append(event)
            else:
                copy_done_events.append(None)
        results = [
            GenerationBatchResult(copy_done=copy_done_events[i])
            for i in range(num_iterations)
        ]
        plans = [
            SimpleNamespace(
                running_batch=MagicMock(name=f"running_batch_{i}"),
                batch_to_run=batches[i],
            )
            for i in range(num_iterations)
        ]

        scheduler.process_decode_queue = calls.process_decode_queue
        scheduler.get_next_disagg_decode_batch_to_run = calls.plan
        scheduler.get_next_disagg_decode_batch_to_run.side_effect = plans
        scheduler.ngram_embedding_manager = MagicMock()
        scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda batch, **_: batch
        )
        scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
        scheduler._can_overlap_decode_queue_with_forward_resource_lease = MagicMock(
            return_value=use_lease
        )
        scheduler._apply_war_barrier = calls.war_barrier
        scheduler.run_batch = calls.run_batch
        scheduler.run_batch.side_effect = results
        scheduler.forward_stream = object()
        full_done_events = []

        def make_full_done_event():
            event = MagicMock()
            event.synchronize.side_effect = calls.full_done_synchronize
            full_done_events.append(event)
            return event

        scheduler.device_module = SimpleNamespace(Event=make_full_done_event)
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=calls.free_group_begin,
            free_group_end=calls.free_group_end,
        )
        scheduler.req_to_token_pool = SimpleNamespace(
            req_generation=[SimpleNamespace(item=lambda: 7)]
        )

        def release(req, *, is_insert):
            calls.release_finished_req_resources()
            req.req_pool_idx = None

        scheduler.batch_result_processor = SimpleNamespace(
            release_finished_req_resources=release
        )

        def process_inputs(reqs):
            if reqs:
                calls.process_nonempty_input()

        scheduler.process_input_requests.side_effect = process_inputs

        def process_result(_, result, **kwargs):
            resource_lease = kwargs["resource_lease"]
            if result.copy_done is not None:
                result.copy_done.synchronize()
            if mutate_result:
                self.assertIsNotNone(resource_lease)
                resource_lease.wait_read_done()
            if defer_result_release and calls.process_result.call_count == 1:
                self.assertIsNotNone(resource_lease)
                self.assertTrue(resource_lease.try_defer_finished_req(leased_req, True))

        scheduler.process_batch_result = calls.process_result
        scheduler.process_batch_result.side_effect = process_result
        scheduler.launch_batch_sample_if_needed = MagicMock()

        with self.assertRaises(StopIteration):
            scheduler.event_loop_overlap_disagg_decode()

        self._last_full_done_events = full_done_events
        self._last_forward_stream = scheduler.forward_stream
        return [entry[0] for entry in calls.mock_calls]

    def test_admission_runs_before_fence(self):
        self.assertEqual(
            self._run_overlap_iterations(use_lease=True),
            [
                "process_decode_queue",
                "plan",
                "run_batch",
                "process_decode_queue",
                "war_barrier",
                "plan",
                "run_batch",
                "process_result",
                "copy_done_synchronize",
            ],
        )

    def test_result_mutation_consumes_fence(self):
        self.assertEqual(
            self._run_overlap_iterations(use_lease=True, mutate_result=True),
            [
                "process_decode_queue",
                "plan",
                "run_batch",
                "process_decode_queue",
                "war_barrier",
                "plan",
                "run_batch",
                "process_result",
                "copy_done_synchronize",
                "war_barrier",
            ],
        )

    def test_finished_request_stays_leased_through_next_admission(self):
        self.assertEqual(
            self._run_overlap_iterations(
                use_lease=True,
                defer_result_release=True,
            ),
            [
                "process_decode_queue",
                "plan",
                "run_batch",
                "process_decode_queue",
                "war_barrier",
                "plan",
                "run_batch",
                "process_result",
                "copy_done_synchronize",
                "process_decode_queue",
                "war_barrier",
                "plan",
                "run_batch",
                "process_result",
                "copy_done_synchronize",
                "free_group_begin",
                "release_finished_req_resources",
                "free_group_end",
            ],
        )

    def test_full_done_event_is_recorded_on_forward_stream(self):
        self._run_overlap_iterations(use_lease=True)

        self.assertGreaterEqual(len(self._last_full_done_events), 1)
        for event in self._last_full_done_events:
            event.record.assert_called_once_with(stream=self._last_forward_stream)

    def test_control_request_fences_before_dispatch(self):
        pause = MagicMock(spec=PauseGenerationReqInput)
        self.assertEqual(
            self._run_overlap_iterations(
                use_lease=True,
                second_recv=[pause],
            ),
            [
                "process_decode_queue",
                "plan",
                "run_batch",
                "war_barrier",
                "full_done_synchronize",
                "process_nonempty_input",
                "process_decode_queue",
                "plan",
                "run_batch",
                "process_result",
                "copy_done_synchronize",
            ],
        )

    def test_missing_copy_event_uses_full_done_before_release(self):
        calls = self._run_overlap_iterations(
            use_lease=True,
            defer_result_release=True,
            has_copy_done=False,
        )

        self.assertLess(
            calls.index("full_done_synchronize"),
            calls.index("release_finished_req_resources"),
        )

    def test_resource_epoch_keeps_common_result_queue_shape(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_overlap = True
        scheduler.last_batch = MagicMock()
        scheduler.last_batch.forward_mode.is_extend.return_value = False
        scheduler.running_batch = SimpleNamespace(reqs=[], batch_is_full=True)
        scheduler.chunked_req = None
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.result_queue = deque(
            [
                (
                    MagicMock(name="batch"),
                    GenerationBatchResult(forward_resource_epoch=object()),
                )
            ]
        )
        scheduler.process_batch_result = MagicMock()
        scheduler.metrics_reporter = SimpleNamespace(
            last_gen_throughput=1.0,
            current_scheduler_metrics_enabled=False,
        )
        scheduler.kv_events_publisher = MagicMock()

        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))

        scheduler.process_batch_result.assert_called_once()
        self.assertEqual(len(scheduler.result_queue), 0)
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_fallback_keeps_immediate_post_launch_barrier(self):
        self.assertEqual(
            self._run_overlap_iterations(use_lease=False),
            [
                "process_decode_queue",
                "plan",
                "run_batch",
                "war_barrier",
                "process_decode_queue",
                "plan",
                "run_batch",
                "war_barrier",
                "process_result",
                "copy_done_synchronize",
            ],
        )


class TestDisaggDecodeResourceLeaseCapability(CustomTestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._war_barrier_enabled = True
        scheduler.req_to_token_pool = object.__new__(DecodeReqToTokenPool)
        scheduler.tree_cache = object.__new__(ChunkCache)
        scheduler.token_to_kv_pool_allocator = object.__new__(TokenToKVPoolAllocator)
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.enable_unified_memory = False
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(enable_staging=False)
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
        )
        scheduler.model_config = SimpleNamespace(is_multimodal=False)
        return scheduler

    def test_standard_cache_modes_use_resource_leases(self):
        scheduler = self._make_scheduler()
        self.assertTrue(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        for cache_cls, allocator_cls in (
            (ChunkCache, PagedTokenToKVPoolAllocator),
            (RadixCache, TokenToKVPoolAllocator),
            (SWAChunkCache, SWATokenToKVPoolAllocator),
            (PureSWAChunkCache, PureSWATokenToKVPoolAllocator),
        ):
            scheduler.tree_cache = object.__new__(cache_cls)
            scheduler.token_to_kv_pool_allocator = object.__new__(allocator_cls)
            self.assertTrue(
                scheduler._can_overlap_decode_queue_with_forward_resource_lease()
            )

    def test_unknown_resource_wrappers_fail_closed(self):
        class CustomChunkCache(ChunkCache):
            pass

        class CustomTokenAllocator(TokenToKVPoolAllocator):
            pass

        scheduler = self._make_scheduler()
        scheduler.tree_cache = object.__new__(CustomChunkCache)
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        scheduler = self._make_scheduler()
        scheduler.token_to_kv_pool_allocator = object.__new__(CustomTokenAllocator)
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        scheduler = self._make_scheduler()
        scheduler.req_to_token_pool = MagicMock()
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

    def test_complex_resource_modes_keep_global_barrier(self):
        scheduler = self._make_scheduler()
        for attr in (
            "enable_decode_hicache",
            "enable_hisparse",
            "enable_unified_memory",
        ):
            setattr(scheduler, attr, True)
            self.assertFalse(
                scheduler._can_overlap_decode_queue_with_forward_resource_lease()
            )
            setattr(scheduler, attr, False)

        scheduler.server_args.disaggregation_decode_enable_offload_kvcache = True
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        scheduler = self._make_scheduler()
        scheduler.disagg_decode_prealloc_queue.enable_staging = True
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        scheduler = self._make_scheduler()
        scheduler.model_config.is_multimodal = True
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

        scheduler = self._make_scheduler()
        scheduler._war_barrier_enabled = False
        self.assertFalse(
            scheduler._can_overlap_decode_queue_with_forward_resource_lease()
        )

    def test_only_data_plane_inputs_bypass_pending_fence(self):
        generate = MagicMock(spec=TokenizedGenerateReqInput)
        generate.session_id = None
        generate.session_params = None
        abort = MagicMock(spec=AbortReq)
        pause = MagicMock(spec=PauseGenerationReqInput)

        self.assertTrue(Scheduler._can_dispatch_inputs_with_forward_resource_lease([]))
        self.assertTrue(
            Scheduler._can_dispatch_inputs_with_forward_resource_lease(
                [generate, abort]
            )
        )
        self.assertFalse(
            Scheduler._can_dispatch_inputs_with_forward_resource_lease([pause])
        )
        generate.session_id = "session"
        self.assertFalse(
            Scheduler._can_dispatch_inputs_with_forward_resource_lease([generate])
        )


class TestProcessBatchResultResourceLease(CustomTestCase):
    def _make_scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.publish_load_snapshot = MagicMock()
        scheduler.batch_result_processor = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()
        return scheduler

    def test_prebuilt_result_quiesces_before_releasing_decode_resources(self):
        scheduler = self._make_scheduler()
        batch = MagicMock()
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = False
        batch.forward_mode.is_prebuilt.return_value = True
        batch.forward_mode.is_idle.return_value = False
        result = MagicMock()
        order = MagicMock()
        lease = MagicMock()
        lease.synchronize_all_and_drain.side_effect = order.synchronize_and_drain
        scheduler.batch_result_processor.process_batch_result_prebuilt.side_effect = (
            order.process_prebuilt
        )

        scheduler.process_batch_result(batch, result, resource_lease=lease)

        self.assertEqual(
            [entry[0] for entry in order.mock_calls],
            ["synchronize_and_drain", "process_prebuilt"],
        )

    def test_decode_result_keeps_fine_grained_resource_lease(self):
        scheduler = self._make_scheduler()
        batch = MagicMock()
        batch.forward_mode.is_decode.return_value = True
        batch.forward_mode.is_extend.return_value = False
        batch.forward_mode.is_prebuilt.return_value = False
        result = MagicMock()
        lease = MagicMock()

        scheduler.process_batch_result(batch, result, resource_lease=lease)

        lease.synchronize_all_and_drain.assert_not_called()
        scheduler.batch_result_processor.process_batch_result_decode.assert_called_once_with(
            batch,
            result,
            resource_lease=lease,
        )

    def test_idle_result_does_not_serialize_resource_lease(self):
        scheduler = self._make_scheduler()
        batch = MagicMock()
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = False
        batch.forward_mode.is_prebuilt.return_value = False
        batch.forward_mode.is_idle.return_value = True
        result = MagicMock()
        lease = MagicMock()

        scheduler.process_batch_result(batch, result, resource_lease=lease)

        lease.synchronize_all_and_drain.assert_not_called()
        scheduler.batch_result_processor.process_batch_result_idle.assert_called_once_with(
            batch,
            result,
        )


class TestDecodeRetractionResourceLease(CustomTestCase):
    def test_quarantine_is_drained_before_retracting_live_requests(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = False
        scheduler.new_token_ratio_tracker = MagicMock()
        batch = MagicMock()
        batch.batch_size.return_value = 2
        batch.is_empty.return_value = False
        batch.check_decode_mem.side_effect = [False, True]
        before_retract = MagicMock()

        with patch(
            "sglang.srt.managers.scheduler.TEST_RETRACT",
            False,
        ):
            result = scheduler.update_running_batch(
                batch,
                before_decode_retract=before_retract,
            )

        self.assertIs(result, batch)
        before_retract.assert_called_once_with()
        batch.retract_decode.assert_not_called()
        batch.prepare_for_decode.assert_called_once_with()

    def test_test_retract_quiesces_before_retracting(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = False
        scheduler.forward_ct = 0
        scheduler.new_token_ratio_tracker = SimpleNamespace(
            current=0.5,
            decay_step=MagicMock(),
        )
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.available_size.side_effect = [10, 20]
        scheduler.tree_cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_allocator=None)
        )
        scheduler.metrics_reporter = SimpleNamespace(
            num_retracted_reqs=0,
            enable_metrics=False,
        )
        scheduler.server_args = MagicMock()
        order = MagicMock()
        before_retract = MagicMock(side_effect=order.before_retract)
        batch = MagicMock()
        batch.batch_size.return_value = 2
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = True

        def retract_decode(_):
            order.retract_decode()
            return [], 0.5, []

        batch.retract_decode.side_effect = retract_decode

        with patch(
            "sglang.srt.managers.scheduler.TEST_RETRACT",
            True,
        ):
            result = scheduler.update_running_batch(
                batch,
                before_decode_retract=before_retract,
            )

        self.assertIs(result, batch)
        self.assertEqual(
            [entry[0] for entry in order.mock_calls],
            ["before_retract", "retract_decode"],
        )
        batch.prepare_for_decode.assert_called_once_with()


class TestDecodeResultResourceLease(CustomTestCase):
    def _make_processor(self):
        server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            enable_metrics=False,
            enable_hisparse=False,
        )
        metrics_reporter = MagicMock()
        metrics_reporter.num_generated_tokens = 0
        metrics_reporter.forward_ct_decode = 0
        return SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=DisaggregationMode.DECODE,
            enable_overlap=False,
            enable_overlap_mlx=False,
            server_args=server_args,
            model_config=SimpleNamespace(think_end_id=None),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            hisparse_coordinator=None,
            req_to_token_pool=MagicMock(),
            decode_offload_manager=None,
            metrics_collector=MagicMock(),
            metrics_reporter=metrics_reporter,
            draft_worker=MagicMock(),
            model_worker=MagicMock(),
            logprob_result_processor=MagicMock(),
            output_streamer=MagicMock(),
            abort_request=MagicMock(),
        )

    def _make_decode_case(self, *, finishes: bool):
        order = MagicMock()
        req = SimpleNamespace(
            is_retracted=False,
            output_ids=[],
            require_reasoning=False,
            time_stats=SimpleNamespace(set_last_decode_finish_time=MagicMock()),
            update_finish_state=order.update_finish_state,
            finished=MagicMock(return_value=finishes),
            mamba_ping_pong_track_buffer=None,
            return_logprob=False,
            return_sampling_mask=False,
            return_hidden_states=False,
            grammar=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(is_none=MagicMock(return_value=True)),
            return_logprob=False,
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=[7],
            can_run_cuda_graph=True,
            num_correct_drafts=0,
            num_block_accept_tokens=0,
            num_cap_tokens=0,
            speculative_num_draft_tokens=None,
        )
        return order, req, batch, result

    def test_finished_request_can_defer_without_waiting(self):
        processor = self._make_processor()
        order, _, batch, result = self._make_decode_case(finishes=True)
        resource_lease = MagicMock()
        resource_lease.try_defer_finished_req.return_value = True

        with patch.object(
            SchedulerBatchResultProcessor,
            "_handle_finish_state_updated_req",
            side_effect=lambda *_, **kwargs: order.finish_handler(
                kwargs["resource_lease"]
            ),
        ):
            processor.process_batch_result_decode(
                batch,
                result,
                resource_lease=resource_lease,
            )

        resource_lease.wait_read_done.assert_not_called()
        self.assertIs(
            order.finish_handler.call_args.args[0],
            resource_lease,
        )

    def test_unfinished_request_does_not_consume_fence(self):
        processor = self._make_processor()
        _, _, batch, result = self._make_decode_case(finishes=False)
        resource_lease = MagicMock()

        with patch.object(
            SchedulerBatchResultProcessor,
            "_handle_finish_state_updated_req",
        ):
            processor.process_batch_result_decode(
                batch,
                result,
                resource_lease=resource_lease,
            )

        resource_lease.wait_read_done.assert_not_called()

    def test_retirement_keeps_worker_hook_and_cache_release_atomic(self):
        processor = self._make_processor()
        req = SimpleNamespace()
        order = MagicMock()
        processor.model_worker.prepare_for_kv_cache_release.side_effect = (
            lambda _: order.prepare()
        )

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor."
            "release_kv_cache",
            side_effect=lambda *_, **__: order.release(),
        ):
            processor.release_finished_req_resources(req, is_insert=False)

        self.assertEqual(
            [entry[0] for entry in order.mock_calls],
            ["prepare", "release"],
        )


if __name__ == "__main__":
    unittest.main()
