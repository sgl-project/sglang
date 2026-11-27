import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_weight_version_tracker import (
    KvWeightVersionRecord,
    KvWeightVersionTracker,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import compute_weight_version_spans
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ServingStub:
    def __init__(self, weight_version: str):
        self.weight_version = weight_version


class _ContextStub:
    def __init__(self, serving: _ServingStub):
        self.serving = serving

    def override(self, source, **fields):
        self.serving.weight_version = fields["weight_version"]


class TestSchedulerRecordWeightVersionChange(CustomTestCase):
    def _serving(self, version: str) -> _ServingStub:
        serving = _ServingStub(version)
        for name, value in (
            ("get_serving", serving),
            ("get_context", _ContextStub(serving)),
        ):
            patcher = patch(f"sglang.srt.managers.scheduler.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return serving

    def _scheduler(
        self, *, inflight=(), waiting=(), chunked=None, staging=()
    ) -> SimpleNamespace:
        return SimpleNamespace(
            collect_inflight_reqs=lambda: set(inflight),
            _num_queued_outputs=lambda: {},
            waiting_queue=list(waiting),
            chunked_req=chunked,
            hisparse_coordinator=(
                SimpleNamespace(
                    ack_staging_queue=[SimpleNamespace(req=req) for req in staging]
                )
                if staging
                else None
            ),
        )

    def test_every_source_of_live_requests_is_stamped(self):
        """A request missed here keeps attributing its next tokens to the superseded version."""
        self._serving("v1")
        inflight, queued, chunked, staged = (object() for _ in range(4))
        scheduler = self._scheduler(
            inflight=[inflight], waiting=[queued], chunked=chunked, staging=[staged]
        )

        with patch(
            "sglang.srt.managers.scheduler.record_weight_version_events",
            return_value=0,
        ) as recorder:
            Scheduler.record_weight_version_change(scheduler, new_version="v2")

        self.assertEqual(
            set(recorder.call_args.args[0]), {inflight, queued, chunked, staged}
        )


class _OverlappedReq:
    def __init__(self, output_ids, *, middle_chunks=0):
        self.output_ids = list(output_ids)
        self.weight_version_events = []
        self.inflight_middle_chunks = middle_chunks
        self.is_retracted = False
        self.beam_group = None

    def finished(self):
        return False


class TestVersionChangeWithAnOverlappedResultQueued(CustomTestCase):
    """The real overlap loop: a control request is handled at the top of an iteration, while the previous
    forward's result still waits in result_queue for process_batch_result."""

    _serving = TestSchedulerRecordWeightVersionChange._serving

    def _run(self, req, *, forwards, tokens_per_result=1):
        """Change the version at the top of the second iteration; return the tokens with the version each forward
        ran on, and the spans reported for the request."""
        serving = self._serving("0")
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_mode=ForwardMode.DECODE,
        )
        batch.copy = lambda: batch
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            enable_overlap_mlx=False,
            _engine_paused=False,
            running_batch=batch,
            last_batch=None,
            is_generation=False,
            waiting_queue=[],
            chunked_req=None,
            hisparse_coordinator=None,
            collect_inflight_reqs=lambda: {req},
            is_disable_overlap_for_batch=lambda *args, **kwargs: False,
            _apply_war_barrier=lambda: None,
            on_idle=lambda: None,
        )
        scheduler._num_queued_outputs = lambda: Scheduler._num_queued_outputs(scheduler)
        iteration, launched = 0, []

        def ingest():
            nonlocal iteration
            iteration += 1
            if iteration == 2:
                self.assertEqual(len(scheduler.result_queue), 1)
                Scheduler.record_weight_version_change(scheduler, "1")
            scheduler.gracefully_exit = iteration > forwards + 1

        def run_batch(_):
            launched.append(serving.weight_version)
            return [100 + iteration] * tokens_per_result

        scheduler.ingest_requests = ingest
        scheduler.get_next_batch_to_run = lambda **kwargs: SimpleNamespace(
            running_batch=batch, batch_to_run=batch if iteration <= forwards else None
        )
        scheduler.run_batch = run_batch
        scheduler.process_batch_result = lambda _, result: req.output_ids.extend(result)
        Scheduler.event_loop_overlap(scheduler)
        spans = compute_weight_version_spans(
            req.weight_version_events, serving.weight_version, len(req.output_ids)
        )
        return launched, [(span.version, span.start, span.end) for span in spans]

    def test_a_queued_decode_token_belongs_to_the_version_its_forward_ran_on(self):
        launched, spans = self._run(_OverlappedReq([101, 102]), forwards=2)
        self.assertEqual(launched, ["0", "1"])
        self.assertEqual(spans, [("0", 0, 3), ("1", 3, 4)])

    def test_a_queued_first_and_last_token_belongs_to_the_old_version(self):
        launched, spans = self._run(_OverlappedReq([]), forwards=1)
        self.assertEqual(launched, ["0"])
        self.assertEqual(spans, [("0", 0, 1)])

    def test_a_queued_middle_chunk_commits_no_token(self):
        req = _OverlappedReq([], middle_chunks=1)
        # the queued result is a prefill chunk: the request's first token comes from a forward after the change
        self._run(req, forwards=1, tokens_per_result=0)
        self.assertEqual(req.weight_version_events, [])

    def test_a_request_retracted_after_the_change_keeps_no_count_of_its_dropped_output(
        self,
    ):
        """The change counted the queued token; the retraction in the same iteration drops that token, and the
        request's next one comes from the new version."""
        req = _OverlappedReq([101, 102])
        req.retraction_count, req.input_embeds, req.send_token_offset = 0, None, 0
        req.kv = SimpleNamespace(holds_kv=False)
        self._serving("0")
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_mode=ForwardMode.DECODE,
        )
        scheduler = SimpleNamespace(
            enable_overlap_mlx=False,
            result_queue=[(batch, None)],
            collect_inflight_reqs=lambda: {req},
            waiting_queue=[],
            chunked_req=None,
            hisparse_coordinator=None,
        )
        scheduler._num_queued_outputs = lambda: Scheduler._num_queued_outputs(scheduler)
        Scheduler.record_weight_version_change(scheduler, "1")
        self.assertEqual(req.weight_version_events[0].num_output_tokens, 3)

        Req.reset_for_retract(req)
        req.output_ids += [201, 202]  # prefilled again and decoded on the new version

        spans = compute_weight_version_spans(
            req.weight_version_events, "1", len(req.output_ids)
        )
        self.assertEqual(
            [(span.version, span.start, span.end) for span in spans],
            [("0", 0, 2), ("1", 2, 4)],
        )

    def test_a_queued_prebuilt_batch_commits_no_token(self):
        req = _OverlappedReq([900])
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_mode=ForwardMode.PREBUILT,
        )
        scheduler = SimpleNamespace(
            enable_overlap_mlx=False, result_queue=[(batch, None)]
        )
        self.assertEqual(Scheduler._num_queued_outputs(scheduler), {})

    def test_the_mlx_loop_queues_jobs_that_are_not_counted(self):
        scheduler = SimpleNamespace(enable_overlap_mlx=True, result_queue=[object()])
        self.assertEqual(Scheduler._num_queued_outputs(scheduler), {})

    def test_a_queued_output_that_is_dropped_leaves_the_span_at_what_was_kept(self):
        req = _OverlappedReq([101, 102])
        _, spans = self._run(req, forwards=1, tokens_per_result=0)
        self.assertEqual(spans, [("0", 0, 2)])


class TestSchedulerBatchWeightVersion(CustomTestCase):
    def test_embedding_rejects_enabled_prefill_tracking(self) -> None:
        """Embedding and reward models reject enabled prefill tracking before allocation."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.server_args = SimpleNamespace(enable_prefill_weight_versions=True)
        scheduler.model_config = SimpleNamespace(
            is_encoder_decoder=False, is_generation=False
        )

        with self.assertRaisesRegex(
            AssertionError, "does not support embedding or reward"
        ):
            scheduler.init_kv_weight_version_tracker()

    def test_embedding_without_prefill_tracking_remains_supported(self) -> None:
        """Embedding initialization remains unchanged when prefill tracking is disabled."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.server_args = SimpleNamespace(enable_prefill_weight_versions=False)
        scheduler.model_config = SimpleNamespace(
            is_encoder_decoder=False, is_generation=False
        )

        scheduler.init_kv_weight_version_tracker()

        self.assertIsNone(scheduler.kv_weight_version_tracker)

    def test_encoder_decoder_rejects_enabled_prefill_tracking(self) -> None:
        """Encoder-decoder models reject prefill tracking before allocating a tracker."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.server_args = SimpleNamespace(enable_prefill_weight_versions=True)
        scheduler.model_config = SimpleNamespace(is_encoder_decoder=True)

        with self.assertRaisesRegex(AssertionError, "does not support encoder-decoder"):
            scheduler.init_kv_weight_version_tracker()

    def test_encoder_decoder_without_prefill_tracking_remains_supported(self) -> None:
        """Disabled tracking leaves encoder-decoder initialization unchanged."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.server_args = SimpleNamespace(enable_prefill_weight_versions=False)
        scheduler.model_config = SimpleNamespace(is_encoder_decoder=True)

        scheduler.init_kv_weight_version_tracker()

        self.assertIsNone(scheduler.kv_weight_version_tracker)

    def _scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.publish_load_snapshot = MagicMock()
        scheduler.scheduler_stage_metrics = None
        scheduler.load_publisher = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.tree_cache = SimpleNamespace(flush_pending_backups=MagicMock())
        scheduler.kv_weight_version_tracker = KvWeightVersionTracker(
            num_slots=8,
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[3]])),
        )
        scheduler.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_overlap=False,
            enable_overlap_mlx=False,
            model_config=SimpleNamespace(think_end_ids=None),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=scheduler.tree_cache,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            kv_weight_version_tracker=scheduler.kv_weight_version_tracker,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=MagicMock(),
            draft_worker=None,
            model_worker=MagicMock(),
            logprob_result_processor=None,
            output_streamer=MagicMock(),
            beam_coordinator=MagicMock(),
            abort_request=lambda *args, **kwargs: None,
        )
        for name, value in (
            ("get_observability", SimpleNamespace(enable_metrics=False)),
            ("get_server_return_hidden_states_mode", CaptureHiddenMode.NULL),
        ):
            patcher = patch(
                f"sglang.srt.managers.scheduler_components.batch_result_processor.{name}",
                return_value=value,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler._record_step_counters = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()
        return scheduler

    def _batch(self) -> ScheduleBatch:
        return ScheduleBatch(
            reqs=[],
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.tensor([3], dtype=torch.int64),
        )

    def test_pending_result_uses_version_captured_when_batch_ran(self) -> None:
        """A pending result keeps the forward-time version across a serving update."""
        serving = _ServingStub("v0")
        scheduler = self._scheduler()
        batch = self._batch()

        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = False
        runner.server_args = SimpleNamespace(enable_prefill_weight_versions=True)
        with patch(
            "sglang.srt.mem_cache.kv_weight_version_tracker.get_serving",
            return_value=serving,
        ):
            result = GenerationBatchResult(
                next_token_ids=torch.tensor([], dtype=torch.int64),
                kv_weight_version_record=KvWeightVersionRecord.maybe_capture(
                    model_runner=runner, forward_batch=batch
                ),
            )
            queued_batch = batch.copy()
            serving.weight_version = "v1"
            Scheduler.process_batch_result(scheduler, queued_batch, result)

        spans = scheduler.kv_weight_version_tracker._lookup_spans(
            queued_batch.out_cache_loc
        )
        self.assertEqual(
            [(span.version, span.start, span.end) for span in spans], [("v0", 0, 1)]
        )

    def test_spec_decode_does_not_restamp_restored_prefill_slots(self) -> None:
        """Restoring the batch after speculative forward must not relabel old prompt KV."""
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                scheduler = self._scheduler()
                scheduler.record_batch_in_overlap = lambda batch: None
                batch = self._batch()
                batch.forward_mode = ForwardMode.DECODE
                batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
                tracker = scheduler.kv_weight_version_tracker
                tracker.record(slot_indices=batch.out_cache_loc, version="v0")

                with scheduler._forward_isolation(batch, overlap=overlap):
                    forward_slots = torch.tensor([4, 5, 6])
                    batch.out_cache_loc = forward_slots
                    result = GenerationBatchResult(
                        next_token_ids=torch.tensor([], dtype=torch.int64),
                        accept_lens=torch.tensor([], dtype=torch.int32),
                        speculative_num_draft_tokens=3,
                        kv_weight_version_record=KvWeightVersionRecord.capture(
                            slot_indices=forward_slots, version="v1"
                        ),
                    )
                    forward_slots.fill_(7)
                scheduler.process_batch_result(batch.copy(), result)

                spans = tracker._lookup_spans(torch.tensor([3, 4, 5, 6]))
                self.assertEqual(
                    [(span.version, span.start, span.end) for span in spans],
                    [("v0", 0, 1), ("v1", 1, 4)],
                )

    def test_result_copy_finishes_before_slot_indices_are_consumed(self) -> None:
        """Pending slot copies complete before the tracker reads their contents."""
        scheduler = self._scheduler()
        batch = self._batch()
        record = KvWeightVersionRecord.capture(
            slot_indices=torch.tensor([0]), version="v0"
        )
        synchronize = MagicMock(side_effect=lambda: record.slot_indices.fill_(3))
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([], dtype=torch.int64),
            kv_weight_version_record=record,
            copy_done=SimpleNamespace(synchronize=synchronize),
        )

        scheduler.process_batch_result(batch, result)

        spans = scheduler.kv_weight_version_tracker._lookup_spans(torch.tensor([3]))
        self.assertEqual([span.version for span in spans], ["v0"])
        self.assertIsNone(result.kv_weight_version_record)
        synchronize.assert_called_once_with()


def _runner(result=(True, "ok")):
    runner = Mock()
    for method in (
        "update_weights_from_disk",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
        "load_weights_from_distributed",
    ):
        getattr(runner.weight_updater, method).return_value = result
    runner.weight_updater.receive_weights_from_distributed.return_value = [
        ("weight", torch.tensor([1.0]))
    ]
    # peers write these runners' live weights
    runner.weight_updater.has_weight_staging.return_value = False
    return runner


def _request(**fields):
    return SimpleNamespace(
        **{
            "weight_version": "v2",
            "flush_cache": True,
            "torch_empty_cache": False,
            "model_path": "m",
            "recapture_cuda_graph": False,
            "load_format": None,
            "selector": "all",
            "names": ["model.layers.0.weight"],
            "dtypes": ["float32"],
            "shapes": [[1]],
            "group_name": "g",
            "serialized_named_tensors": [b""],
            **fields,
        }
    )


class _WeightUpdaterManagerTestBase(CustomTestCase):
    def setUp(self):
        patcher = patch("torch.distributed.barrier")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.recorded = []

    def _build_manager(self, target, draft):
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=target,
                weight_update_runners=lambda: [("target", target)],
                deserialize_own_rank=lambda payloads: [],
            ),
            draft_worker=(
                None
                if draft is None
                else SimpleNamespace(weight_update_runners=lambda: [("draft", draft)])
            ),
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )

    def _manager(self, target, draft=None, *, session=True):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = self._build_manager(target, draft)
        if session:
            # update_weights_from_* require an open session
            manager._session = _WeightUpdateSession(selector="all")
        return manager


class TestRecordWeightVersionAfterUpdate(_WeightUpdaterManagerTestBase):
    def test_successful_update_records_the_version(self):
        """A refit that reports success advances the scheduler-side version."""
        output = self._manager(_runner(), session=False).update_weights_from_disk(
            _request()
        )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_update_does_not_record_the_version(self):
        """A refit that fails must leave the version alone, or later tokens are mislabelled."""
        output = self._manager(
            _runner((False, "boom")), session=False
        ).update_weights_from_disk(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_draft_failure_does_not_record_the_version(self):
        """The target succeeding is not enough: a failed draft refit leaves the engine mixed."""
        manager = self._manager(
            _runner(), draft=_runner((False, "draft boom")), session=False
        )

        output = manager.update_weights_from_disk(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_distributed_update_records_the_version(self):
        """The distributed refit is the path an RL trainer actually drives, so it must record too."""
        manager = self._manager(_runner())
        output = manager.update_weights_from_distributed(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, [])
        output = manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_distributed_update_does_not_record_the_version(self):
        """A failed distributed refit leaves the version alone, exactly like the disk path."""
        output = self._manager(
            _runner((False, "boom"))
        ).update_weights_from_distributed(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_tensor_update_records_the_version(self):
        """The tensor refit records the version once the load reports success."""
        updater = self._manager(_runner())
        output = updater.update_weights_from_tensor(_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, [])
        with patch("torch.distributed.barrier"):
            output = updater.end_weight_update(EndWeightUpdateReqInput())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_successful_ipc_update_records_the_version(self):
        """The checkpoint-engine IPC refit records the version like every other path."""
        output = self._manager(_runner(), session=False).update_weights_from_ipc(
            _request()
        )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_ipc_update_does_not_record_the_version(self):
        """The IPC path branches on success separately from the cache flush, so failure must record nothing."""
        output = self._manager(
            _runner((False, "boom")), session=False
        ).update_weights_from_ipc(_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])


class TestWeightUpdateSession(_WeightUpdaterManagerTestBase):
    def test_distributed_update_receives_once_on_target_loads_into_each(self):
        """A draft runner that received its own broadcast would deadlock the update group."""
        target, draft = _runner(), _runner()
        weights = target.weight_updater.receive_weights_from_distributed.return_value
        req = _request()

        output = self._manager(target, draft).update_weights_from_distributed(req)

        self.assertTrue(output.success)
        target.weight_updater.receive_weights_from_distributed.assert_called_once_with(
            names=req.names,
            dtypes=req.dtypes,
            shapes=req.shapes,
            group_name=req.group_name,
            load_format=req.load_format,
        )
        target.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.receive_weights_from_distributed.assert_not_called()

    def test_distributed_update_target_only_selector_skips_draft(self):
        """selector="target" must not load into the draft."""
        target, draft = _runner(), _runner()

        output = self._manager(target, draft).update_weights_from_distributed(
            _request(selector="target")
        )

        self.assertTrue(output.success)
        target.weight_updater.load_weights_from_distributed.assert_called_once()
        draft.weight_updater.load_weights_from_distributed.assert_not_called()

    def test_end_runs_post_load_on_both_when_load_was_bypassed(self):
        """A P2P/RDMA session never calls load_weights, so end must run post_load_weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)

        output = manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertTrue(output.success)
        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=True
            )
        self.assertIsNone(manager._session)

    def test_end_skips_post_load_on_both_when_weights_loaded(self):
        """load_weights already ran post_load_weights; running it twice would double-apply."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft)
        manager._session = _WeightUpdateSession(selector="all", loaded_weights=True)

        manager.end_weight_update(EndWeightUpdateReqInput())

        for runner in (target, draft):
            runner.weight_updater.end_weight_update.assert_called_once_with(
                run_post_load=False
            )

    def test_session_selector_confines_begin_and_end_to_selected_runners(self):
        """end finalizing a runner begin never restored would repack unrestored weights."""
        target, draft = _runner(), _runner()
        manager = self._manager(target, draft, session=False)

        manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
        manager.end_weight_update(EndWeightUpdateReqInput())

        target.weight_updater.begin_weight_update.assert_not_called()
        target.weight_updater.end_weight_update.assert_not_called()
        draft.weight_updater.begin_weight_update.assert_called_once_with(
            use_staging=False
        )
        draft.weight_updater.end_weight_update.assert_called_once()

    def test_begin_rejects_reentry(self):
        """A second begin would leave the first session's runners unfinalized."""
        target = _runner()

        output = self._manager(target).begin_weight_update(BeginWeightUpdateReqInput())

        self.assertFalse(output.success)
        self.assertIn("already open", output.message)
        target.weight_updater.begin_weight_update.assert_not_called()

    def test_end_without_session_is_rejected(self):
        """Finalizing runners begin never restored would repack weights twice."""
        target = _runner()

        output = self._manager(target, session=False).end_weight_update(
            EndWeightUpdateReqInput()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.end_weight_update.assert_not_called()

    def test_update_without_session_is_rejected_without_loading(self):
        """A caller that skips begin gets an error back instead of crashing the scheduler."""
        target = _runner()

        output = self._manager(target, session=False).update_weights_from_distributed(
            _request()
        )

        self.assertFalse(output.success)
        self.assertIn("begin_weight_update", output.message)
        target.weight_updater.receive_weights_from_distributed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
