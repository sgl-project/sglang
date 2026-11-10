import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    PluginWeightUpdateReqInput,
    PluginWeightUpdateReqOutput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.deferred_responses import (
    DeferredResponses,
)
from sglang.srt.managers.scheduler_components.request_list import (
    KeyedReqList,
    ListKind,
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
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ServingStub:
    def __init__(self, weight_version: str):
        self.weight_version = weight_version


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
        scheduler.deferred_responses = DeferredResponses(
            disaggregation_mode=DisaggregationMode.NULL,
            output_streamer=None,
            send_abort=None,
        )
        scheduler.token_to_kv_pool_allocator = None
        scheduler.req_to_token_pool = None
        scheduler.publish_load_snapshot = MagicMock()
        scheduler.scheduler_stage_metrics = None
        scheduler.load_publisher = MagicMock()
        scheduler.load_inquirer = MagicMock()
        scheduler.tree_cache = SimpleNamespace(flush_pending_backups=MagicMock())
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.prefill_launched_reqs = KeyedReqList(ListKind.PREFILL_LAUNCHED)
        scheduler.invariant_checker = MagicMock()
        scheduler.kv_weight_version_tracker = KvWeightVersionTracker(
            num_slots=8,
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[3]])),
        )
        scheduler.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=DisaggregationMode.NULL,
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
            request_retirer=release_only_retirer(tree_cache=scheduler.tree_cache),
            abort_request=lambda *args, **kwargs: None,
            still_in_batch=lambda req: True,
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
        batch = ScheduleBatch(
            reqs=[],
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.tensor([3], dtype=torch.int64),
        )
        # A launched step, as process_batch_result expects.
        batch.add_inflight_step()
        return batch

    def test_pending_result_uses_version_captured_when_batch_ran(self) -> None:
        """A pending result keeps the forward-time version across a serving update."""
        serving = _ServingStub("v0")
        scheduler = self._scheduler()
        batch = self._batch()
        forward_batch = SimpleNamespace(
            out_cache_loc=batch.out_cache_loc,
            out_cache_loc_virtual=None,
            origin_out_cache_loc=None,
        )

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
                    model_runner=runner, forward_batch=forward_batch
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
    runner.weight_updater.get_model.return_value = torch.nn.Module()
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
    runner.weight_updater.load_rejection.return_value = None
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
        self.assertEqual(self.recorded, ["v2"])
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
        self.assertEqual(self.recorded, ["v2"])
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
    def test_rejects_all_runners_before_restoring_any_derived_weights(self):
        target, draft = _runner(), _runner()
        draft.weight_updater.get_model.return_value._derived_weight_cache_error = (
            "CP decode attention TP requires restart"
        )
        manager = self._manager(target, draft, session=False)
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            result = manager.begin_weight_update(BeginWeightUpdateReqInput())
        self.assertFalse(result.success)
        self.assertIn("CP decode attention TP", result.message)
        self.assertIsNone(manager._session)
        target.weight_updater.begin_weight_update.assert_not_called()
        draft.weight_updater.begin_weight_update.assert_not_called()

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
        draft.weight_updater.begin_weight_update.assert_called_once_with()
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


class TestPluginWeightUpdate(_WeightUpdaterManagerTestBase):
    def setUp(self):
        super().setUp()
        gather = patch(
            "torch.distributed.all_gather_object",
            side_effect=lambda outputs, obj, group: outputs.__setitem__(0, obj),
        )
        gather.start()
        self.addCleanup(gather.stop)
        override = get_parallel().override(tp_rank=0, tp_size=1)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)
        self.flushes = []

    def _manager_serving(self, handlers, *, target=None, draft=None, session=True):
        manager = self._manager(target or _runner(), draft, session=session)
        manager.flush_cache = lambda **kwargs: self.flushes.append(kwargs) or True
        platform = patch(
            "sglang.srt.managers.scheduler_components.weight_updater.current_platform",
            SimpleNamespace(get_weight_update_handlers=lambda: handlers),
        )
        platform.start()
        self.addCleanup(platform.stop)
        return manager

    def test_received_tensors_load_into_every_selected_runner(self):
        """Like update_weights_from_distributed: the transport receives, the core loads target and draft."""
        weights = [("w", object())]
        target, draft = _runner(), _runner()
        manager = self._manager_serving(
            {"ici": lambda *, context, payload: weights}, target=target, draft=draft
        )

        output = manager.plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici", weight_version="3")
        )

        self.assertTrue(output.success)
        self.assertEqual(output.received_tensor_counts, [1])
        target.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        draft.weight_updater.load_weights_from_distributed.assert_called_once_with(
            weights
        )
        self.assertEqual(self.recorded, ["3"])
        self.assertEqual(len(self.flushes), 1)

    def test_end_runs_post_load_only_after_writes_that_bypassed_load_weights(self):
        """A transport that writes parameters itself returns no tensors, like p2p peer writes."""
        for received, run_post_load in (([("w", object())], False), ([], True)):
            with self.subTest(received=received):
                target = _runner()
                manager = self._manager_serving(
                    {"ici": lambda *, context, payload: received}, target=target
                )
                manager.plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))
                manager.end_weight_update(EndWeightUpdateReqInput())

                target.weight_updater.end_weight_update.assert_called_once_with(
                    run_post_load=run_post_load
                )

    def test_a_failed_receive_loads_nothing_on_its_rank_and_keeps_the_cache_and_version(
        self,
    ):
        """A rank that failed to receive loads nothing; the request fails and the old KV and version stay."""

        def broken(*, context, payload):
            raise ValueError("no receive buffers")

        target = _runner()
        output = self._manager_serving(
            {"ici": broken}, target=target
        ).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici", weight_version="3")
        )

        self.assertFalse(output.success)
        self.assertIn("no receive buffers", output.message)
        target.weight_updater.load_weights_from_distributed.assert_not_called()
        self.assertEqual((self.flushes, self.recorded), ([], []))

    def test_rank_0_reports_another_ranks_failure_from_the_gather(self):
        """Only rank 0 replies: a failure gathered from rank 1 fails the request after rank 0 loaded."""
        gathered = [(1, None), (0, "TP rank 1: no receive buffers")]
        target = _runner()
        manager = self._manager_serving(
            {"ici": lambda *, context, payload: [("w", object())]}, target=target
        )

        with (
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda outputs, obj, group: outputs.__setitem__(
                    slice(None), gathered
                ),
            ),
            get_parallel().override(tp_rank=0, tp_size=2),
        ):
            output = manager.plugin_weight_update(
                PluginWeightUpdateReqInput(handler="ici", weight_version="3")
            )

        self.assertFalse(output.success)
        self.assertIn("TP rank 1", output.message)
        self.assertIn("other TP ranks have loaded", output.message)
        self.assertEqual(output.received_tensor_counts, [])
        target.weight_updater.load_weights_from_distributed.assert_called_once()
        self.assertEqual((self.flushes, self.recorded), ([], []))

    def test_a_runner_that_rejects_loads_fails_the_request_before_the_transport_runs(
        self,
    ):
        """The weight-cache and derived-weight guards cover direct writes too, so they run first."""
        called = []
        target = _runner()
        target.weight_updater.load_rejection.return_value = (
            "[weight_cache] plugin_weight_update is not supported"
        )

        output = self._manager_serving(
            {"ici": lambda *, context, payload: called.append(1) or []}, target=target
        ).plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))

        self.assertFalse(output.success)
        self.assertIn("weight_cache", output.message)
        self.assertEqual(called, [])

    def test_transport_state_persists_across_calls(self):
        """A transport that registers buffers in one call must find them in the next."""

        def receive(*, context, payload):
            context.state.setdefault("calls", 0)
            context.state["calls"] += 1
            return []

        manager = self._manager_serving({"ici": receive})
        for _ in range(2):
            manager.plugin_weight_update(PluginWeightUpdateReqInput(handler="ici"))

        self.assertEqual(manager._plugin_state["calls"], 2)

    def test_requests_outside_a_session_or_for_unknown_handlers_are_rejected(self):
        """A caller that skips begin, or names a missing transport, gets an error, not a crash."""
        called = []
        handlers = {"ici": lambda *, context, payload: called.append(1) or []}

        outside = self._manager_serving(handlers, session=False).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="ici")
        )
        unknown = self._manager_serving(handlers).plugin_weight_update(
            PluginWeightUpdateReqInput(handler="nccl")
        )

        self.assertFalse(outside.success)
        self.assertIn("begin_weight_update", outside.message)
        self.assertFalse(unknown.success)
        self.assertIn("nccl", unknown.message)
        self.assertEqual(called, [])


class TestPluginWeightUpdateReply(CustomTestCase):
    def test_attention_dp_leaders_replies_merge_like_the_distributed_route(self):
        """Under attention DP the tokenizer gets one reply per leader; one failure fails the request."""
        ok = PluginWeightUpdateReqOutput(
            success=True, message="Success", received_tensor_counts=[3, 3]
        )
        failed = PluginWeightUpdateReqOutput(
            success=False, message="TP rank 1: no receive buffers"
        )

        merged_ok = PluginWeightUpdateReqOutput.merged([ok, ok])
        merged_failed = PluginWeightUpdateReqOutput.merged([ok, failed])

        self.assertEqual(
            (merged_ok.success, merged_ok.received_tensor_counts), (True, [3, 3])
        )
        self.assertFalse(merged_failed.success)
        self.assertIn("no receive buffers", merged_failed.message)
        self.assertEqual(merged_failed.received_tensor_counts, [])


if __name__ == "__main__":
    unittest.main()
