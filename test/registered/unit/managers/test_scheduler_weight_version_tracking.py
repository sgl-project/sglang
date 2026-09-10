import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_weight_version_tracker import (
    KvWeightVersionRecord,
    KvWeightVersionTracker,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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

    def test_a_new_version_is_adopted(self):
        """The scheduler has to end up on the version it was told about, or nothing downstream can read it."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v2")

        self.assertEqual(serving.weight_version, "v2")

    def test_same_version_is_a_noop(self):
        """Re-announcing the current version must not be treated as a change."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version="v1")

        self.assertEqual(serving.weight_version, "v1")

    def test_none_version_is_a_noop(self):
        """An update that carries no version must leave the recorded one alone."""
        serving = self._serving("v1")

        Scheduler.record_weight_version_change(self._scheduler(), new_version=None)

        self.assertEqual(serving.weight_version, "v1")

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
            server_args=SimpleNamespace(),
            model_config=SimpleNamespace(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=None,
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


class TestRecordWeightVersionAfterUpdate(CustomTestCase):
    def _updater(
        self, target_result, draft_result=None, method="update_weights_from_disk"
    ):
        self.recorded = []
        return SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(**{method: lambda recv_req: target_result}),
            draft_worker=(
                None
                if draft_result is None
                else SimpleNamespace(**{method: lambda recv_req: draft_result})
            ),
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )

    def _request(self, **fields):
        return SimpleNamespace(
            weight_version="v2",
            flush_cache=True,
            torch_empty_cache=False,
            **fields,
        )

    def test_successful_update_records_the_version(self):
        """A refit that reports success advances the scheduler-side version."""
        updater = self._updater(target_result=(True, "ok"))

        output = updater.update_weights_from_disk(self._request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_update_does_not_record_the_version(self):
        """A refit that fails must leave the version alone, or later tokens are mislabelled."""
        updater = self._updater(target_result=(False, "boom"))

        output = updater.update_weights_from_disk(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_draft_failure_does_not_record_the_version(self):
        """The target succeeding is not enough: a failed draft refit leaves the engine mixed."""
        updater = self._updater(
            target_result=(True, "ok"), draft_result=(False, "draft boom")
        )

        output = updater.update_weights_from_disk(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def _runner_updater(self, target_result, receive=lambda *args: {}):
        self.recorded = []
        runner = SimpleNamespace(
            begin_weight_update=lambda: None,
            end_weight_update=lambda **kwargs: None,
            weight_updater=SimpleNamespace(
                receive_weights_from_distributed=receive,
                load_weights=lambda weights: None,
                update_weights_from_tensor=lambda **kwargs: target_result,
            ),
        )
        updater = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=runner,
                ps=SimpleNamespace(tp_rank=0),
                iter_runners=lambda: [("", runner)],
            ),
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(
                record_weight_version_change=lambda new_version: self.recorded.append(
                    new_version
                )
            ),
        )
        with patch("torch.distributed.barrier"):
            updater.begin_weight_update(
                SimpleNamespace(selector="target", sync_base=True)
            )
        return updater

    def _distributed_request(self):
        return self._request(
            names=[],
            dtypes=[],
            shapes=[],
            group_name="g",
            load_format=None,
            selector="target",
        )

    def test_successful_distributed_update_records_the_version_at_session_end(
        self,
    ) -> None:
        """A distributed refit publishes its version only after the session finishes."""
        updater = self._runner_updater(target_result=(True, "ok"))

        output = updater.update_weights_from_distributed(self._distributed_request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, [])
        with patch("torch.distributed.barrier"):
            output = updater.end_weight_update(
                SimpleNamespace(expected_lora_checksums=None)
            )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_distributed_update_does_not_record_the_version(self):
        """A failed distributed refit leaves the version alone, exactly like the disk path."""

        def boom(*args):
            raise RuntimeError("boom")

        updater = self._runner_updater(target_result=(False, "boom"), receive=boom)

        output = updater.update_weights_from_distributed(self._distributed_request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])

    def test_successful_tensor_update_records_the_version_at_session_end(
        self,
    ) -> None:
        """A tensor refit publishes its version only after the session finishes."""
        updater = self._runner_updater(target_result=(True, "ok"))

        with patch("torch.distributed.barrier"), patch(
            "sglang.srt.managers.scheduler_components.weight_updater."
            "MultiprocessingSerializer.deserialize",
            return_value=[],
        ):
            output = updater.update_weights_from_tensor(
                self._request(
                    serialized_named_tensors=[b""], selector="target", load_format=None
                )
            )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, [])
        with patch("torch.distributed.barrier"):
            output = updater.end_weight_update(
                SimpleNamespace(expected_lora_checksums=None)
            )

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_successful_ipc_update_records_the_version(self):
        """The checkpoint-engine IPC refit records the version like every other path."""
        updater = self._updater(
            target_result=(True, "ok"), method="update_weights_from_ipc"
        )

        with patch("torch.distributed.barrier"):
            output = updater.update_weights_from_ipc(self._request())

        self.assertTrue(output.success)
        self.assertEqual(self.recorded, ["v2"])

    def test_failed_ipc_update_does_not_record_the_version(self):
        """The IPC path branches on success separately from the cache flush, so failure must record nothing."""
        updater = self._updater(
            target_result=(False, "boom"), method="update_weights_from_ipc"
        )

        with patch("torch.distributed.barrier"):
            output = updater.update_weights_from_ipc(self._request())

        self.assertFalse(output.success)
        self.assertEqual(self.recorded, [])


if __name__ == "__main__":
    unittest.main()
