"""Ownership of the overlap scheduler's batch snapshot.

Under the overlap scheduler a ``ScheduleBatch``'s device tensors are read by
kernels that are still queued when ``run_batch`` returns. The scheduler parks a
snapshot of the batch on the batch's result, and the result carries it until
``process_batch_result`` has run.
"""

import contextlib
import dataclasses
import unittest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.utils import (  # noqa: E402
    EmbeddingBatchResult,
    GenerationBatchResult,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


@dataclasses.dataclass
class _FakeBatch:
    """A batch carrying real ``ScheduleBatch`` field names.

    ``_snapshot_batch_for_overlap`` walks ``dataclasses.fields``, so the names
    have to be real ones or the test would cover a field set that does not
    exist. ``test_field_names_are_real`` enforces that.
    """

    req_pool_indices: Optional[torch.Tensor] = None
    out_cache_loc: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    # Slot ids for the deferred mamba copy-on-write and clear. These are read by
    # kernels queued during the forward, and ``filter_batch`` unbinds them from
    # the batch on the next scheduler iteration, so the snapshot is their only
    # remaining reference.
    mamba_cow_src_indices: Optional[torch.Tensor] = None
    mamba_cow_dst_indices: Optional[torch.Tensor] = None
    mamba_clear_indices: Optional[torch.Tensor] = None
    # Read by _forward_isolation and run_batch.
    spec_algorithm: Any = None
    sampling_info: Any = None
    forward_mode: Any = None
    input_ids: Any = None
    reqs: Any = dataclasses.field(default_factory=list)
    return_logprob: bool = False
    return_hidden_states: bool = False
    forward_iter: Any = None
    launch_ts: Any = None
    after_idle_gap: Any = None


# Sentinel for the forward-only sampling_info copy; see _new_batch.
_FORWARD_SAMPLING_INFO = object()


def _new_batch() -> _FakeBatch:
    return _FakeBatch(
        req_pool_indices=torch.zeros(2, dtype=torch.int64),
        out_cache_loc=torch.zeros(2, dtype=torch.int64),
        seq_lens=torch.zeros(2, dtype=torch.int64),
        mamba_cow_src_indices=torch.zeros(1, dtype=torch.int64),
        mamba_cow_dst_indices=torch.zeros(1, dtype=torch.int64),
        mamba_clear_indices=torch.zeros(1, dtype=torch.int64),
        spec_algorithm=MagicMock(**{"is_none.return_value": True}),
        forward_mode=MagicMock(**{"is_prebuilt.return_value": False}),
        # _forward_isolation swaps in copy_for_forward() before snapshotting, so
        # the snapshot has to hold the copy, not this.
        sampling_info=MagicMock(
            **{"copy_for_forward.return_value": _FORWARD_SAMPLING_INFO}
        ),
    )


class TestSchedulerOverlapKeepAlive(CustomTestCase):
    def test_snapshot_holds_the_batch_and_each_schedule_batch_field(self):
        """Exhaustive over the real field set, so a new field cannot be missed."""
        scheduler = Scheduler.__new__(Scheduler)
        batch = ScheduleBatch.__new__(ScheduleBatch)
        fields = dataclasses.fields(ScheduleBatch)
        for field in fields:
            setattr(batch, field.name, object())

        refs = scheduler._snapshot_batch_for_overlap(batch)

        expected = [batch] + [getattr(batch, f.name) for f in fields]
        self.assertEqual(len(refs), len(expected))
        for got, want in zip(refs, expected):
            self.assertIs(got, want)

    def test_forward_isolation_yields_the_snapshot_only_in_overlap_mode(self):
        """The production wiring: run_batch parks what this yields."""
        scheduler = Scheduler.__new__(Scheduler)
        batch = _new_batch()

        with scheduler._forward_isolation(batch, overlap=True) as refs:
            self.assertIsNotNone(refs)
            self.assertTrue(any(item is batch.mamba_cow_src_indices for item in refs))
            # Taken after the sampling_info swap, so it holds the forward-only
            # copy. A snapshot taken before the swap would hold the original.
            self.assertTrue(any(item is _FORWARD_SAMPLING_INFO for item in refs))

        with scheduler._forward_isolation(batch, overlap=False) as refs:
            self.assertIsNone(refs)

    def test_snapshot_survives_the_batch_unbinding_its_fields(self):
        """``filter_batch`` clears the mamba slot ids on the next iteration."""
        scheduler = Scheduler.__new__(Scheduler)
        batch = _new_batch()
        cow_src = batch.mamba_cow_src_indices

        refs = scheduler._snapshot_batch_for_overlap(batch)
        batch.mamba_cow_src_indices = None
        batch.mamba_clear_indices = None

        self.assertTrue(any(item is cow_src for item in refs))


class TestProcessBatchResultReleasesTheSnapshot(CustomTestCase):
    """The release runs in ``Scheduler.process_batch_result`` itself."""

    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.batch_result_processor = MagicMock()
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.metrics_reporter = MagicMock()
        scheduler.publish_load_snapshot = MagicMock()
        scheduler.maybe_send_health_check_signal = MagicMock()
        scheduler._record_step_counters = MagicMock()
        scheduler._maybe_clear_mm_inputs = MagicMock()
        scheduler.enable_fpm = False
        return scheduler

    @staticmethod
    def _prebuilt_batch() -> MagicMock:
        batch = MagicMock()
        batch.reqs = []
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = False
        batch.forward_mode.is_prebuilt.return_value = True
        return batch

    @staticmethod
    def _idle_batch() -> MagicMock:
        batch = MagicMock()
        batch.reqs = []
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = False
        batch.forward_mode.is_prebuilt.return_value = False
        batch.forward_mode.is_idle.return_value = True
        return batch

    def test_release_runs_after_the_processor(self):
        """The refs are still attached while the processor runs, and gone after."""
        scheduler = self._new_scheduler()
        result = GenerationBatchResult(
            overlap_keep_alive_refs=scheduler._snapshot_batch_for_overlap(_new_batch())
        )
        seen = []
        scheduler.batch_result_processor.process_batch_result_idle.side_effect = (
            lambda _batch, res: seen.append(res.overlap_keep_alive_refs is not None)
        )

        scheduler.process_batch_result(self._idle_batch(), result)

        self.assertEqual(seen, [True])
        self.assertIsNone(result.overlap_keep_alive_refs)

    def test_generation_result_drops_both_keep_alive_sets(self):
        scheduler = self._new_scheduler()
        batch = self._idle_batch()
        snapshot = scheduler._snapshot_batch_for_overlap(_new_batch())
        # The worker's refs ride on the same result and are released with it;
        # this is the spec V2 verify ForwardBatch in production.
        result = GenerationBatchResult(
            overlap_keep_alive_refs=snapshot,
            extra_keep_alive_refs=[object()],
        )

        scheduler.process_batch_result(batch, result)

        scheduler.batch_result_processor.process_batch_result_idle.assert_called_once()
        self.assertIsNone(result.overlap_keep_alive_refs)
        self.assertIsNone(result.extra_keep_alive_refs)

    def test_embedding_result_keeps_only_its_own_field(self):
        """EmbeddingBatchResult does not declare extra_keep_alive_refs, so the
        release must not create one on it."""
        scheduler = self._new_scheduler()
        batch = MagicMock()
        batch.reqs = []
        batch.is_dllm.return_value = False
        batch.forward_mode.is_decode.return_value = False
        batch.forward_mode.is_extend.return_value = True
        result = EmbeddingBatchResult(
            embeddings=[],
            copy_done=MagicMock(),
            overlap_keep_alive_refs=scheduler._snapshot_batch_for_overlap(_new_batch()),
        )

        scheduler.process_batch_result(batch, result)

        # The release carries its own barrier on this class too.
        result.copy_done.synchronize.assert_called_once()
        self.assertIsNone(result.overlap_keep_alive_refs)
        self.assertFalse(hasattr(result, "extra_keep_alive_refs"))

    def test_prebuilt_path_waits_and_releases(self):
        """process_batch_result_prebuilt takes no result, so the release's own
        wait is the only barrier on that branch."""
        scheduler = self._new_scheduler()
        order = []
        copy_done = MagicMock(
            **{"synchronize.side_effect": lambda: order.append("wait")}
        )
        result = GenerationBatchResult(
            copy_done=copy_done,
            overlap_keep_alive_refs=scheduler._snapshot_batch_for_overlap(_new_batch()),
        )
        scheduler.batch_result_processor.process_batch_result_prebuilt.side_effect = (
            lambda *_: order.append("process")
        )

        scheduler.process_batch_result(self._prebuilt_batch(), result)

        copy_done.synchronize.assert_called_once()
        self.assertEqual(order, ["process", "wait"])
        self.assertIsNone(result.overlap_keep_alive_refs)

    def test_a_raising_processor_still_releases(self):
        scheduler = self._new_scheduler()
        result = GenerationBatchResult(
            copy_done=MagicMock(),
            overlap_keep_alive_refs=scheduler._snapshot_batch_for_overlap(_new_batch()),
        )
        scheduler.batch_result_processor.process_batch_result_idle.side_effect = (
            RuntimeError("processor blew up")
        )

        with self.assertRaises(RuntimeError):
            scheduler.process_batch_result(self._idle_batch(), result)

        result.copy_done.synchronize.assert_called_once()
        self.assertIsNone(result.overlap_keep_alive_refs)

    def test_prebuilt_tolerates_a_result_without_copy_done(self):
        """_run_batch_prebuilt returns a result whose copy_done is None."""
        scheduler = self._new_scheduler()
        result = GenerationBatchResult(
            overlap_keep_alive_refs=scheduler._snapshot_batch_for_overlap(_new_batch())
        )

        scheduler.process_batch_result(self._prebuilt_batch(), result)

        self.assertIsNone(result.overlap_keep_alive_refs)


class TestEmbeddingCopyToCpu(CustomTestCase):
    def test_it_refuses_to_run_without_an_event_from_the_caller(self):
        result = EmbeddingBatchResult(embeddings=[])

        with self.assertRaises(RuntimeError):
            result.copy_to_cpu()

    def test_an_empty_embedding_batch_still_copies_pooled_hidden_states(self):
        """The old early return skipped this block along with the event."""
        pooled = torch.arange(4)
        result = EmbeddingBatchResult(
            embeddings=[], pooled_hidden_states=[pooled], copy_done=MagicMock()
        )

        result.copy_to_cpu()

        self.assertEqual(result.pooled_hidden_states[0].device.type, "cpu")
        self.assertTrue(torch.equal(result.pooled_hidden_states[0], torch.arange(4)))
        result.copy_done.record.assert_called_once()


class TestRunBatchAttachesTheSnapshot(CustomTestCase):
    """``run_batch`` must hand the snapshot to the result it returns."""

    def _new_scheduler(self, **overrides) -> Scheduler:
        # Mirrors what run_batch reads; expect to extend it when run_batch grows.
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.forward_ct = 0
        scheduler._sched_idled = False
        scheduler.scripted_scheduler_hook = None
        scheduler.profiler_manager = MagicMock()
        scheduler.forward_sleep_time = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.is_generation = True
        scheduler.enable_overlap = True
        scheduler.enable_unified_memory = False
        scheduler.future_map = MagicMock()
        scheduler._confidence_budget_prepare = None
        scheduler.forward_stream = MagicMock()
        scheduler.schedule_stream = MagicMock()
        # No real streams: the branch below enqueues nothing on a device.
        scheduler.forward_stream_ctx = contextlib.nullcontext()
        scheduler.device_module = MagicMock()
        scheduler._maybe_report_active_ranks = MagicMock()
        # delay_sample_func is not None, so run_batch takes the branch that
        # defers the device-to-host copy and touches no real streams.
        scheduler.model_worker = MagicMock(
            **{
                "forward_batch_generation.return_value": GenerationBatchResult(
                    delay_sample_func=lambda: None
                )
            }
        )
        for name, value in overrides.items():
            setattr(scheduler, name, value)
        return scheduler

    def test_the_returned_result_carries_the_batch_snapshot(self):
        scheduler = self._new_scheduler()
        batch = _new_batch()
        cow_src = batch.mamba_cow_src_indices

        with patch.object(scheduler_module, "resolve_forward_inputs"):
            result = scheduler.run_batch(batch)

        self.assertIsNotNone(
            result.overlap_keep_alive_refs,
            "run_batch returned a result that does not hold the batch snapshot",
        )
        refs = result.overlap_keep_alive_refs
        self.assertTrue(any(item is batch for item in refs))
        self.assertTrue(any(item is cow_src for item in refs))

    def test_the_embedding_path_parks_and_records_copy_done(self):
        pooler_output = MagicMock(embeddings=[], pooled_hidden_states=None)
        scheduler = self._new_scheduler(
            is_generation=False,
            tp_worker=MagicMock(
                **{"forward_batch_embedding.return_value": (pooler_output, False)}
            ),
        )
        batch = _new_batch()

        with patch.object(scheduler_module, "resolve_forward_inputs"):
            result = scheduler.run_batch(batch)

        self.assertIsNotNone(
            result.overlap_keep_alive_refs,
            "run_batch returned an embedding result without the batch snapshot",
        )
        self.assertTrue(any(item is batch for item in result.overlap_keep_alive_refs))
        # A batch that produced no embeddings still needs the event the result
        # processor waits on, and it has to be recorded, not just created.
        self.assertIsNotNone(result.copy_done)
        result.copy_done.record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
