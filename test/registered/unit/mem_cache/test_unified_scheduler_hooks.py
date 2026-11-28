"""How the scheduler drives the unified pool: the forwards it records, so that
no page move races them, and the opportunistic compactions it runs between
steps.

The scheduler is a bare instance with only the state each path reads; the pool
is a real chain. Events are host-controlled stand-ins and stream waits are
recorded instead of issued, so these run on the CPU.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test_unified_page_writability import _Bands, _chain  # noqa: E402

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.utils import EmbeddingBatchResult
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    """A CUDA event stand-in whose completion the test controls."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired

    def record(self, stream=None):
        pass


class _Device:
    Event = _Event

    @staticmethod
    def stream(stream):
        return contextlib.nullcontext()


class _Storage:
    """The storage's forward record, over a bare chain."""

    record_forward = KVStorage.record_forward
    retire_forward = KVStorage.retire_forward

    def __init__(self, allocator):
        self._allocator = allocator
        # The rows' working state slots: none in these batches.
        self.requests = SimpleNamespace(
            get_mamba_indices=lambda rows: torch.empty(0, dtype=torch.int64)
        )


def _scheduler(allocator) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device_module = _Device
    scheduler.schedule_stream = None
    scheduler.req_to_token_pool = None
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(storage=_Storage(allocator))
    return scheduler


class TestSpeculativeForwardRecords(_Bands):
    def test_a_compaction_waits_the_speculative_forward_of_a_last_stage(self):
        """On the last pipeline stage the speculative worker writes draft and
        verify slots, reserved before the step, that ``out_cache_loc`` does
        not name: a speculative forward's token writes are recorded as unknown
        on every stage, so an urgent compaction waits it."""
        alloc = _chain()
        band = alloc.full_attn_allocator
        freed = band.alloc(8)
        drafts = band.alloc(8)  # the draft slots the forward writes
        band.free(freed)  # holes below them: a compaction moves them
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=None,
        )
        forward = Scheduler._record_unified_forward(
            _scheduler(alloc),
            batch=batch,
            stream=None,
            written_tokens=batch.out_cache_loc,
        )
        self.waits.keys.clear()
        before = band.ids.virtual_to_physical[drafts].clone()
        self.assertGreater(band.flush_for_allocation(), 0)
        self.assertFalse(torch.equal(before, band.ids.virtual_to_physical[drafts]))
        self.assertIn(forward, self.waits.keys, "a written page moved unwaited")


class _Stop(Exception):
    pass


class TestEmbeddingForwardRecords(_Bands):
    """Under overlap an embedding or reward forward runs on the forward stream
    while the next batch is scheduled: it is recorded like a generation
    forward, so compaction waits it and rows freed under it stay pending until
    its result is processed."""

    def setUp(self):
        super().setUp()
        self.alloc = _chain()
        self.band = self.alloc.full_attn_allocator
        freed = self.band.alloc(8)
        self.written = self.band.alloc(8)
        self.band.free(freed)  # holes below the written rows
        scheduler = _scheduler(self.alloc)
        scheduler.scheduler_stage_metrics = None
        scheduler._on_step_launch = mock.Mock()
        scheduler.forward_sleep_time = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.is_generation = False
        scheduler.enable_overlap = True
        scheduler.enable_unified_memory = True
        scheduler.enable_unified_cache_external_linker = False
        scheduler.record_batch_in_overlap = mock.Mock()
        scheduler.forward_stream_ctx = contextlib.nullcontext()
        scheduler.forward_stream = mock.Mock()
        scheduler.future_map = None
        scheduler.resolve_forward_inputs = mock.Mock()
        scheduler.tp_worker = SimpleNamespace(
            forward_batch_embedding=lambda batch: (
                SimpleNamespace(embeddings=torch.zeros(1), pooled_hidden_states=None),
                False,
            )
        )
        scheduler._maybe_report_active_ranks = mock.Mock()
        self.scheduler = scheduler
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_prebuilt=lambda: False, is_extend=lambda: True
            ),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            out_cache_loc=self.written,
            mamba_track_indices=None,
            mamba_cow_dst_indices=None,
            reqs=[],
        )

    def _run(self):
        with mock.patch.object(EmbeddingBatchResult, "copy_to_cpu"):
            result = Scheduler.run_batch(self.scheduler, self.batch)
        forward = getattr(result, "unified_forward_done", None)
        self.assertIsNotNone(forward, "the forward was not recorded")
        return result, forward

    def test_a_compaction_waits_an_overlapped_embedding_forward(self):
        _, forward = self._run()
        self.waits.keys.clear()
        before = self.band.ids.virtual_to_physical[self.written].clone()
        self.assertGreater(self.band.flush_for_allocation(), 0)
        self.assertFalse(
            torch.equal(before, self.band.ids.virtual_to_physical[self.written])
        )
        self.assertIn(forward, self.waits.keys, "a written page moved unwaited")

    def test_the_processed_step_retires_the_forward_not_processing_entry(self):
        """Result processing starts before the processor synchronizes the
        step's completion: the forward's records and the pages freed under it
        stay protected until the processed hook, after that synchronize."""
        steps = self.alloc.forward_steps
        self.scheduler.token_to_kv_pool_allocator.forward_steps = steps
        self.scheduler._on_step_launch = lambda batch: steps.forward_launched()
        other = self.band.alloc(8)
        result, forward = self._run()
        self.band.free(other)
        self.assertTrue(self.band._pending_until_retire)
        self.scheduler.deferred_responses = mock.Mock()
        self.scheduler.publish_load_snapshot = mock.Mock(side_effect=_Stop)
        with self.assertRaises(_Stop):
            Scheduler.process_batch_result(self.scheduler, self.batch, result)
        self.assertFalse(self.band.usage.forward_retired(forward))
        self.assertTrue(self.band._pending_until_retire)
        self.scheduler.invariant_checker = mock.Mock()
        self.scheduler.running_reqs = mock.Mock()
        self.scheduler.prefill_launched_reqs = mock.Mock()
        self.batch.drop_inflight_step = lambda: None
        Scheduler._on_step_processed(self.scheduler, self.batch, result)
        self.assertTrue(self.band.usage.forward_retired(forward))
        self.assertFalse(self.band._pending_until_retire)


class TestOpportunisticCompactionErrors(CustomTestCase):
    """An error of an opportunistic compaction, between steps or when idle,
    propagates: a move that stopped midway leaves page tables and the pool's
    ledger inconsistent, and the next step must not run on them."""

    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.enable_unified_memory = True
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            flush_opportunistic=mock.Mock(side_effect=RuntimeError("a move failed"))
        )
        return scheduler

    def test_an_error_after_processing_a_result_early_propagates(self):
        scheduler = self._scheduler()
        scheduler.gracefully_exit = False
        scheduler.increment_scheduler_iterations = mock.Mock()
        scheduler.ingest_requests = mock.Mock()
        scheduler.pause_state = PauseState()
        scheduler.last_batch = None
        scheduler.running_batch = None
        scheduler.process_batch_result = mock.Mock()

        def early(last_batch):
            scheduler.result_queue.append((None, None))
            return True

        scheduler.should_process_last_batch_early = early
        # Reached only when the error was swallowed.
        scheduler.get_next_batch_to_run = mock.Mock(side_effect=_Stop)
        with self.assertRaisesRegex(RuntimeError, "a move failed"):
            Scheduler.event_loop_overlap(scheduler)
        scheduler.process_batch_result.assert_called_once()

    def test_an_error_when_idle_propagates(self):
        scheduler = self._scheduler()
        scheduler.dp_attn_adapter = mock.Mock()
        scheduler.deferred_responses = mock.Mock()
        scheduler.maybe_send_health_check_signal = mock.Mock()
        scheduler.is_fully_idle = mock.Mock(return_value=True)
        scheduler.metrics_reporter = mock.Mock()
        # The rest of the idle path has no state here: a swallowed error
        # fails on the first thing it reads.
        with self.assertRaisesRegex(RuntimeError, "a move failed"):
            Scheduler.on_idle(scheduler)


if __name__ == "__main__":
    unittest.main()
