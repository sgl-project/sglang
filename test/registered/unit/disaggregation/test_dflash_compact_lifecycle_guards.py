"""CPU regressions for decode transfer ownership across destructive lifecycle calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_transfer_queue(*, active=0, deferred=0):
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.queue = [object() for _ in range(active)]
    queue._deferred_releases = [object() for _ in range(deferred)]
    return queue


def _make_idle_decode_scheduler(transfer_queue):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running_batch = MagicMock()
    scheduler.running_batch.is_empty.return_value = True
    scheduler.running_batch.reqs = []
    scheduler.chunked_req = None
    scheduler.dllm_manager = MagicMock()
    scheduler.dllm_manager.any_staging_reqs.return_value = False
    scheduler.last_batch = None
    scheduler.cur_batch_for_debug = object()
    scheduler.enable_overlap = False
    scheduler.ps = ParallelState.trivial()
    scheduler.running_mbs = []
    scheduler.waiting_queue = []
    scheduler.grammar_manager = MagicMock()
    scheduler.grammar_manager.grammar_queue = []
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[]
    )
    scheduler.disagg_decode_transfer_queue = transfer_queue
    scheduler.decode_offload_manager = None
    scheduler.enable_hisparse = False
    scheduler.enable_hierarchical_cache = False

    # Destructive flush collaborators.  Tests assert these remain untouched
    # when either visible transfers or queue-removed deferred holds exist.
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.is_stats_logging_rank = False
    scheduler.draft_worker = MagicMock()
    return scheduler


class TestCompactDecodeLifecycleGuards(CustomTestCase):
    def test_visible_transfer_and_deferred_hold_both_block_fully_idle(self):
        for active, deferred in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(active=active, deferred=deferred):
                scheduler = _make_idle_decode_scheduler(
                    _make_transfer_queue(active=active, deferred=deferred)
                )
                self.assertFalse(scheduler.is_fully_idle())

        scheduler = _make_idle_decode_scheduler(_make_transfer_queue())
        self.assertTrue(scheduler.is_fully_idle())

    def test_flush_preserves_deferred_owner_and_all_pools(self):
        transfer_queue = _make_transfer_queue(deferred=1)
        held_owner = transfer_queue._deferred_releases[0]
        scheduler = _make_idle_decode_scheduler(transfer_queue)

        self.assertFalse(scheduler.flush_cache())

        self.assertEqual(transfer_queue._deferred_releases, [held_owner])
        scheduler.tree_cache.reset.assert_not_called()
        scheduler.req_to_token_pool.clear.assert_not_called()
        scheduler.token_to_kv_pool_allocator.clear.assert_not_called()
        scheduler.draft_worker.clear_cache_pool.assert_not_called()

    def test_queue_memory_release_fails_closed_without_forgetting_owners(self):
        for active, deferred in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(active=active, deferred=deferred):
                transfer_queue = _make_transfer_queue(active=active, deferred=deferred)
                active_owners = list(transfer_queue.queue)
                deferred_owners = list(transfer_queue._deferred_releases)

                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"active_transfers={active}.*deferred_drain_ack_holds={deferred}",
                ):
                    transfer_queue.release_memory_occupation()

                self.assertEqual(transfer_queue.queue, active_owners)
                self.assertEqual(transfer_queue._deferred_releases, deferred_owners)

        # An actually idle queue needs no destructive cleanup and is accepted.
        _make_transfer_queue().release_memory_occupation()

    def test_forced_weight_release_preflight_does_not_pause_or_mutate_state(self):
        transfer_queue = _make_transfer_queue(deferred=1)
        held_owner = transfer_queue._deferred_releases[0]
        prealloc_queue = MagicMock()
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            disagg_decode_transfer_queue=transfer_queue,
            disagg_decode_prealloc_queue=prealloc_queue,
        )
        memory_saver = MagicMock()
        flush_cache = MagicMock()
        updater = SchedulerWeightUpdaterManager(
            tp_worker=MagicMock(),
            draft_worker=None,
            tp_cpu_group=MagicMock(),
            memory_saver_adapter=memory_saver,
            flush_cache=flush_cache,
            # Simulate an erroneous/forced upper-level idle result.  The local
            # ownership preflight must still reject the teardown.
            is_fully_idle=MagicMock(return_value=True),
            scheduler=scheduler,
        )

        with self.assertRaisesRegex(RuntimeError, "deferred_drain_ack_holds=1"):
            updater.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )

        self.assertEqual(transfer_queue._deferred_releases, [held_owner])
        self.assertEqual(updater.offload_tags, set())
        prealloc_queue.release_memory_occupation.assert_not_called()
        memory_saver.pause.assert_not_called()
        flush_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
