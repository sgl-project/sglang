"""Unit tests for HiCache PP synchronization."""

import pickle
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.mem_cache.unified_radix_cache import (
    _HICACHE_PP_ENVELOPE_SIZE,
    _HICACHE_PP_IDENTITY,
    _HICACHE_PP_PREFETCH_START,
    _HICACHE_PP_QUEUE_SLOTS,
    _HICACHE_PP_STORAGE_START,
    _HICACHE_PP_TERMINATE,
    UnifiedRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(CustomTestCase):
    def test_drain_waits_all_and_clears(self):
        holder = _Holder()
        works = [_FakeWork(), _FakeWork(), _FakeWork()]
        holder.work_list = list(works)

        UnifiedRadixCache._drain_async_work(holder)

        self.assertTrue(all(w.waited for w in works))
        self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        holder = _Holder()
        holder.work_list = []

        UnifiedRadixCache._drain_async_work(holder)

        self.assertEqual(holder.work_list, [])


class TestUnifiedPPSyncBatching(CustomTestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.pp_group = object()
        cache.host_memory_mode = "cache"
        cache._hicache_storage_configured = False
        cache._hicache_pp_sync_round = 0
        cache._hicache_pp_prefetch_pending = {}
        cache._hicache_pp_prefetch_results = {}
        cache._hicache_pp_prefetch_keys = {}
        cache._hicache_pp_prefetch_inflight = set()
        cache._hicache_pp_write_acks_consumed = 0
        cache._hicache_pp_write_ack_snapshots = {}
        cache._hicache_pp_round_reservations = {}
        cache._hicache_pp_reserved_counts = [0] * _HICACHE_PP_QUEUE_SLOTS
        cache._hicache_pp_sync_state_logged = False
        cache.work_list = []
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce_attn_groups = MagicMock()
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
        )
        return cache

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        follower = self._make_cache(1, [True], [True])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=1)

        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_ready_count_uses_slowest_pp_ack_prefix(self):
        """A PP0-ready ACK cannot be popped before every stage has it queued."""
        leader = self._make_cache(0, [], [True])
        follower = self._make_cache(1, [], [])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.loading_check.assert_called_once_with(finish_count=0)
        follower.loading_check.assert_called_once_with(finish_count=0)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache._hicache_storage_configured = True
        cache.tree_core.enable_storage = True
        cache._all_reduce_attn_groups = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cc = cache.cache_controller
        for size, name in enumerate(
            (
                "prefetch_hit_queue",
                "ack_prefetch_queue",
                "ack_backup_queue",
                "host_mem_release_queue",
            ),
            start=1,
        ):
            queue = Queue()
            for _ in range(size):
                queue.put(object())
            setattr(cc, name, queue)

        cache.check_hicache_events()

        cache._all_reduce.assert_not_called()
        cache._all_reduce_attn_groups.assert_called_once()
        cache.writing_check.assert_called_once_with(finish_count=1)
        cache.loading_check.assert_called_once_with(finish_count=2)
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=1,
            n_ack_prefetch=2,
            n_backup=3,
            n_release=4,
            extra_release_counts={},
            log_metrics=True,
        )

    def _make_storage_cache(self, pp_rank, backup_count):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=True,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.pp_group = object()
        cache.host_memory_mode = "cache"
        cache._hicache_storage_configured = True
        cache._hicache_pp_sync_round = 0
        cache._hicache_pp_prefetch_pending = {}
        cache._hicache_pp_prefetch_results = {}
        cache._hicache_pp_prefetch_keys = {}
        cache._hicache_pp_prefetch_inflight = set()
        cache._hicache_pp_write_acks_consumed = 0
        cache._hicache_pp_write_ack_snapshots = {}
        cache._hicache_pp_round_reservations = {}
        cache._hicache_pp_reserved_counts = [0] * _HICACHE_PP_QUEUE_SLOTS
        cache._hicache_pp_sync_state_logged = False
        cache.work_list = []
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce_attn_groups = MagicMock()
        cache.flush_pending_backups = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.dec_host_lock_ref = MagicMock()
        cache.ongoing_backup = {}

        backup_queue = Queue()
        for operation_id in range(backup_count):
            operation = SimpleNamespace(id=operation_id, completed_tokens=1)
            backup_queue.put(operation)
            cache.ongoing_backup[operation_id] = (object(), object())

        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[],
            ack_load_queue=[],
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=backup_queue,
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            mem_pool_host=MagicMock(),
        )

        def broadcast_pp0_counts(counts, _):
            if counts.numel() == 8:
                counts.copy_(torch.tensor([0, 0, 0, 0, 2, 0, 0, 0]))
            else:
                counts.zero_()

        cache._all_reduce = MagicMock(side_effect=broadcast_pp0_counts)
        return cache

    def test_pp_drain_uses_common_prefix_when_backup_counts_diverge(self):
        """A lagging PP stage must not wait for a nonexistent backup ACK."""
        leader = self._make_storage_cache(pp_rank=0, backup_count=2)
        follower = self._make_storage_cache(pp_rank=1, backup_count=1)
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_backup_queue.qsize(), 1)
        self.assertEqual(follower.cache_controller.ack_backup_queue.qsize(), 0)
        leader.dec_host_lock_ref.assert_called_once()
        follower.dec_host_lock_ref.assert_called_once()

    def test_single_stage_direct_storage_drain_uses_unified_consumer(self):
        cache = self._make_storage_cache(pp_rank=0, backup_count=1)
        cache.pp_size = 1

        self.assertTrue(cache.drain_storage_control_queues())

        self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)
        cache.dec_host_lock_ref.assert_called_once()

    def test_pp_rejects_divergent_duplicate_reclaim_digest(self):
        leader = self._make_storage_cache(pp_rank=0, backup_count=0)
        follower = self._make_storage_cache(pp_rank=1, backup_count=0)
        leader.tree_core.write_back_duplicate_reclaim_digest = 11
        follower.tree_core.write_back_duplicate_reclaim_digest = 17
        errors = []

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        for cache in (leader, follower):
            try:
                cache._apply_hicache_pp_ring_payload(final)
            except Exception as error:
                errors.append((cache.pp_rank, error))

        self.assertEqual({rank for rank, _ in errors}, {0, 1})
        self.assertTrue(all(isinstance(error, AssertionError) for _, error in errors))
        self.assertTrue(
            all(
                "duplicate-reclaim victims diverged" in str(error)
                for _, error in errors
            )
        )

    def test_storage_config_keeps_mixed_enablement_nonblocking(self):
        leader = self._make_storage_cache(pp_rank=0, backup_count=1)
        follower = self._make_storage_cache(pp_rank=1, backup_count=1)
        follower.enable_storage = False
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_backup_queue.qsize(), 0)
        self.assertEqual(follower.cache_controller.ack_backup_queue.qsize(), 1)
        leader.dec_host_lock_ref.assert_called_once()
        follower.dec_host_lock_ref.assert_not_called()

    def test_different_prefetch_keys_share_fixed_envelope(self):
        leader = self._make_cache(0, [], [])
        follower = self._make_cache(1, [], [])
        for cache in (leader, follower):
            cache._pp_sync = MagicMock()
        leader._register_hicache_pp_prefetch_verdict("terminate", "req-a", True)
        follower._register_hicache_pp_prefetch_verdict("ready", "req-b", False)
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)

        self.assertEqual(proposal.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        self.assertEqual(final.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        leader._pp_sync.assert_not_called()
        follower._pp_sync.assert_not_called()

    def test_ready_count_and_loading_check_do_not_consume_ack_twice(self):
        leader = self._make_cache(0, [], [True])
        follower = self._make_cache(1, [], [True])
        for cache in (leader, follower):
            del cache.writing_check
            del cache.loading_check
            cache.metrics_collector = None
            cache.ongoing_load_back = {cache.pp_rank: (object(), object(), object())}
            cache.dec_lock_ref = MagicMock()
            cache.dec_host_lock_ref = MagicMock()
            cache.tree_core.finish_load_back = MagicMock()
            ack = cache.cache_controller.ack_load_queue[0]
            ack.finish_event.synchronize = MagicMock()
            ack.node_ids = [cache.pp_rank]
            ack.num_tokens_by_pool = {}
            ack.num_bytes = 0
            ack.timing_enabled = False

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader.loading_check()
        follower.loading_check()
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)
        leader.loading_check()
        follower.loading_check()

        self.assertEqual(leader.cache_controller.ack_load_queue, [])
        self.assertEqual(follower.cache_controller.ack_load_queue, [])

    def test_blocking_write_during_pending_round_is_not_consumed_twice(self):
        leader = self._make_cache(0, [True], [])
        follower = self._make_cache(1, [True], [])
        for cache in (leader, follower):
            del cache.writing_check
            cache.metrics_collector = None
            cache.ongoing_write_through = {cache.pp_rank: object()}
            cache._finish_write_through_ack = MagicMock(
                side_effect=lambda ack_id, cache=cache: cache.ongoing_write_through.pop(
                    ack_id
                )
            )
            ack = cache.cache_controller.ack_write_queue[0]
            ack.finish_event.synchronize = MagicMock()
            ack.node_ids = [cache.pp_rank]

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader.writing_check(write_back=True)
        follower.writing_check(write_back=True)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_write_queue, [])
        self.assertEqual(follower.cache_controller.ack_write_queue, [])


class TestHiCachePPConsensusRing(CustomTestCase):
    """Regression coverage for piggybacking on the scheduler consensus ring."""

    def setUp(self):
        self.helper = TestUnifiedPPSyncBatching()

    def test_ring_uses_common_ack_prefix(self):
        leader = self.helper._make_cache(0, [], [True])
        follower = self.helper._make_cache(1, [], [])

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.loading_check.assert_called_once_with(finish_count=0)
        follower.loading_check.assert_called_once_with(finish_count=0)

    def test_ring_applies_prefetch_verdict_on_same_round(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])
        self.assertIsNone(
            leader._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", True
            )
        )
        self.assertIsNone(
            follower._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", False
            )
        )

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        self.assertNotIn(
            leader._hicache_pp_prefetch_tag("terminate", "req-shared"),
            follower._hicache_pp_prefetch_results,
        )

        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)
        self.assertTrue(
            leader._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", False
            )
        )
        self.assertTrue(
            follower._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", False
            )
        )

    def test_ring_keeps_fixed_shape_without_hicache_collective(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])
        leader._pp_sync = MagicMock()
        follower._pp_sync = MagicMock()
        leader._register_hicache_pp_prefetch_verdict("terminate", "req-a", True)
        follower._register_hicache_pp_prefetch_verdict("ready", "req-b", False)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            proposal = leader._build_hicache_pp_ring_payload()
            final = follower._build_hicache_pp_ring_payload(proposal)

        self.assertEqual(proposal.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        self.assertEqual(final.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        all_reduce.assert_not_called()
        leader._pp_sync.assert_not_called()
        follower._pp_sync.assert_not_called()

    def test_ring_first_and_last_round_have_no_pending_work(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])

        self.assertFalse(leader._apply_hicache_pp_ring_payload(None))
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        self.assertTrue(leader._apply_hicache_pp_ring_payload(final))
        self.assertTrue(follower._apply_hicache_pp_ring_payload(final))
        self.assertFalse(leader._apply_hicache_pp_ring_payload(final))
        self.assertFalse(follower._apply_hicache_pp_ring_payload(final))

    def test_ring_result_is_applied_at_existing_consensus_consumer(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        scheduler = scheduler_pp_mixin.SchedulerPPMixin()
        scheduler.tree_cache = SimpleNamespace(
            _apply_hicache_pp_ring_payload=MagicMock()
        )

        forwarded = scheduler.process_bootstrapped_queue(
            scheduler_pp_mixin._PPBootstrapPayload(None, final)
        )

        transport_copy = pickle.loads(pickle.dumps(forwarded))
        self.assertEqual(len(forwarded), 2)
        self.assertEqual(transport_copy.rids, forwarded.rids)
        scheduler.tree_cache._apply_hicache_pp_ring_payload.assert_called_once_with(
            final
        )
        self.assertIs(forwarded.hicache, final)

    def test_overlapping_rounds_offer_each_queue_item_once(self):
        leader = self.helper._make_cache(0, [], [True])
        follower = self.helper._make_cache(1, [], [True])
        first = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )
        second = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )

        self.assertEqual(int(second.envelope[1]), 0)
        for cache in (leader, follower):
            cache._apply_hicache_pp_ring_payload(second)
            cache._apply_hicache_pp_ring_payload(first)
            self.assertFalse(cache._apply_hicache_pp_ring_payload(first))
            self.assertEqual(
                [
                    item.kwargs["finish_count"]
                    for item in cache.loading_check.call_args_list
                ],
                [0, 1],
            )

    def test_unaccepted_local_claim_is_reoffered_after_result(self):
        leader = self.helper._make_storage_cache(0, backup_count=5)
        follower = self.helper._make_storage_cache(1, backup_count=2)
        first = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )
        leader._apply_hicache_pp_ring_payload(first)
        follower._apply_hicache_pp_ring_payload(first)

        next_proposal = leader._build_hicache_pp_ring_payload()

        self.assertEqual(int(first.envelope[_HICACHE_PP_STORAGE_START + 2]), 2)
        self.assertEqual(int(next_proposal.envelope[_HICACHE_PP_STORAGE_START + 2]), 3)

    def test_follower_pp0_only_slots_do_not_reserve_counts(self):
        follower = self.helper._make_cache(1, [], [])
        follower._register_hicache_pp_prefetch_verdict(
            "terminate", "follower-only", True
        )
        before = list(follower._hicache_pp_reserved_counts)

        payload = follower._build_hicache_pp_ring_payload()

        self.assertEqual(
            int(payload.envelope[_HICACHE_PP_TERMINATE]), _HICACHE_PP_IDENTITY
        )
        self.assertTrue(
            torch.all(
                payload.envelope[_HICACHE_PP_PREFETCH_START:] == _HICACHE_PP_IDENTITY
            )
        )
        self.assertEqual(follower._hicache_pp_reserved_counts, before)


if __name__ == "__main__":
    unittest.main()
