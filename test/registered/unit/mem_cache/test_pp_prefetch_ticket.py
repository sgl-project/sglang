"""CPU tests for ticket submission, eager prefetch, and buffer ownership."""

import threading
import unittest
from array import array
from queue import Queue
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PPPrefetchPoolSpec,
    PPPrefetchState,
    PPPrefetchTicket,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_ticket(**kwargs):
    fields = dict(
        rid="hit",
        token_ids=list(range(8)),
        last_hash="ab" * 32,
        prefix_keys=["prefix-key"],
        matched_prefix_tokens=[10, 11, 12, 13],
        extra_key="adapter",
        cache_salt="tenant",
        is_bigram=False,
        pool_specs=(),
        storage_hit_count=8,
    )
    return PPPrefetchTicket(**(fields | kwargs))


class TestPPPrefetchTicket(unittest.TestCase):
    def setUp(self):
        # Avoid allocating CUDA pools or starting the controller's background threads.
        self.controller = c = HybridCacheController.__new__(HybridCacheController)
        c.page_size = c.prefetch_threshold = 4
        c.get_hash_str = get_hash_str
        c.pp_rank = c.tp_rank = 0
        c.pp_size, c.tp_size = 4, 2
        c.pp_group = "pp"
        c.pp_prefetch_command_group = "command"
        c.prefetch_hits_sync_groups = ["pp", "tp"]
        c.prefetch_completion_sync_groups = ["pp", "tp"]
        c.pp_prefetch_states = {}
        c.pp_prefetch_decisions = {}
        c.pp_prefetch_state_lock = threading.Lock()
        c.pp_prefetch_command_queue = Mock(spec=Queue)
        c.prefetch_buffer = Queue()
        c.prefetch_queue = Queue()
        c.prefetch_sync_queue = Queue()
        c.ack_prefetch_queue = Queue()
        c.storage_stop_event = threading.Event()
        c.prefetch_tokens_occupied = 0
        c.mem_pool_host = Mock()
        c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
        c._storage_hit_query = Mock(return_value=(["h0", "h1"], 8))
        c._all_reduce = Mock()
        ranks = {"pp": [0, 2, 4, 6], "tp": [0, 1], "command": [0, 2, 4, 6]}
        patcher = patch.object(
            torch.distributed, "get_process_group_ranks", side_effect=ranks.__getitem__
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def submit(self, rid="hit", pool_transfers=None):
        return self.controller.submit_prefetch(
            rid,
            RadixKey(array("q", range(8)), extra_key="adapter", cache_salt="tenant"),
            "ab" * 32,
            ["prefix-key"],
            [10, 11, 12, 13],
            pool_transfers,
        )

    def sync_acks(self, *acks):
        c = self.controller
        for ack in acks:
            c.prefetch_sync_queue.put(ack)
        # Run exactly these ACKs without timing-dependent sleeps or live collectives.
        with patch.object(
            c.storage_stop_event, "is_set", side_effect=[False] * len(acks) + [True]
        ):
            c.prefetch_sync_thread_func()

    def test_miss_and_subpage_hit_do_not_broadcast(self):
        c = self.controller
        for hit in (0, 3):
            with self.subTest(hit=hit):
                c._storage_hit_query.return_value = ([], hit)
                submission = self.submit()
                self.assertFalse(submission.decision)
                self.assertIsNone(submission.operation)
                self.assertEqual(c.pp_prefetch_states, {})
                c.pp_prefetch_command_queue.put.assert_not_called()
                c.pp_prefetch_command_queue.join.assert_not_called()
                self.assertFalse(c.get_prefetch_submission("hit").decision)
                self.assertIsNone(c.get_prefetch_submission("hit"))
                self.assertFalse(c.release_pp_prefetch("hit"))

    def test_query_failure_falls_back_to_miss(self):
        c = self.controller
        c._storage_hit_query.side_effect = RuntimeError("storage unavailable")
        with self.assertLogs(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller", level="ERROR"
        ):
            self.assertFalse(self.submit().decision)
        # A failed local query still participates in TP consensus.
        self.assertEqual(c._all_reduce.call_args.args[0].item(), 0)
        c.pp_prefetch_command_queue.put.assert_not_called()

    def test_tp_peer_miss_suppresses_local_hit(self):
        c = self.controller
        c._all_reduce.side_effect = lambda tensor, *_: tensor.zero_()
        self.assertFalse(self.submit().decision)
        self.assertEqual(c.pp_prefetch_states, {})
        c.pp_prefetch_command_queue.put.assert_not_called()

    def test_query_shards_cover_every_pp_namespace(self):
        c = self.controller
        for tp_size in (1, 2, 8):
            queried = []
            for tp_rank in range(tp_size):
                c.tp_size, c.tp_rank = tp_size, tp_rank
                c._storage_hit_query.reset_mock()
                self.submit(rid=f"{tp_size}-{tp_rank}")
                queried.extend(
                    args.kwargs["pp_rank"]
                    for args in c._storage_hit_query.call_args_list
                )
            self.assertEqual(sorted(queried), list(range(c.pp_size)))

    def test_hit_uses_tp_min_and_broadcasts_one_aligned_ticket(self):
        c = self.controller
        c._storage_hit_query.side_effect = [(["h0", "h1"], 8), (["h0"], 7)]

        def reduce_hit(tensor, op, groups):
            self.assertEqual(tensor.item(), 7)
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            self.assertEqual(groups, ["tp"])  # No cross-PP hit collective.
            tensor.fill_(5)

        c._all_reduce.side_effect = reduce_hit
        submission = self.submit()
        ticket = c.pp_prefetch_command_queue.put.call_args.args[0]
        self.assertEqual(ticket, make_ticket(storage_hit_count=4))
        self.assertTrue(submission.decision)
        self.assertTrue(submission.operation.is_pp_broadcast)
        self.assertIs(c.pp_prefetch_states["hit"].operation, submission.operation)
        c.pp_prefetch_command_queue.put.assert_called_once()
        c.pp_prefetch_command_queue.join.assert_called_once_with()
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        self.assertTrue(c.get_prefetch_submission("hit").decision)

    def test_duplicate_ticket_is_rejected_without_second_broadcast(self):
        c = self.controller
        first = self.submit()
        with self.assertRaisesRegex(RuntimeError, "Duplicate PP prefetch"):
            self.submit()
        self.assertIs(c.pp_prefetch_states["hit"].operation, first.operation)
        c.pp_prefetch_command_queue.put.assert_called_once()

    def test_downstream_skips_submission_and_cannot_query(self):
        c = self.controller
        c.pp_rank = 1
        submission = c.get_prefetch_submission("not-arrived")
        self.assertIsNone(submission.operation)
        self.assertIsNone(submission.decision)
        with self.assertRaisesRegex(RuntimeError, "Only PP0"):
            self.submit()
        c._storage_hit_query.assert_not_called()

    def test_tp_only_keeps_normal_prefetch_queue(self):
        c = self.controller
        c.pp_prefetch_command_group = None
        self.assertIsNone(c.get_prefetch_submission("hit"))
        submission = self.submit()
        self.assertIsNone(submission.decision)
        self.assertIs(c.prefetch_queue.get_nowait(), submission.operation)
        c._storage_hit_query.assert_not_called()
        c.pp_prefetch_command_queue.put.assert_not_called()

    def test_downstream_prefetches_before_request_without_requery(self):
        c = self.controller
        c.pp_rank = 1
        ticket = make_ticket(
            is_bigram=True,
            token_ids=list(range(9)),
            pool_specs=(
                PPPrefetchPoolSpec(PoolName.SWA, 4, keys=["h1"]),
                PPPrefetchPoolSpec(
                    PoolName.DRAFT_SWA, 0, indices_from_pool=PoolName.SWA
                ),
            ),
        )
        commands = iter([ticket, None])

        def broadcast(objects, **kwargs):
            self.assertEqual(kwargs, {"src": 0, "group": "command"})
            objects[0] = next(commands)

        with patch.object(torch.distributed, "broadcast_object_list", broadcast):
            c.pp_prefetch_command_thread_func()
        operation = c.prefetch_buffer.get_nowait()
        self.assertIs(c.pp_prefetch_states["hit"].operation, operation)
        self.assertEqual(len(operation.token_ids), 8)
        self.assertTrue(operation.token_ids.is_bigram)
        self.assertEqual(operation.storage_start, 4)
        self.assertEqual(operation.storage_hit_count, 8)
        self.assertEqual(len(operation.hash_value), 2)
        self.assertEqual(operation.all_hash_values, operation.hash_value)
        self.assertEqual(operation.prefix_keys, ticket.prefix_keys)
        self.assertEqual(c.prefetch_tokens_occupied, 8)
        swa, draft_swa = operation.pool_transfers
        self.assertEqual(swa.host_indices.numel(), 4)
        self.assertIs(draft_swa.host_indices, swa.host_indices)
        self.assertFalse(operation.pool_transfers_done)
        self.assertEqual(c.mem_pool_host.alloc.call_count, 2)
        c.mem_pool_host.alloc.assert_called_with(4, pool=PoolName.SWA)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        c._storage_hit_query.assert_not_called()
        c.pp_prefetch_command_queue.get.assert_not_called()

    def test_storage_query_forwards_pp_namespace_for_kv_and_sidecars(self):
        c = self.controller
        c.storage_backend = Mock()
        c.storage_backend.batch_exists.return_value = 1
        for transfers in (None, [PoolTransfer(PoolName.SWA, keys=["h1"])]):
            with self.subTest(sidecars=bool(transfers)):
                operation = PrefetchOperation(
                    "hit",
                    list(range(8)),
                    prefix_keys=["prefix"],
                    pool_transfers=transfers,
                )
                c.storage_backend.batch_exists_v2.return_value = (
                    operation.pool_storage_result
                )
                operation.pool_storage_result.kv_hit_pages = 1
                hashes, hits = HybridCacheController._storage_hit_query(
                    c, operation, pp_rank=3
                )
                query = (
                    c.storage_backend.batch_exists_v2
                    if transfers
                    else c.storage_backend.batch_exists
                )
                extra_info = query.call_args.args[-1]
                self.assertEqual(extra_info.extra_info, {"pp_rank": 3})
                self.assertEqual(extra_info.prefix_keys, ["prefix"])
                self.assertIsNot(extra_info.prefix_keys, operation.prefix_keys)
                self.assertEqual(hits, 4)
                self.assertEqual(hashes, operation.all_hash_values[:1])

    def test_source_reuses_operation_and_acknowledges_commands(self):
        c = self.controller
        operation = self.submit().operation
        ticket = c.pp_prefetch_states["hit"].ticket
        c.pp_prefetch_command_queue = Queue()
        c.pp_prefetch_command_queue.put(ticket)
        c.pp_prefetch_command_queue.put(None)
        with patch.object(torch.distributed, "broadcast_object_list") as broadcast:
            c.pp_prefetch_command_thread_func()
        self.assertIs(c.prefetch_buffer.get_nowait(), operation)
        self.assertEqual(broadcast.call_count, 2)  # Ticket and shutdown sentinel.
        self.assertEqual(c.pp_prefetch_command_queue.unfinished_tasks, 0)

    def test_command_allocation_failure_still_enters_io_completion_path(self):
        c = self.controller
        operation = self.submit().operation
        ticket = c.pp_prefetch_states["hit"].ticket
        c.mem_pool_host.alloc.return_value = None
        c.mem_pool_host.alloc.side_effect = None
        c.pp_prefetch_command_queue.get.side_effect = [ticket, None]
        with patch.object(torch.distributed, "broadcast_object_list"):
            c.pp_prefetch_command_thread_func()
        self.assertIs(c.prefetch_buffer.get_nowait(), operation)
        self.assertTrue(operation.is_terminated())
        self.assertEqual(operation.host_indices.numel(), 0)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))

    def test_pool_specs_copy_metadata_without_host_indices(self):
        for source in (None, PoolName.KV):
            with self.subTest(source=source):
                transfer = PoolTransfer(
                    PoolName.INDEXER,
                    host_indices=torch.arange(8),
                    keys=["h0", "h1"],
                    indices_from_pool=source,
                )
                spec = PPPrefetchPoolSpec.from_transfer(transfer)
                self.assertEqual(spec.num_slots, 8 if source is None else 0)
                self.assertEqual(spec.indices_from_pool, source)
                transfer.keys.append("h2")
                self.assertEqual(spec.keys, ["h0", "h1"])
                self.assertFalse(hasattr(spec, "host_indices"))

    def test_allocate_and_release_sidecars_without_double_free(self):
        c = self.controller
        swa_indices = torch.arange(20, 24)
        transfers = [
            PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV),
            PoolTransfer(
                PoolName.SWA,
                host_indices=swa_indices,
                keys=["h1"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            PoolTransfer(PoolName.DRAFT_SWA, indices_from_pool=PoolName.SWA),
        ]
        ticket = make_ticket(
            pool_specs=tuple(PPPrefetchPoolSpec.from_transfer(t) for t in transfers)
        )
        operation = PrefetchOperation("hit", ticket.token_ids, pool_transfers=transfers)
        self.assertTrue(c._allocate_pp_prefetch_buffers(ticket, operation))
        c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
        indexer, swa, draft_swa = operation.pool_transfers
        self.assertIs(indexer.host_indices, operation.host_indices)
        self.assertIs(swa.host_indices, swa_indices)
        self.assertIs(draft_swa.host_indices, swa_indices)
        self.assertEqual(swa.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        state = PPPrefetchState(ticket, operation)
        c.pp_prefetch_states["hit"] = state
        state.ready_event.set()
        self.assertTrue(c.release_pp_prefetch("hit"))
        self.assertFalse(c.release_pp_prefetch("hit"))
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        self.assertEqual(
            [args.kwargs["pool"] for args in c.mem_pool_host.free.call_args_list],
            [PoolName.KV, PoolName.SWA],
        )

    def test_sidecar_allocation_failure_rolls_back_owned_buffers(self):
        c = self.controller
        kv, swa = torch.arange(8), torch.arange(4)
        c.mem_pool_host.alloc.side_effect = [kv, swa, None]
        ticket = make_ticket(
            pool_specs=(
                PPPrefetchPoolSpec(PoolName.SWA, 4),
                PPPrefetchPoolSpec(PoolName.MAMBA, 1),
            )
        )
        operation = PrefetchOperation("hit", ticket.token_ids)
        self.assertFalse(c._allocate_pp_prefetch_buffers(ticket, operation))
        self.assertEqual(c.mem_pool_host.free.call_count, 2)
        self.assertIs(c.mem_pool_host.free.call_args_list[0].args[0], swa)
        self.assertIs(c.mem_pool_host.free.call_args_list[1].args[0], kv)
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        self.assertIsNone(operation.host_indices)

    def test_missing_derived_source_rolls_back_but_preserves_borrowed_sidecar(self):
        c = self.controller
        borrowed = PoolTransfer(PoolName.SWA, host_indices=torch.arange(4))
        ticket = make_ticket(
            pool_specs=(
                PPPrefetchPoolSpec.from_transfer(borrowed),
                PPPrefetchPoolSpec(
                    PoolName.INDEXER, 0, indices_from_pool=PoolName.MAMBA
                ),
            )
        )
        operation = PrefetchOperation(
            "hit", ticket.token_ids, pool_transfers=[borrowed]
        )
        self.assertFalse(c._allocate_pp_prefetch_buffers(ticket, operation))
        c.mem_pool_host.free.assert_called_once()
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs, {"pool": PoolName.KV})
        self.assertIs(operation.pool_transfers[0], borrowed)
        self.assertIsNotNone(borrowed.host_indices)

    def test_progressive_ack_publishes_only_global_completion(self):
        c = self.controller
        operation = self.submit().operation
        state = c.pp_prefetch_states["hit"]
        c._allocate_pp_prefetch_buffers(state.ticket, operation)
        operation.hash_value = ["h0", "h1"]
        operation.pool_transfers_done = False

        def reduce_ack(tensor, op, groups):
            self.assertFalse(state.ready_event.is_set())
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            self.assertEqual(groups, c.prefetch_completion_sync_groups)
            tensor.clamp_(max=4 if tensor.ndim == 0 else 1)

        c._all_reduce.side_effect = reduce_ack
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=8))
        self.assertEqual(operation.completed_tokens, 4)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        self.sync_acks(PrefetchAck("hit", operation, pool_hits={PoolName.SWA: 2}))
        self.assertEqual(operation.pool_storage_result.extra_pool_hit_pages["swa"], 1)
        self.assertTrue(operation.pool_transfers_done)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        self.assertTrue(c.is_pp_prefetch_ready("hit"))
        self.assertEqual(operation.hash_value, ["h0"])
        self.assertEqual(operation.storage_hit_count, 4)
        self.assertTrue(c.ack_prefetch_queue.empty())

    def test_normal_ack_still_goes_to_scheduler(self):
        c = self.controller
        operation = PrefetchOperation("tp", list(range(8)))
        ack = PrefetchAck("tp", operation, completed_tokens=4, completed_req=True)
        self.sync_acks(ack)
        self.assertIs(c.ack_prefetch_queue.get_nowait(), ack)
        self.assertEqual(c.pp_prefetch_states, {})

    def test_release_waits_for_final_ack(self):
        c = self.controller
        operation = self.submit().operation
        state = c.pp_prefetch_states["hit"]
        c._allocate_pp_prefetch_buffers(state.ticket, operation)
        self.assertTrue(c.release_pp_prefetch("hit"))
        self.assertTrue(state.release_requested)
        c.mem_pool_host.free.assert_not_called()
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=4))
        c.mem_pool_host.free.assert_not_called()
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        c.mem_pool_host.free.assert_called_once()
        self.assertEqual(c.pp_prefetch_states, {})
        self.assertEqual(c.pp_prefetch_decisions, {})
        self.assertEqual(c.prefetch_tokens_occupied, 0)

    def test_take_ready_transfers_ownership_exactly_once(self):
        c = self.controller
        operation = self.submit().operation
        state = c.pp_prefetch_states["hit"]
        c._allocate_pp_prefetch_buffers(state.ticket, operation)
        operation.completed_tokens = 8
        # Simulate a downstream consumer waiting for its local final ACK.
        state.ready_event = Mock(spec=threading.Event)
        self.assertIs(c.take_ready_pp_prefetch("hit"), state)
        state.ready_event.wait.assert_called_once_with()
        self.assertIsNone(c.take_ready_pp_prefetch("hit"))
        self.assertFalse(c.release_pp_prefetch("hit"))
        c.mem_pool_host.free.assert_not_called()
        self.assertEqual(c.prefetch_tokens_occupied, 8)  # Now owned by buffer mode.

    def test_take_zero_completion_releases_buffers(self):
        c = self.controller
        operation = self.submit().operation
        state = c.pp_prefetch_states["hit"]
        c._allocate_pp_prefetch_buffers(state.ticket, operation)
        state.ready_event.set()
        self.assertIs(c.take_ready_pp_prefetch("hit"), state)
        self.assertIsNone(operation.host_indices)
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        c.mem_pool_host.free.assert_called_once()


class TestPPTicketAdmission(unittest.TestCase):
    def setUp(self):
        self.cache = cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.pp_rank = 1
        cache.cache_controller = Mock()
        cache.buffer_pipeline = Mock()
        cache._all_reduce = Mock()
        cache.ongoing_prefetch = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.root_node_handle = Mock(return_value=0)
        cache._handle_prefetch_result = Mock()

    def test_local_downstream_ready_cannot_admit_before_pp0(self):
        cache = self.cache
        cache.cache_controller.is_pp_prefetch_ready.return_value = True
        self.assertFalse(
            cache.check_prefetch_progress("hit", pp_prefetch_ticketed=True)
        )
        self.assertEqual(cache._all_reduce.call_args.args[0].item(), 0)
        cache.cache_controller.is_pp_prefetch_ready.assert_not_called()
        cache.cache_controller.take_ready_pp_prefetch.assert_not_called()

    def test_pp0_not_ready_does_not_consume_ticket(self):
        cache = self.cache
        cache.pp_rank = 0
        cache.cache_controller.is_pp_prefetch_ready.return_value = False
        self.assertFalse(
            cache.check_prefetch_progress("hit", pp_prefetch_ticketed=True)
        )
        cache.cache_controller.is_pp_prefetch_ready.assert_called_once_with("hit")
        cache.cache_controller.take_ready_pp_prefetch.assert_not_called()

    def test_admitted_ticket_enters_existing_buffer_result_path(self):
        cache = self.cache
        ticket = make_ticket()
        operation = PrefetchOperation("hit", ticket.token_ids)
        operation.host_indices = torch.arange(8)
        operation.completed_tokens = 4
        cache.cache_controller.take_ready_pp_prefetch.return_value = PPPrefetchState(
            ticket, operation
        )
        cache._all_reduce.side_effect = lambda tensor, _: tensor.fill_(1)
        self.assertTrue(cache.check_prefetch_progress("hit", pp_prefetch_ticketed=True))
        cache.cache_controller.take_ready_pp_prefetch.assert_called_once_with("hit")
        cache.buffer_pipeline.set_prefix_ctx.assert_called_once_with(
            "hit",
            ticket.matched_prefix_tokens,
            extra_key="adapter",
            cache_salt="tenant",
        )
        cache.buffer_pipeline.try_lock_anchor.assert_called_once_with("hit")
        ongoing = cache.ongoing_prefetch["hit"]
        self.assertIs(ongoing.operation, operation)
        self.assertEqual(ongoing.prefetch_key.extra_key, "adapter")
        self.assertEqual(ongoing.prefetch_key.cache_salt, "tenant")
        tail = cache.cache_controller.append_host_mem_release.call_args.args[0]
        self.assertEqual(tail.tolist(), [4, 5, 6, 7])
        cache._handle_prefetch_result.assert_called_once_with(operation)


if __name__ == "__main__":
    unittest.main()
