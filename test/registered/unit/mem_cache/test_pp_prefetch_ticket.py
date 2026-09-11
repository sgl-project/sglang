"""CPU regressions for PP ticket ordering, admission, and buffer ownership."""

import pickle
import threading
import unittest
from array import array
from queue import Queue
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PPPrefetchDecision,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPPPrefetchTicket(unittest.TestCase):
    def setUp(self):
        self.c = c = HybridCacheController.__new__(HybridCacheController)
        c.page_size = c.prefetch_threshold = 4
        c.get_hash_str = get_hash_str
        c.pp_rank = c.tp_rank = 0
        c.pp_size, c.tp_size = 4, 2
        c.pp_group, c.pp_prefetch_command_group = "pp", "command"
        c.pp_prefetch_command_thread = None
        c.prefetch_hits_sync_groups = c.prefetch_completion_sync_groups = ["pp", "tp"]
        c.pp_prefetch_states, c.pp_prefetch_decisions = {}, {}
        c.pp_prefetch_state_lock = threading.Lock()
        c.pp_prefetch_command_queue = Queue()
        c.prefetch_queue = Queue()
        c.prefetch_sync_queue = Queue()
        c.ack_prefetch_queue = Queue()
        c.host_mem_release_queue = Queue()
        c.prefetch_buffer = Queue()
        c.storage_stop_event = threading.Event()
        c.prefetch_tokens_occupied = 0
        c.mem_pool_host = Mock(page_size=4)
        c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
        c._storage_hit_query = Mock(return_value=(["h0", "h1"], 8))
        c._all_reduce = Mock()
        ranks = {"pp": [0, 2, 4, 6], "tp": [0, 1], "command": [0, 2, 4, 6]}
        patcher = patch.object(
            torch.distributed, "get_process_group_ranks", side_effect=ranks.__getitem__
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cache = cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.pp_rank = 1
        cache.tree_core = Mock(enable_storage=True)
        cache.cache_controller = c
        cache._all_reduce = Mock()
        cache.ongoing_prefetch = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.root_node_handle = Mock(return_value=0)
        cache.buffer_pipeline = Mock()
        cache._handle_prefetch_result = Mock()

    def submit(self, rid="hit", pools=None):
        key = RadixKey(
            array("q", range(9)), "adapter", is_bigram=True, cache_salt="tenant"
        )
        return self.c.submit_prefetch(
            rid, key, "ab" * 32, ["prefix"], [10, 11, 12, 13], pools
        )

    def run_commands(self, *tickets):
        commands = iter((*tickets, None))
        self.c.pp_prefetch_command_queue.put(None)

        def broadcast(objects, **kwargs):
            if self.c.pp_rank:
                objects[0] = next(commands)

        with patch.object(torch.distributed, "broadcast_object_list", broadcast):
            self.c.pp_prefetch_command_thread_func()

    def sync_acks(self, *acks):
        for ack in acks:
            self.c.prefetch_sync_queue.put(ack)
        with patch.object(
            self.c.storage_stop_event,
            "is_set",
            side_effect=[False] * len(acks) + [True],
        ):
            self.c.prefetch_sync_thread_func()

    def test_hit_and_miss_are_reused_across_enqueue_retract_and_retry(self):
        c = self.c
        for hit in (0, 8):
            with self.subTest(hit=hit):
                rid = str(hit)
                c._storage_hit_query.return_value = ([], hit)
                self.assertEqual(self.submit(rid).decision, bool(hit))
                queries = c._storage_hit_query.call_count
                for _ in range(2):
                    self.assertEqual(
                        self.cache.prefetch_from_storage(rid, 0, []), bool(hit)
                    )
                self.assertEqual(c._storage_hit_query.call_count, queries)
                if hit:
                    state = c.pp_prefetch_states[rid]
                    state.ready_event.set()
                    c.take_ready_pp_prefetch(rid)
                    self.assertFalse(self.cache.prefetch_from_storage(rid, 0, []))
                self.assertTrue(self.cache.check_prefetch_progress(rid))
                self.assertFalse(c.release_pp_prefetch(rid))
                self.assertIsNone(c.get_prefetch_submission(rid))
        self.assertEqual(c.pp_prefetch_command_queue.qsize(), 1)
        self.cache._all_reduce.assert_not_called()

    def test_submission_does_not_wait_for_worker_and_uses_only_tp_consensus(self):
        c = self.c
        with patch.object(
            c.pp_prefetch_command_queue,
            "join",
            side_effect=AssertionError("scheduler blocked"),
        ):
            self.assertTrue(self.submit().decision)
            self.assertTrue(self.submit("next").decision)
        self.assertEqual(c.pp_prefetch_command_queue.qsize(), 2)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        c.mem_pool_host.alloc.assert_not_called()
        self.assertEqual(
            c._all_reduce.call_args.args[1:], (torch.distributed.ReduceOp.MIN, ["tp"])
        )
        self.assertEqual(
            [call.kwargs["pp_rank"] for call in c._storage_hit_query.call_args_list],
            [0, 2, 0, 2],
        )

    def test_tp_only_preserves_normal_prefetch(self):
        c = self.c
        c.pp_prefetch_command_group = None
        result = self.submit()
        self.assertIsNone(result.decision)
        self.assertIs(c.prefetch_queue.get_nowait(), result.operation)
        c._storage_hit_query.assert_not_called()

    def test_downstream_request_and_ticket_order_preserves_pp0_admission(self):
        c, cache = self.c, self.cache
        self.submit()
        ticket = pickle.loads(pickle.dumps(c.pp_prefetch_states["hit"].ticket))
        c.pp_rank = 1
        c.pp_prefetch_states.clear()
        cache.bind_prefetch_ticket("hit")
        self.assertFalse(cache.check_prefetch_progress("hit"))
        c._storage_hit_query.reset_mock()
        self.run_commands(ticket)
        operation = c.prefetch_buffer.get_nowait()
        self.assertTrue(operation.token_ids.is_bigram)
        self.assertEqual(len(operation.token_ids), 8)
        c._storage_hit_query.assert_not_called()
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=8))
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        self.assertFalse(cache.check_prefetch_progress("hit"))  # PP0 has not admitted.
        cache._all_reduce.side_effect = lambda tensor, _: tensor.fill_(1)
        self.assertTrue(cache.check_prefetch_progress("hit"))
        cache._handle_prefetch_result.assert_called_once_with(operation)
        key = cache.ongoing_prefetch["hit"].prefetch_key
        self.assertEqual((key.extra_key, key.cache_salt), ("adapter", "tenant"))
        self.assertTrue(key.is_bigram)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))

    def test_allocation_error_preserves_ack_sequence_and_next_ticket(self):
        c = self.c
        self.submit(
            pools=[
                PoolTransfer(PoolName.SWA, host_indices=torch.arange(4), keys=["h1"])
            ]
        )
        ticket = c.pp_prefetch_states.pop("hit").ticket
        following = pickle.loads(pickle.dumps(ticket))
        following.rid = "next"
        c.pp_rank = 1
        kv = torch.arange(8)
        c.mem_pool_host.alloc.side_effect = [
            kv,
            KeyError(PoolName.SWA),
            torch.arange(8),
            torch.arange(4),
        ]
        with self.assertLogs(level="ERROR"):
            self.run_commands(ticket, following)
        failed = c.pp_prefetch_states["hit"].operation
        self.assertTrue(failed.is_terminated())
        c.mem_pool_host.free.assert_called_once_with(kv, pool=PoolName.KV)
        c.page_get_func = Mock(return_value=1)
        c.storage_backend = Mock()
        c.storage_backend.batch_get_v2.return_value = {"swa": [True]}
        with (
            patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1),
            patch.object(
                c.storage_stop_event, "is_set", side_effect=[False, False, True]
            ),
        ):
            c.prefetch_io_aux_func()
        acks = [c.prefetch_sync_queue.get_nowait() for _ in range(8)]
        self.assertEqual([a.rid for a in acks], ["hit"] * 4 + ["next"] * 4)
        self.assertEqual(
            [a.completed_tokens for a in acks], [0, 0, None, None, 4, 8, None, None]
        )
        self.assertEqual((acks[2].pool_hits, acks[6].pool_hits), ({}, {"swa": 1}))
        self.assertEqual(
            c.page_get_func.call_count, 2
        )  # No reads for the failed ticket.
        self.sync_acks(*acks)
        self.assertTrue(c.is_pp_prefetch_ready("hit"))
        self.assertEqual(c.pp_prefetch_states["next"].operation.completed_tokens, 8)
        self.assertEqual(c.prefetch_tokens_occupied, 8)

    def test_cancel_before_ticket_defers_free_until_final_ack(self):
        c, cache = self.c, self.cache
        self.submit(pools=[PoolTransfer(PoolName.SWA, host_indices=torch.arange(4))])
        ticket = c.pp_prefetch_states.pop("hit").ticket
        c.pp_rank = 1
        cache.bind_prefetch_ticket("hit")
        self.assertTrue(c.release_pp_prefetch("hit"))
        self.assertTrue(c.release_pp_prefetch("hit"))
        self.assertIs(c.pp_prefetch_decisions["hit"], PPPrefetchDecision.CANCELLED)
        self.run_commands(ticket)
        operation = c.prefetch_buffer.get_nowait()
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=4))
        c.mem_pool_host.free.assert_not_called()
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        self.assertEqual(
            [call.kwargs["pool"] for call in c.mem_pool_host.free.call_args_list],
            [PoolName.KV, PoolName.SWA],
        )
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        self.assertEqual(c.pp_prefetch_states, {})
        self.assertEqual(c.pp_prefetch_decisions, {})

    def test_failed_source_allocation_does_not_free_borrowed_sidecar_early(self):
        c = self.c
        borrowed = PoolTransfer(PoolName.SWA, host_indices=torch.arange(4))
        operation = self.submit(pools=[borrowed]).operation
        c.mem_pool_host.alloc.side_effect = lambda *args, **kwargs: None
        self.run_commands()
        self.assertTrue(operation.is_terminated())
        c.mem_pool_host.free.assert_not_called()
        self.sync_acks(
            PrefetchAck("hit", operation, completed_tokens=0, completed_req=True)
        )
        c.take_ready_pp_prefetch("hit")
        self.assertIsNone(borrowed.host_indices)
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs["pool"], PoolName.SWA)

    def test_command_failure_finishes_queue_task_and_is_reported_on_poll(self):
        c = self.c
        with (
            patch.object(
                torch.distributed,
                "broadcast_object_list",
                side_effect=RuntimeError("broken"),
            ),
            self.assertLogs(level="ERROR"),
        ):
            worker = threading.Thread(
                target=c.pp_prefetch_command_thread_func, daemon=True
            )
            c.pp_prefetch_command_thread = worker
            worker.start()
            self.submit()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(c.pp_prefetch_command_queue.unfinished_tasks, 0)
        for rid in ("hit", "next"):
            if rid == "next":
                self.submit(rid)
            with self.assertRaisesRegex(RuntimeError, "ticket thread exited"):
                c.is_pp_prefetch_ready(rid)


if __name__ == "__main__":
    unittest.main()
