"""CPU regressions for PP ticket ordering, admission, and buffer ownership."""

import pickle
import threading
import unittest
from array import array
from contextlib import nullcontext
from queue import Empty, Queue
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.managers.cache_controller import PrefetchAck
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PPPrefetchDecision,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPPPrefetchTicket(unittest.TestCase):
    def setUp(self):
        self.c = c = HybridCacheController.__new__(HybridCacheController)
        c.page_size = c.prefetch_threshold = 4
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
        c.mem_pool_host.layout_lease.side_effect = nullcontext
        c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
        c._storage_hit_query = Mock(return_value=(["h0", "h1"], 8))
        c._all_reduce = Mock()
        ranks = {"pp": [0, 2, 4, 6], "tp": [0, 1], "command": [0, 2, 4, 6]}
        patcher = patch.object(
            torch.distributed, "get_process_group_ranks", side_effect=ranks.__getitem__
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(
            torch.distributed, "get_rank", side_effect=lambda: c.pp_rank * 2 + c.tp_rank
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
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache.linker = None
        cache.root_node_handle = Mock(return_value=0)
        cache.buffer_pipeline = Mock(spec=BufferModePipeline)
        cache._handle_prefetch_result = Mock()

    def submit(self, rid="hit", pools=None, attempt_id=0, assume_stored=False):
        key = RadixKey(
            array("q", range(9)), "adapter", is_bigram=True, cache_salt="tenant"
        )
        return self.c.submit_prefetch(
            CacheRequestHandle(rid, attempt_id),
            key,
            "ab" * 32,
            ["prefix"],
            [10, 11, 12, 13],
            pools,
            assume_stored=assume_stored,
        )

    def run_commands(self, *tickets):
        commands = iter((*tickets, None))
        self.c.pp_prefetch_command_queue.put(None)

        def broadcast(objects, *args, **kwargs):
            return [next(commands)] if self.c.pp_rank else objects

        with patch(f"{HybridCacheController.__module__}.broadcast_pyobj", broadcast):
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
                handle = CacheRequestHandle(rid, 0)
                c._storage_hit_query.return_value = ([], hit)
                self.assertEqual(self.submit(rid).decision, bool(hit))
                queries = c._storage_hit_query.call_count
                for _ in range(2):
                    self.assertEqual(
                        self.cache.prefetch_from_storage(handle, 0, []), bool(hit)
                    )
                self.assertEqual(c._storage_hit_query.call_count, queries)
                if hit:
                    state = c.pp_prefetch_states[rid]
                    state.ready_event.set()
                    c.take_ready_pp_prefetch(rid)
                    self.assertFalse(
                        self.cache.prefetch_from_storage(
                            CacheRequestHandle(rid, 1), 0, []
                        )
                    )
                self.assertTrue(self.cache.check_prefetch_progress(handle))
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
            self.assertTrue(self.submit(assume_stored=True).decision)
            self.assertTrue(self.submit("next").decision)
        self.assertEqual(c.pp_prefetch_command_queue.qsize(), 2)
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        c.mem_pool_host.alloc.assert_not_called()
        self.assertFalse(c.pp_prefetch_states["hit"].operation.assume_stored)
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
        result = self.submit(attempt_id=3, assume_stored=True)
        self.assertIsNone(result.decision)
        self.assertEqual(result.operation.handle, CacheRequestHandle("hit", 3))
        self.assertTrue(result.operation.assume_stored)
        self.assertIs(c.prefetch_queue.get_nowait(), result.operation)
        c._storage_hit_query.assert_not_called()

    def test_downstream_request_and_ticket_order_preserves_pp0_admission(self):
        c, cache = self.c, self.cache
        self.submit(attempt_id=3)
        handle = CacheRequestHandle("hit", 3)
        ticket = pickle.loads(pickle.dumps(c.pp_prefetch_states["hit"].ticket))
        ticket.last_hash = None
        c.pp_rank = 1
        c.pp_prefetch_states.clear()
        cache.bind_prefetch_ticket("hit")
        self.assertFalse(cache.check_prefetch_progress(handle))
        c._storage_hit_query.reset_mock()
        self.run_commands(ticket)
        operation = c.prefetch_buffer.get_nowait()
        self.assertEqual(operation.handle, handle)
        self.assertEqual(
            operation.hash_value, get_storage_hash_str(ticket.prefetch_key, page_size=4)
        )
        self.assertTrue(operation.token_ids.is_bigram)
        self.assertEqual(len(operation.token_ids), 8)
        c._storage_hit_query.assert_not_called()
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=8))
        self.assertFalse(c.is_pp_prefetch_ready("hit"))
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        self.assertFalse(cache.check_prefetch_progress(handle))  # PP0 has not admitted.
        cache._all_reduce.side_effect = lambda tensor, _: tensor.fill_(1)
        self.assertTrue(cache.check_prefetch_progress(handle))
        cache._handle_prefetch_result.assert_called_once_with(operation)
        cache.buffer_pipeline.try_lock_anchor.assert_called_once_with(handle, 8)
        key = cache.ongoing_prefetch[handle].prefetch_key
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
        following.handle = CacheRequestHandle("next", 0)
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
        cache.finish(ticket.handle, CacheRequestOutcome.ABORT)
        cache.finish(ticket.handle, CacheRequestOutcome.ABORT)
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

    def test_lazy_sidecars_use_hit_pages_and_pool_page_size(self):
        c = self.c
        c.mem_pool_host.get_pool.side_effect = lambda name: Mock(
            page_size=1 if name == PoolName.MAMBA else 4
        )
        self.submit(
            pools=[
                PoolTransfer(PoolName.SWA, keys=["pending"] * 3),
                PoolTransfer(PoolName.MAMBA, keys=["pending"]),
                PoolTransfer(PoolName.DRAFT_SWA, indices_from_pool=PoolName.SWA),
            ]
        )
        ticket = pickle.loads(pickle.dumps(c.pp_prefetch_states["hit"].ticket))
        for rank in (0, 1):
            with self.subTest(rank=rank):
                c.pp_rank = rank
                if rank:
                    c.pp_prefetch_states.clear()
                c.mem_pool_host.alloc.reset_mock()
                self.run_commands(ticket)
                operation = c.prefetch_buffer.get_nowait()
                self.assertEqual(
                    c.mem_pool_host.alloc.call_args_list,
                    [
                        call(8, pool=PoolName.KV),
                        call(8, pool=PoolName.SWA),
                        call(1, pool=PoolName.MAMBA),
                    ],
                )
                swa, mamba, draft_swa = operation.pool_transfers
                self.assertIs(draft_swa.host_indices, swa.host_indices)
                self.assertEqual(mamba.host_indices.numel(), 1)

        # Missing pool metadata also rolls back KV, before a failed-ticket ACK.
        c.pp_prefetch_states.clear()
        c.mem_pool_host.get_pool.side_effect = KeyError(PoolName.SWA)
        with self.assertLogs(level="ERROR"):
            self.run_commands(ticket)
        self.assertTrue(c.prefetch_buffer.get_nowait().is_terminated())
        self.assertEqual(c.mem_pool_host.free.call_count, 1)
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs["pool"], PoolName.KV)

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
            patch(
                f"{HybridCacheController.__module__}.broadcast_pyobj",
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

    def test_idle_source_broadcasts_empty_then_processes_ticket_and_stop(self):
        c = self.c
        c.tp_rank = 1  # The source is a global rank, not pp_rank=0.
        operation = self.submit().operation
        c.pp_prefetch_command_queue.put(None)
        get = c.pp_prefetch_command_queue.get
        idle_polls = [True, True]

        def get_after_idle(*, timeout):
            self.assertEqual(timeout, 60)
            if idle_polls:
                idle_polls.pop()
                raise Empty
            return get(block=False)

        with (
            patch.object(c.pp_prefetch_command_queue, "get", get_after_idle),
            patch.object(
                torch.distributed, "get_process_group_ranks", return_value=[1, 3, 5, 7]
            ),
            patch(
                f"{HybridCacheController.__module__}.broadcast_pyobj",
                side_effect=lambda objects, *args, **kwargs: objects,
            ) as broadcast,
        ):
            c.pp_prefetch_command_thread_func()
        calls = broadcast.call_args_list
        self.assertEqual([len(c.args[0]) for c in calls], [0, 0, 1, 1])
        for invocation in calls:
            self.assertEqual(invocation.args[1:], (1, "command"))
            self.assertEqual(invocation.kwargs, {"src": 1})
        self.assertEqual(c.pp_prefetch_command_queue.unfinished_tasks, 0)
        self.assertTrue(c.pp_prefetch_command_queue.empty())
        self.assertIs(c.prefetch_buffer.get_nowait(), operation)
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_called_once()

    def test_idle_downstream_skips_empty_broadcasts_before_ticket_and_stop(self):
        c = self.c
        self.submit()
        ticket = c.pp_prefetch_states.pop("hit").ticket
        c.pp_rank = 1
        with (
            patch(
                f"{HybridCacheController.__module__}.broadcast_pyobj",
                side_effect=[[], [], [ticket], [None]],
            ),
            patch.object(c.pp_prefetch_command_queue, "task_done") as task_done,
        ):
            c.pp_prefetch_command_thread_func()
        self.assertEqual(c.prefetch_buffer.get_nowait().request_id, "hit")
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_called_once()
        task_done.assert_not_called()


if __name__ == "__main__":
    unittest.main()
