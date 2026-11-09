"""CPU regressions: buffer-mode HiCache under PP keeps every rank's admission
inputs identical (drain counts, ticket staging)."""

import threading
import unittest
from array import array
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_batch_fingerprint,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _controller(pp_rank: int) -> HybridCacheController:
    c = HybridCacheController.__new__(HybridCacheController)
    c.page_size = c.prefetch_threshold = 4
    c.storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=pp_rank,
        pp_size=2,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name=None,
    )
    c.pp_group, c.pp_prefetch_command_group = "pp", "command"
    c.pp_prefetch_command_thread = None
    c.prefetch_hits_sync_groups = ["pp"]
    c.pp_prefetch_states, c.pp_prefetch_decisions = {}, {}
    c.pp_prefetch_state_lock = threading.Lock()
    c.pp_prefetch_command_queue = Queue()
    c.prefetch_buffer = Queue()
    c.host_mem_release_queue = Queue()
    c.prefetch_tokens_occupied = 0
    c.mem_pool_host = Mock(page_size=4)
    c.mem_pool_host.available_size.return_value = 1 << 20
    c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
    c.storage_backend_type = "mooncake"

    def query(operation):
        hashes = get_storage_hash_str(operation.token_ids, None, page_size=4)
        operation.all_hash_values = hashes
        operation.query_pool_hit_pages = {PoolName.KV: 2}
        return hashes[:2], 8

    c._storage_hit_query = Mock(side_effect=query)
    c._all_reduce = Mock()
    return c


def _run_tickets(c: HybridCacheController, *tickets) -> None:
    commands = iter((*tickets, None))
    c.pp_prefetch_command_queue.put(None)

    def broadcast(objects, *args, **kwargs):
        return [next(commands)] if c.storage_config.pp_rank else objects

    with (
        patch(f"{HybridCacheController.__module__}.broadcast_pyobj", broadcast),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_process_group_ranks", return_value=[0]),
    ):
        c.pp_prefetch_command_thread_func()


def _submit(c: HybridCacheController, rid: str):
    return c.submit_prefetch(
        CacheRequestHandle(rid, 0),
        RadixKey(array("q", range(9))),
        None,
        None,
        [],
        None,
    )


class TestReadyCountsFollowPP0(unittest.TestCase):
    """Bug 1: stage-local drain counts let each stage free staging and device
    locks at its own step, so admission builds different batches."""

    def _cache(self, pp_rank: int) -> UnifiedRadixCache:
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.pp_rank, cache.pp_size = pp_rank, 4
        cache.host_memory_mode = "buffer_only"
        cache.tree_core = SimpleNamespace(
            write_back_duplicate_reclaim_digest=0, enable_storage=True
        )
        cc = SimpleNamespace(
            ack_write_queue=[SimpleNamespace(finish_event=Mock())],
            ack_load_queue=[],
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
        )
        cc.ack_write_queue[0].finish_event.query.return_value = True
        cache.cache_controller = cc
        cache._all_reduce = Mock()
        cache._all_reduce_attn_groups = Mock()
        return cache

    def test_later_stage_takes_pp0_counts_not_its_own(self):
        cache = self._cache(pp_rank=1)
        received = [3, 2, 0, 0, 5, 0, 0, 0]

        def from_pp0(tensor, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            # A later stage contributes nothing: it adopts PP0's counts.
            self.assertEqual(tensor[:6].tolist(), [0] * 6)
            tensor[:6] = torch.tensor(received[:6])

        cache._all_reduce.side_effect = from_pp0
        writes, loads, storage, _ = cache._sync_hicache_ready_counts()
        cache._all_reduce_attn_groups.assert_not_called()
        self.assertEqual((writes, loads, storage), (3, 2, (0, 0, 5, 0)))

    def test_pp0_counts_its_own_ready_acks(self):
        cache = self._cache(pp_rank=0)
        cache._all_reduce.side_effect = lambda tensor, op: None
        writes, _, _, _ = cache._sync_hicache_ready_counts()
        self.assertEqual(writes, 1)

    def test_missing_ack_on_a_later_stage_is_reported(self):
        cache = self._cache(pp_rank=1)
        with self.assertRaisesRegex(RuntimeError, "write acks diverged"):
            cache.writing_check(finish_count=2)


class TestTicketStagingOnSchedulerStep(unittest.TestCase):
    """Bug 1: ticket staging used to be allocated on the worker thread, so a
    rank's host availability depended on when its worker ran."""

    def test_worker_leaves_host_memory_to_the_agreed_step(self):
        c = _controller(pp_rank=0)
        _submit(c, "r")
        _run_tickets(c)
        c.mem_pool_host.alloc.assert_not_called()
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        self.assertEqual(c.pp_prefetch_phase("r"), 1)
        self.assertTrue(c.allocate_pp_prefetch("r"))
        c.start_pp_prefetch_io("r", True, True)
        self.assertEqual(c.prefetch_tokens_occupied, 8)
        self.assertEqual(c.prefetch_buffer.qsize(), 1)

    def test_scheduler_follows_pp0_phase_and_agreed_staging(self):
        c = _controller(pp_rank=1)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.pp_rank = 1
        cache.cache_controller = c
        pp0 = _controller(pp_rank=0)
        _submit(pp0, "r")
        ticket = pp0.pp_prefetch_states["r"].ticket
        _run_tickets(c, ticket)
        verdicts = iter((1, 1))
        cache._all_reduce = Mock(
            side_effect=lambda tensor, op: tensor.fill_(next(verdicts))
        )
        self.assertFalse(cache._check_pp_prefetch_progress(CacheRequestHandle("r", 0)))
        c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
        self.assertEqual(c.prefetch_buffer.qsize(), 1)


class TestPPBatchFingerprint(unittest.TestCase):
    def _batch(self, rids, extend, prefix):
        return SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in rids],
            extend_num_tokens=extend,
            prefix_lens=prefix,
        )

    def test_fingerprint_separates_batches(self):
        base = _pp_batch_fingerprint(self._batch(["a", "b"], 16384, [0, 64]))
        for other in (
            self._batch(["a", "b"], 16300, [0, 64]),
            self._batch(["a", "b"], 16384, [0, 128]),
            self._batch(["a", "c"], 16384, [0, 64]),
            self._batch(["a"], 16384, [0, 64]),
        ):
            self.assertNotEqual(base, _pp_batch_fingerprint(other))

    def test_later_stage_refuses_rows_built_for_another_batch(self):
        scheduler = SimpleNamespace(
            forward_stream_ctx=nullcontext(),
            _pp_wait_forward_dependencies=Mock(),
            run_batch=Mock(),
            scheduler_stage_metrics=None,
            pp_group=SimpleNamespace(is_last_rank=False, rank_in_group=1),
        )
        local = self._batch(["a"], 16384, [0])
        sent = _pp_batch_fingerprint(self._batch(["a"], 16300, [84]))
        proxy = SimpleNamespace(tensors={"__pp_batch_fingerprint__": sent})
        with self.assertRaisesRegex(RuntimeError, "different batches"):
            SchedulerPPMixin._pp_launch_batch(scheduler, 0, local, proxy, [None], None)
        scheduler.run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
