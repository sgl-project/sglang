"""CPU regressions for PP ticket ordering, admission, and buffer ownership."""

import pickle
import tempfile
import threading
import unittest
from array import array
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PPPrefetchDecision,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=19, suite="base-a-test-cpu")


class TestPPPrefetchTicket(unittest.TestCase):
    def setUp(self):
        self.c = c = HybridCacheController.__new__(HybridCacheController)
        c.page_size = c.prefetch_threshold = 4
        c.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=4,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=None,
        )
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
        c.mem_pool_host = Mock(page_size=4, entry_map={})
        c.mem_pool_host.available_size.return_value = 1 << 20
        c.mem_pool_host.layout_lease.side_effect = nullcontext
        c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
        c.storage_backend_type = "mooncake"
        c.storage_backend = Mock()
        c._storage_hit_query = Mock(side_effect=self.query_hits)
        c._all_reduce = Mock()
        ranks = {"pp": [0, 2, 4, 6], "tp": [0, 1], "command": [0, 2, 4, 6]}
        patcher = patch.object(
            torch.distributed, "get_process_group_ranks", side_effect=ranks.__getitem__
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(
            torch.distributed,
            "get_rank",
            side_effect=lambda: c.storage_config.pp_rank * 2 + c.storage_config.tp_rank,
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
        cache._storage_prefetch_wait = {}
        cache.linker = None
        cache.root_node_handle = Mock(return_value=0)
        cache.buffer_pipeline = Mock(spec=BufferModePipeline)
        cache._handle_prefetch_result = Mock()

    def query_hits(self, operation, hit=8):
        hashes = get_storage_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.c.page_size
        )
        operation.all_hash_values = hashes
        operation.query_pool_hit_pages = {PoolName.KV: hit // self.c.page_size}
        return hashes[: hit // self.c.page_size], hit

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
            return [next(commands)] if self.c.storage_config.pp_rank else objects

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

    def run_io(self, operations):
        with (
            patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1),
            patch.object(
                self.c.storage_stop_event,
                "is_set",
                side_effect=[False] * operations + [True],
            ),
        ):
            self.c.prefetch_io_aux_func()
        acks = []
        while not self.c.prefetch_sync_queue.empty():
            acks.append(self.c.prefetch_sync_queue.get_nowait())
        return acks

    def stage(self, rid, agreed=True):
        """The scheduler's agreed staging step for one ticket."""
        allocated = self.c.allocate_pp_prefetch(rid)
        self.c.start_pp_prefetch_io(rid, agreed and allocated, allocated)
        return allocated

    def test_hit_and_miss_are_reused_across_enqueue_retract_and_retry(self):
        c, cache = self.c, self.cache
        cache.pp_rank = 0
        c.page_get_func = Mock(return_value=1)
        for hit in (0, 8):
            with self.subTest(hit=hit):
                rid = str(hit)
                handle = CacheRequestHandle(rid, 0)
                c._storage_hit_query.side_effect = lambda operation: self.query_hits(
                    operation, hit
                )
                self.assertTrue(self.submit(rid).decision)
                queries = c._storage_hit_query.call_count
                for _ in range(2):
                    self.assertTrue(cache.prefetch_from_storage(handle, 0, []))
                self.assertEqual(c._storage_hit_query.call_count, queries)
                self.run_commands()
                self.assertEqual(c._storage_hit_query.call_count, queries + 1)
                if hit:
                    # The scheduler stages the agreed hit, then waits for the GET.
                    self.assertFalse(cache.check_prefetch_progress(handle))
                    self.sync_acks(*self.run_io(1))
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertFalse(
                    cache.prefetch_from_storage(CacheRequestHandle(rid, 1), 0, [])
                )
                self.assertFalse(c.release_pp_prefetch(rid))
                self.assertIsNone(c.get_prefetch_submission(rid))
        self.assertTrue(c.pp_prefetch_command_queue.empty())

    def test_all_backends_query_before_one_worker_consensus_without_staging(self):
        """The worker agrees on a hit but never touches host memory: staging
        is a scheduler-thread step, so host state cannot differ by timing."""
        c = self.c
        for backend in ("file", "mooncake", "dynamic"):
            with self.subTest(backend=backend):
                c.storage_backend_type = backend
                c._storage_hit_query.reset_mock()
                c._all_reduce.reset_mock()
                c.mem_pool_host.alloc.reset_mock()
                c._storage_hit_query.side_effect = AssertionError(
                    "scheduler queried storage"
                )
                c._all_reduce.side_effect = AssertionError(
                    "scheduler entered a collective"
                )
                with patch.object(
                    c.pp_prefetch_command_queue,
                    "join",
                    side_effect=AssertionError("scheduler blocked"),
                ):
                    first = self.submit(backend, assume_stored=True)
                    second = self.submit(f"next_{backend}")
                self.assertTrue(first.decision)
                self.assertTrue(second.decision)
                self.assertEqual(c.pp_prefetch_command_queue.qsize(), 2)
                self.assertEqual(c.pp_prefetch_phase(backend), 0)
                self.assertFalse(first.operation.assume_stored)
                c._storage_hit_query.assert_not_called()
                c._all_reduce.assert_not_called()

                votes = []

                def reduce_hits(tensor, op, groups):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    self.assertEqual(groups, c.prefetch_hits_sync_groups)
                    self.assertEqual(tensor.numel(), 1 + len(PoolName))
                    self.assertEqual(tensor[0].item(), 8)
                    votes.append(tensor[0].item())

                c._storage_hit_query.side_effect = self.query_hits
                c._all_reduce.side_effect = reduce_hits
                self.run_commands()
                self.assertEqual(
                    c._storage_hit_query.call_args_list,
                    [call(first.operation), call(second.operation)],
                )
                self.assertEqual(votes, [8, 8])
                c.mem_pool_host.alloc.assert_not_called()
                self.assertTrue(c.prefetch_buffer.empty())
                self.assertEqual(c.pp_prefetch_phase(backend), 1)
                self.assertEqual(c.pp_prefetch_phase(f"next_{backend}"), 1)
                self.assertTrue(self.stage(backend))
                self.assertTrue(self.stage(f"next_{backend}"))
                self.assertIs(c.prefetch_buffer.get_nowait(), first.operation)
                self.assertIs(c.prefetch_buffer.get_nowait(), second.operation)
                self.assertEqual(c.pp_prefetch_phase(backend), 0)

    def test_file_attachment_enables_pp_tickets(self):
        c = self.c
        c.pp_prefetch_command_group = None
        c.storage_host_pool = Mock()
        c.host_memory_mode = "buffer_only"
        c.storage_backend_type = "file"
        with (
            patch.object(HiCacheController, "attach_storage_backend"),
            patch.object(torch.distributed, "get_world_size", return_value=4),
            patch(
                "sglang.srt.distributed.parallel_state.create_custom_parallel_group",
                return_value="command",
            ),
        ):
            c.attach_storage_backend("file", prefetch_threshold=4)
        self.assertEqual(c.pp_prefetch_command_group, "command")

    def test_file_stage_hits_agree_before_scheduler_staging(self):
        """PP0's local files cannot reveal a shorter prefix on another stage;
        each stage stages only the agreed prefix, after the worker's vote."""
        c = self.c
        key = RadixKey(
            array("q", range(9)), "adapter", is_bigram=True, cache_salt="tenant"
        )
        hashes = get_storage_hash_str(key, "ab" * 32, page_size=4)
        local_votes = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, "get", return_value=None
            ),
        ):
            backends = []
            for rank, pages in ((0, 2), (1, 1)):
                config = replace(
                    c.storage_config,
                    pp_rank=rank,
                    pp_size=2,
                    extra_config={
                        "file_storage_path": str(Path(directory) / str(rank)),
                        "enable_metadata_cache": False,
                        "max_size": 0,
                        "min_free_space": 0,
                    },
                )
                backend = HiCacheFile(config)
                for hash_value in hashes[:pages]:
                    self.assertTrue(backend.set(hash_value, torch.ones(1)))
                backends.append(backend)

            self.assertEqual(backends[0].batch_exists(hashes), 2)
            self.assertEqual(backends[1].batch_exists(hashes), 1)
            c.storage_backend = backends[0]
            c.storage_backend_type = "file"
            c._storage_hit_query = Mock(
                wraps=HybridCacheController._storage_hit_query.__get__(c)
            )
            result = self.submit(attempt_id=3)
            ticket = pickle.loads(pickle.dumps(c.pp_prefetch_states["hit"].ticket))

            for rank, peer_hit in ((0, 4), (1, 8)):
                with self.subTest(rank=rank):
                    c.storage_config.pp_rank = rank
                    c.storage_backend = backends[rank]
                    if rank:
                        c.pp_prefetch_states.clear()
                    c._storage_hit_query.reset_mock()
                    c.mem_pool_host.alloc.reset_mock()
                    c.mem_pool_host.free.reset_mock()
                    c.prefetch_tokens_occupied = 0

                    def reduce_hits(tensor, op, groups):
                        self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                        self.assertEqual(groups, ["pp", "tp"])
                        self.assertEqual(tensor.numel(), 1 + len(PoolName))
                        local_votes.append(tensor[0].item())
                        c.mem_pool_host.alloc.assert_not_called()
                        tensor[0] = min(tensor[0].item(), peer_hit)
                        kv_index = 1 + list(PoolName).index(PoolName.KV)
                        tensor[kv_index] = min(tensor[kv_index].item(), peer_hit // 4)

                    c._all_reduce.side_effect = reduce_hits
                    self.run_commands(ticket)
                    self.assertTrue(c.prefetch_buffer.empty())
                    self.assertTrue(self.stage("hit"))
                    operation = c.prefetch_buffer.get_nowait()
                    self.assertEqual(operation.handle, CacheRequestHandle("hit", 3))
                    self.assertEqual(operation.storage_hit_count, 4)
                    self.assertEqual(operation.hash_value, hashes[:1])
                    self.assertEqual(operation.host_indices.numel(), 4)
                    self.assertFalse(operation.is_terminated())
                    self.assertEqual(c.prefetch_tokens_occupied, 4)
                    c.mem_pool_host.alloc.assert_called_once_with(4, pool=PoolName.KV)
                    c.mem_pool_host.free.assert_not_called()
                    if rank == 0:
                        self.assertIs(operation, result.operation)
                    c._storage_hit_query.assert_called_once_with(operation)
                    self.assertEqual(operation.query_pool_hit_pages, {PoolName.KV: 1})

        self.assertEqual(local_votes, [8, 4])

    def test_file_stage_hit_fits_capacity_before_consensus(self):
        """A large local hit must not discard a prefix that fits on every stage."""
        c = self.c
        c.page_size = c.prefetch_threshold = 256
        key = RadixKey(array("q", range(8193)), is_bigram=True)
        hashes = get_storage_hash_str(key, page_size=256)
        c.storage_backend_type = "file"
        c._storage_hit_query = Mock(
            wraps=HybridCacheController._storage_hit_query.__get__(c)
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, "get", return_value=None
            ),
        ):
            backends = []
            for rank, pages in ((0, 32), (1, 8)):
                config = replace(
                    c.storage_config,
                    pp_rank=rank,
                    pp_size=2,
                    extra_config={
                        "file_storage_path": str(Path(directory) / str(rank)),
                        "enable_metadata_cache": False,
                        "max_size": 0,
                        "min_free_space": 0,
                    },
                )
                backend = HiCacheFile(config)
                for hash_value in hashes[:pages]:
                    self.assertTrue(backend.set(hash_value, torch.ones(1)))
                backends.append(backend)

            for capacity in (4096, 4097):
                c.storage_config.pp_rank = 0
                c.pp_prefetch_states.clear()
                c.pp_prefetch_command_queue = Queue()
                result = c.submit_prefetch(
                    CacheRequestHandle("capacity", 0), key, None, None, [], None
                )
                ticket = pickle.loads(
                    pickle.dumps(c.pp_prefetch_states["capacity"].ticket)
                )
                for rank, local_vote in ((0, 4096), (1, 2048)):
                    with self.subTest(capacity=capacity, rank=rank):
                        c.storage_config.pp_rank = rank
                        c.storage_backend = backends[rank]
                        if rank:
                            c.pp_prefetch_states.clear()
                        host_pool = LogicalHostPool(4096, page_size=256)
                        c.mem_pool_host.available_size.return_value = capacity
                        c.mem_pool_host.available_size.reset_mock()
                        c.mem_pool_host.alloc.reset_mock()
                        c.mem_pool_host.free.reset_mock()
                        c.mem_pool_host.alloc.side_effect = lambda size, **_: (
                            host_pool.alloc(size)
                        )
                        c.mem_pool_host.free.side_effect = lambda indices, **_: (
                            host_pool.free(indices)
                        )
                        c.prefetch_tokens_occupied = 0
                        c._storage_hit_query.reset_mock()
                        c._all_reduce.reset_mock()
                        local_votes = []

                        def reduce_hits(tensor, op, groups):
                            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                            self.assertEqual(groups, c.prefetch_hits_sync_groups)
                            self.assertEqual(tensor.numel(), 1 + len(PoolName))
                            local_votes.append(tensor[0].item())
                            kv_index = 1 + list(PoolName).index(PoolName.KV)
                            self.assertEqual(
                                tensor[kv_index].item(), 32 if rank == 0 else 8
                            )
                            peer_vote = 2048 if rank == 0 else 4096
                            tensor[0] = min(tensor[0].item(), peer_vote)
                            tensor[kv_index] = min(tensor[kv_index].item(), 8)

                        c._all_reduce.side_effect = reduce_hits
                        self.run_commands(ticket)
                        self.assertTrue(self.stage("capacity"))
                        operation = c.prefetch_buffer.get_nowait()
                        if rank == 0:
                            self.assertIs(operation, result.operation)
                        self.assertEqual(operation.storage_hit_count, 2048)
                        self.assertEqual(local_votes, [local_vote])
                        c.mem_pool_host.alloc.assert_called_once_with(
                            2048, pool=PoolName.KV
                        )
                        self.assertEqual(operation.hash_value, hashes[:8])
                        self.assertEqual(operation.host_indices.numel(), 2048)
                        self.assertFalse(operation.is_terminated())
                        self.assertEqual(
                            operation.query_pool_hit_pages, {PoolName.KV: 8}
                        )
                        self.assertEqual(c.prefetch_tokens_occupied, 2048)
                        self.assertEqual(host_pool.available_size(), 2048)
                        c._storage_hit_query.assert_called_once_with(operation)
                        c.mem_pool_host.available_size.assert_called_once_with(
                            pool=PoolName.KV
                        )
                        c._all_reduce.assert_called_once()
                        c.mem_pool_host.free.assert_not_called()

    def test_checkpoint_vote_stays_within_capacity_capped_hit(self):
        """A stage whose hit is capped to free host space must not vote for
        restorable checkpoints past that cap, or the agreed hit outgrows the
        buffers it stages."""
        c = self.c
        stored_pages, free_pages = 8, 2
        c.mem_pool_host.available_size.return_value = free_pages * 4
        c.mem_pool_host.get_pool.side_effect = lambda name: Mock(
            page_size=4 if name == PoolName.KV else 1
        )

        def query(operation):
            hashes = get_storage_hash_str(
                operation.token_ids, operation.last_hash, page_size=4
            )
            operation.all_hash_values = hashes
            operation.query_pool_hit_pages = {PoolName.KV: stored_pages}
            operation.pool_storage_result.restorable_prefix_pages = list(
                range(1, stored_pages + 1)
            )
            return hashes[:stored_pages], stored_pages * 4

        c._storage_hit_query.side_effect = query
        votes = []
        c._all_reduce.side_effect = lambda tensor, *_: votes.append(tensor.tolist())
        kda = PoolTransfer(
            PoolName.MAMBA, keys=["kda"], hit_policy=PoolHitPolicy.TRAILING_PAGES
        )
        c.submit_prefetch(
            CacheRequestHandle("capped", 0),
            RadixKey(array("q", range(stored_pages * 4 + 1))),
            None,
            None,
            [],
            [kda],
        )
        self.run_commands(c.pp_prefetch_states["capped"].ticket)
        self.assertTrue(self.stage("capped"))

        operation = c.prefetch_buffer.get_nowait()
        self.assertEqual(votes[0][: stored_pages + 1], [1, 1, 1] + [0] * 6)
        self.assertEqual(operation.host_indices.numel(), free_pages * 4)
        self.assertEqual(operation.storage_hit_count, free_pages * 4)
        self.assertEqual(len(operation.hash_value), free_pages)
        self.assertFalse(operation.is_terminated())

    def test_local_miss_or_query_error_completes_before_next_ticket(self):
        """A stage miss/error completes on the worker without a GET and
        without host memory, ahead of the next ticket."""
        c, cache = self.c, self.cache
        c.storage_backend_type = "file"
        cache.pp_rank = 0
        for name, local_hit, peer_hit, threshold in (
            ("local_miss", 0, 8, 4),
            ("peer_miss", 8, 0, 4),
            ("below_threshold", 7, 8, 6),
            ("query_error", RuntimeError("local storage unavailable"), 8, 4),
        ):
            with self.subTest(name=name):
                failed_rid, next_rid = name, f"next_{name}"
                c.prefetch_threshold = threshold
                c._storage_hit_query.reset_mock()
                c._all_reduce.reset_mock()
                c.mem_pool_host.alloc.reset_mock()
                c.mem_pool_host.free.reset_mock()
                cache._handle_prefetch_result.reset_mock()
                occupied_before = c.prefetch_tokens_occupied
                hits = iter((local_hit, 8))

                def query_hits(operation):
                    hit = next(hits)
                    if isinstance(hit, Exception):
                        operation.query_pool_hit_pages = {PoolName.KV: 2}
                        raise hit
                    return self.query_hits(operation, hit)

                c._storage_hit_query.side_effect = query_hits
                self.assertTrue(self.submit(failed_rid, attempt_id=3).decision)
                self.assertTrue(self.submit(next_rid).decision)
                peers = iter((peer_hit, 8))
                local_votes = []

                def reduce_hits(tensor, op, groups):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    self.assertEqual(groups, ["pp", "tp"])
                    self.assertEqual(tensor.numel(), 1 + len(PoolName))
                    local_votes.append(tensor[0].item())
                    tensor[0] = min(tensor[0].item(), next(peers))

                c._all_reduce.side_effect = reduce_hits
                logs = (
                    self.assertLogs(level="ERROR")
                    if isinstance(local_hit, Exception)
                    else nullcontext()
                )
                with logs:
                    self.run_commands()
                failed = c.pp_prefetch_states[failed_rid].operation
                following = c.pp_prefetch_states[next_rid].operation
                self.assertEqual(failed.storage_hit_count, 0)
                self.assertEqual(failed.hash_value, [])
                self.assertEqual(failed.host_indices.numel(), 0)
                self.assertTrue(failed.is_terminated())
                self.assertIsNone(following.host_indices)
                self.assertFalse(following.is_terminated())
                self.assertEqual(
                    local_votes,
                    [8 if name == "peer_miss" else 0, 8],
                )
                self.assertEqual(c._all_reduce.call_count, 2)
                self.assertEqual(
                    c._storage_hit_query.call_args_list,
                    [call(failed), call(following)],
                )
                # The worker neither stages nor issues a GET for either ticket.
                c.mem_pool_host.alloc.assert_not_called()
                c.mem_pool_host.free.assert_not_called()
                self.assertTrue(c.prefetch_buffer.empty())
                if name == "query_error":
                    self.assertEqual(failed.query_pool_hit_pages, {})
                self.assertEqual(c.prefetch_tokens_occupied, occupied_before)

                c._all_reduce.side_effect = None
                handle = CacheRequestHandle(failed_rid, 3)
                self.assertEqual(c.pp_prefetch_phase(failed_rid), 2)
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[handle], 0)
                self.assertNotIn(handle, cache.ongoing_prefetch)
                cache._handle_prefetch_result.assert_not_called()
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertFalse(c.release_pp_prefetch(failed_rid))

                next_handle = CacheRequestHandle(next_rid, 0)
                self.assertFalse(cache.check_prefetch_progress(next_handle))
                c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
                self.assertEqual(c.prefetch_tokens_occupied, occupied_before + 8)
                c.page_get_func = Mock(return_value=1)
                acks = self.run_io(1)
                self.assertEqual([ack.rid for ack in acks], [next_rid] * 3)
                self.assertEqual([ack.completed_tokens for ack in acks], [4, 8, None])
                self.sync_acks(*acks)
                self.assertTrue(cache.check_prefetch_progress(next_handle))
                cache._handle_prefetch_result.assert_called_once_with(following)
                self.assertEqual(c._storage_hit_query.call_count, 2)
                self.assertEqual(c.pp_prefetch_command_queue.unfinished_tasks, 0)

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
        c.storage_config.pp_rank = 1
        c.pp_prefetch_states.clear()
        cache.bind_prefetch_ticket("hit")
        self.assertFalse(cache.check_prefetch_progress(handle))
        c._storage_hit_query.reset_mock()
        self.run_commands(ticket)
        # A later stage stages nothing until PP0's agreed step reaches it.
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_not_called()
        self.assertFalse(cache.check_prefetch_progress(handle))
        c.mem_pool_host.alloc.assert_not_called()
        phases = iter((1, 1))
        cache._all_reduce.side_effect = lambda tensor, _: tensor.fill_(next(phases))
        self.assertFalse(cache.check_prefetch_progress(handle))
        c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
        operation = c.prefetch_buffer.get_nowait()
        self.assertEqual(operation.handle, handle)
        self.assertEqual(
            operation.hash_value, get_storage_hash_str(ticket.prefetch_key, page_size=4)
        )
        self.assertTrue(operation.token_ids.is_bigram)
        self.assertEqual(len(operation.token_ids), 8)
        c._storage_hit_query.assert_called_once_with(operation)
        self.sync_acks(PrefetchAck("hit", operation, completed_tokens=8))
        self.assertEqual(c.pp_prefetch_phase("hit"), 0)
        self.sync_acks(PrefetchAck("hit", operation, completed_req=True))
        cache._all_reduce.side_effect = None
        self.assertFalse(cache.check_prefetch_progress(handle))  # PP0 has not admitted.
        cache._all_reduce.side_effect = lambda tensor, _: tensor.fill_(2)
        self.assertTrue(cache.check_prefetch_progress(handle))
        cache._handle_prefetch_result.assert_called_once_with(operation)
        cache.buffer_pipeline.try_lock_anchor.assert_called_once_with(handle, 8)
        key = cache.ongoing_prefetch[handle].prefetch_key
        self.assertEqual((key.extra_key, key.cache_salt), ("adapter", "tenant"))
        self.assertTrue(key.is_bigram)
        self.assertEqual(c.pp_prefetch_phase("hit"), 0)

    def test_staging_divergence_against_pp0_is_reported(self):
        """A later stage that cannot stage what PP0 staged must not run on."""
        c = self.c
        self.submit()
        ticket = c.pp_prefetch_states.pop("hit").ticket
        c.storage_config.pp_rank = 1
        self.run_commands(ticket)
        c.mem_pool_host.alloc.side_effect = lambda *args, **kwargs: None
        allocated = c.allocate_pp_prefetch("hit")
        self.assertFalse(allocated)
        with self.assertRaisesRegex(RuntimeError, "staging diverged"):
            c.start_pp_prefetch_io("hit", True, allocated)

    def test_allocation_error_votes_a_miss_and_keeps_the_next_ticket(self):
        c = self.c
        self.submit(
            pools=[PoolTransfer(PoolName.SWA, keys=["h1"])],
        )
        ticket = c.pp_prefetch_states.pop("hit").ticket
        following = pickle.loads(pickle.dumps(ticket))
        following.handle = CacheRequestHandle("next", 0)
        c.storage_config.pp_rank = 1
        kv = torch.arange(8)
        c.mem_pool_host.alloc.side_effect = [
            kv,
            KeyError(PoolName.SWA),
            torch.arange(8),
            torch.arange(4),
        ]
        c.mem_pool_host.get_pool.side_effect = lambda name: Mock(page_size=4)
        self.run_commands(ticket, following)
        failed = c.pp_prefetch_states["hit"].operation
        next_operation = c.pp_prefetch_states["next"].operation
        self.assertEqual(
            c._storage_hit_query.call_args_list, [call(failed), call(next_operation)]
        )
        with self.assertLogs(level="ERROR"):
            self.assertFalse(c.allocate_pp_prefetch("hit"))
        c.mem_pool_host.free.assert_called_once_with(kv, pool=PoolName.KV)
        c.start_pp_prefetch_io("hit", False, False)
        self.assertTrue(failed.is_terminated())
        self.assertEqual(failed.storage_hit_count, 0)
        self.assertEqual(failed.hash_value, [])
        self.assertEqual(failed.host_indices.numel(), 0)
        self.assertEqual(c.pp_prefetch_phase("hit"), 2)
        self.assertTrue(self.stage("next"))
        self.assertIs(c.prefetch_buffer.get_nowait(), next_operation)
        self.assertTrue(c.prefetch_buffer.empty())
        self.assertEqual(c.prefetch_tokens_occupied, 8)

    def test_cancel_before_ticket_stages_nothing(self):
        c, cache = self.c, self.cache
        self.submit(pools=[PoolTransfer(PoolName.SWA, keys=["h1"])])
        ticket = c.pp_prefetch_states.pop("hit").ticket
        c.storage_config.pp_rank = 1
        cache.bind_prefetch_ticket("hit")
        cache.finish(ticket.handle, CacheRequestOutcome.ABORT)
        cache.finish(ticket.handle, CacheRequestOutcome.ABORT)
        self.assertIs(c.pp_prefetch_decisions["hit"], PPPrefetchDecision.CANCELLED)
        self.run_commands(ticket)
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_not_called()
        c.mem_pool_host.free.assert_not_called()
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        self.assertEqual(c.pp_prefetch_states, {})
        self.assertEqual(c.pp_prefetch_decisions, {})

    def test_release_during_get_frees_through_release_queues(self):
        """A GET finishing on an I/O thread must not free host memory there:
        the pages go to the release queues every rank drains at one step."""
        c = self.c
        c.mem_pool_host.entry_map = {
            PoolName.SWA: Mock(
                is_primary_index_anchor=False, host_pool=Mock(page_size=4)
            )
        }
        c.extra_host_mem_release_queues = {PoolName.SWA: Queue()}
        c.mem_pool_host.get_pool.side_effect = lambda name: Mock(page_size=4)
        operation = self.submit(
            pools=[PoolTransfer(PoolName.SWA, keys=["h1"])]
        ).operation
        self.run_commands()
        self.assertTrue(self.stage("hit"))
        self.assertEqual(c.prefetch_tokens_occupied, 8)
        self.assertTrue(c.release_pp_prefetch("hit"))
        self.assertEqual(c.prefetch_tokens_occupied, 0)
        c.mem_pool_host.free.reset_mock()
        self.sync_acks(
            PrefetchAck("hit", operation, completed_tokens=8),
            PrefetchAck("hit", operation, completed_req=True),
        )
        c.mem_pool_host.free.assert_not_called()
        self.assertEqual(c.host_mem_release_queue.qsize(), 2)
        self.assertEqual(c.extra_host_mem_release_queues[PoolName.SWA].qsize(), 1)
        self.assertNotIn("hit", c.pp_prefetch_states)
        self.assertEqual(c.prefetch_tokens_occupied, 0)

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
                c.storage_config.pp_rank = rank
                if rank:
                    c.pp_prefetch_states.clear()
                c.mem_pool_host.alloc.reset_mock()
                self.run_commands(ticket)
                c.mem_pool_host.alloc.assert_not_called()
                self.assertTrue(self.stage("hit"))
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

        # Missing pool metadata rolls back KV and votes a miss.
        c.pp_prefetch_states.clear()
        c.mem_pool_host.get_pool.side_effect = KeyError(PoolName.SWA)
        c.mem_pool_host.free.reset_mock()
        self.run_commands(ticket)
        with self.assertLogs(level="ERROR"):
            self.assertFalse(self.stage("hit"))
        self.assertTrue(c.prefetch_buffer.empty())
        self.assertTrue(c.pp_prefetch_states["hit"].operation.is_terminated())
        self.assertEqual(c.mem_pool_host.free.call_count, 1)
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs["pool"], PoolName.KV)

    def test_failed_source_allocation_does_not_free_borrowed_sidecar_early(self):
        c = self.c
        borrowed = PoolTransfer(PoolName.SWA, host_indices=torch.arange(4))
        operation = self.submit(pools=[borrowed]).operation
        c.mem_pool_host.alloc.side_effect = lambda *args, **kwargs: None
        self.run_commands()
        c.mem_pool_host.alloc.assert_not_called()
        self.assertFalse(self.stage("hit"))
        self.assertTrue(operation.is_terminated())
        self.assertEqual(operation.hash_value, [])
        self.assertEqual(operation.storage_hit_count, 0)
        c._storage_hit_query.assert_called_once_with(operation)
        c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
        c._all_reduce.assert_called_once()
        packed, op, groups = c._all_reduce.call_args.args
        self.assertEqual(packed[0].item(), 8)
        self.assertEqual(op, torch.distributed.ReduceOp.MIN)
        self.assertEqual(groups, c.prefetch_hits_sync_groups)
        c.mem_pool_host.free.assert_not_called()
        c.take_ready_pp_prefetch("hit")
        self.assertIsNone(borrowed.host_indices)
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs["pool"], PoolName.SWA)

    def test_peer_vote_sizes_staging_and_keeps_borrowed_sidecar(self):
        """The agreed vote sizes the staging; a borrowed sidecar is freed once,
        by its owner, never by the KV path."""
        c = self.c
        for peer_vote in (4, 0):
            with self.subTest(peer_vote=peer_vote):
                rid = f"peer_{peer_vote}"
                c._storage_hit_query.reset_mock()
                c._all_reduce.reset_mock()
                c.mem_pool_host.alloc.reset_mock()
                c.mem_pool_host.free.reset_mock()
                c.prefetch_tokens_occupied = 0
                borrowed_indices = torch.arange(4)
                operation = self.submit(
                    rid,
                    pools=[
                        PoolTransfer(PoolName.SWA, host_indices=borrowed_indices),
                        PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV),
                    ],
                ).operation

                def reduce_hits(tensor, op, groups):
                    self.assertEqual(tensor[0].item(), 8)
                    c.mem_pool_host.alloc.assert_not_called()
                    tensor[0] = peer_vote

                c._all_reduce.side_effect = reduce_hits
                self.run_commands()
                c._all_reduce.side_effect = None
                swa, indexer = operation.pool_transfers
                self.assertEqual(operation.storage_hit_count, peer_vote)
                self.assertEqual(operation.is_terminated(), peer_vote == 0)
                if peer_vote:
                    self.assertTrue(self.stage(rid))
                    c.mem_pool_host.alloc.assert_called_once_with(4, pool=PoolName.KV)
                    self.assertEqual(operation.host_indices.numel(), 4)
                    swa, indexer = operation.pool_transfers
                    self.assertIs(indexer.host_indices, operation.host_indices)
                    self.assertIs(indexer.keys, operation.hash_value)
                    self.assertEqual(c.prefetch_tokens_occupied, 4)
                else:
                    c.mem_pool_host.alloc.assert_not_called()
                    self.assertEqual(c.pp_prefetch_phase(rid), 2)
                self.assertIs(swa.host_indices, borrowed_indices)
                c.mem_pool_host.free.assert_not_called()
                if not peer_vote:
                    c.take_ready_pp_prefetch(rid)
                    self.assertIsNone(swa.host_indices)
                    self.assertEqual(c.mem_pool_host.free.call_count, 1)
                    self.assertIs(
                        c.mem_pool_host.free.call_args.args[0], borrowed_indices
                    )

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
        c.storage_config.tp_rank = 1  # The source is a global rank, not pp_rank=0.
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
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_not_called()
        self.assertTrue(c.pp_prefetch_states["hit"].queried_event.is_set())
        self.assertIs(c.pp_prefetch_states["hit"].operation, operation)

    def test_idle_downstream_skips_empty_broadcasts_before_ticket_and_stop(self):
        c = self.c
        self.submit()
        ticket = c.pp_prefetch_states.pop("hit").ticket
        c.storage_config.pp_rank = 1
        with (
            patch(
                f"{HybridCacheController.__module__}.broadcast_pyobj",
                side_effect=[[], [], [ticket], [None]],
            ),
            patch.object(c.pp_prefetch_command_queue, "task_done") as task_done,
        ):
            c.pp_prefetch_command_thread_func()
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_not_called()
        self.assertTrue(c.pp_prefetch_states["hit"].queried_event.is_set())
        task_done.assert_not_called()


if __name__ == "__main__":
    unittest.main()
