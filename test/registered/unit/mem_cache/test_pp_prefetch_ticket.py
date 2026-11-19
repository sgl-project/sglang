"""CPU regressions for PP ticket ordering, admission, and buffer ownership."""

import pickle
import tempfile
import threading
import time
import unittest
from array import array
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchAck
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
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
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache import fixtures
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=39, suite="base-a-test-cpu")

# The pool schema a hit query over stages holding only the KV host pool
# reduces: the host pools some stage registers.
_SCHEMA = [str(PoolName.KV)]


@contextmanager
def _stop_after_transfers(c, operations):
    """Drive the IO worker through ``operations`` transfers: the stop event
    reads clear while they run (a transfer consults it per batch and skips
    the backend once it is set) and set at the loop check after the last."""
    done = []
    real = c._page_transfer

    def page_transfer(operation):
        try:
            return real(operation)
        finally:
            done.append(operation)

    with (
        patch.object(c, "_page_transfer", page_transfer),
        patch.object(
            c.storage_stop_event,
            "is_set",
            side_effect=lambda: len(done) >= operations,
        ),
    ):
        yield


class _TwoStageGroup:
    """An all-gather between two simulated pipeline stages, one per thread."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=10)
        self.slots = [None, None]

    def all_gather(self, stage, obj):
        self.slots[stage] = obj
        self.barrier.wait()
        gathered = list(self.slots)
        self.barrier.wait()
        return gathered


class TestPrefetchAckSchema(unittest.TestCase):
    def test_stages_with_different_pools_reduce_one_shape(self):
        """BUG REGRESSION. Stages holding different host pools must reduce a
        failed prefetch's ACK with tensors of one shape."""
        group = _TwoStageGroup()
        local_pools = ({"kv": None, "window_4096": None}, {"kv": None})
        reduced = [None, None]

        def stage(index):
            c = HybridCacheController.__new__(HybridCacheController)
            c.mem_pool_host = Mock(entry_map=local_pools[index])
            c.prefetch_completion_sync_groups = [group]
            c._host_pool_schemas = []
            c._all_gather_object = lambda obj, g: g.all_gather(index, obj)
            c._all_reduce = lambda tensor, op, groups: reduced.__setitem__(
                index, tensor.shape
            )
            ack = PrefetchAck(rid="r", operation=None, pool_hits={})
            c._reduce_prefetch_ack(ack)
            return ack

        threads = [threading.Thread(target=stage, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertIsNotNone(reduced[0])
        self.assertEqual(reduced[0], reduced[1])


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
        c.prefetch_io_aux_thread = None
        c.prefetch_tokens_occupied = 0
        c.mem_pool_host = Mock(page_size=4, entry_map={PoolName.KV: Mock()})
        c.mem_pool_host.available_size.return_value = 1 << 20
        c.mem_pool_host.layout_lease.side_effect = nullcontext
        c.mem_pool_host.alloc.side_effect = lambda size, **_: torch.arange(size)
        c.storage_backend_type = "mooncake"
        c.storage_backend = Mock()
        c._storage_hit_query = Mock(side_effect=self.query_hits)
        c._all_reduce = Mock()
        c._host_pool_schemas = []
        c._all_gather_object = Mock(side_effect=lambda obj, group: [obj])
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
            _stop_after_transfers(self.c, operations),
        ):
            self.c.prefetch_io_aux_func()
        acks = []
        while not self.c.prefetch_sync_queue.empty():
            acks.append(self.c.prefetch_sync_queue.get_nowait())
        return acks

    def test_stop_event_answers_tickets_as_misses_but_still_reduces(self):
        """A ticket delivered after the stop event is answered without a
        backend query or a host allocation, but its hits reduce still runs,
        so the stages stay paired; the operation reaches prefetch_buffer
        terminated at zero hits."""
        self.assertTrue(self.submit("hit").decision)
        self.c.storage_stop_event.set()
        queries = self.c._storage_hit_query.call_count
        reduces = self.c._all_reduce.call_count
        with patch.object(self.c, "_allocate_pp_prefetch_buffers") as allocate:
            self.run_commands()
        self.assertEqual(self.c._storage_hit_query.call_count, queries)
        allocate.assert_not_called()
        self.assertEqual(self.c._all_reduce.call_count, reduces + 1)
        operation = self.c.prefetch_buffer.get_nowait()
        self.assertTrue(operation.is_terminated())
        self.assertEqual(operation.storage_hit_count, 0)
        self.assertTrue(self.c.prefetch_buffer.empty())

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
                self.sync_acks(*self.run_io(1))
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertFalse(
                    cache.prefetch_from_storage(CacheRequestHandle(rid, 1), 0, [])
                )
                self.assertFalse(c.release_pp_prefetch(rid))
                self.assertIsNone(c.get_prefetch_submission(rid))
        self.assertTrue(c.pp_prefetch_command_queue.empty())

    def test_all_backends_query_and_allocate_before_one_worker_consensus(self):
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
                self.assertFalse(c.is_pp_prefetch_ready(backend))
                self.assertFalse(first.operation.assume_stored)
                c.mem_pool_host.alloc.assert_not_called()
                c._storage_hit_query.assert_not_called()
                c._all_reduce.assert_not_called()

                votes = []

                def reduce_hits(tensor, op, groups):
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    self.assertEqual(groups, c.prefetch_hits_sync_groups)
                    self.assertEqual(tensor.numel(), 1 + len(_SCHEMA))
                    self.assertEqual(tensor[0].item(), 8)
                    self.assertEqual(c.mem_pool_host.alloc.call_count, len(votes) + 1)
                    votes.append(tensor[0].item())

                c._storage_hit_query.side_effect = self.query_hits
                c._all_reduce.side_effect = reduce_hits
                self.run_commands()
                self.assertEqual(
                    c._storage_hit_query.call_args_list,
                    [call(first.operation), call(second.operation)],
                )
                self.assertEqual(votes, [8, 8])
                self.assertEqual(c._all_reduce.call_count, 2)
                self.assertIs(c.prefetch_buffer.get_nowait(), first.operation)
                self.assertIs(c.prefetch_buffer.get_nowait(), second.operation)

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

    def test_file_stage_hits_trim_buffers_after_local_allocation(self):
        """PP0's local files cannot reveal a shorter prefix on another stage."""
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
                    local_hit = 8 if rank == 0 else 4

                    def reduce_hits(tensor, op, groups):
                        self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                        self.assertEqual(groups, ["pp", "tp"])
                        self.assertEqual(tensor.numel(), 1 + len(_SCHEMA))
                        local_votes.append(tensor[0].item())
                        c.mem_pool_host.alloc.assert_called_once_with(
                            local_hit, pool=PoolName.KV
                        )
                        tensor[0] = min(tensor[0].item(), peer_hit)
                        kv_index = 1 + _SCHEMA.index(PoolName.KV)
                        tensor[kv_index] = min(tensor[kv_index].item(), peer_hit // 4)

                    c._all_reduce.side_effect = reduce_hits
                    self.run_commands(ticket)
                    operation = c.prefetch_buffer.get_nowait()
                    self.assertEqual(operation.handle, CacheRequestHandle("hit", 3))
                    self.assertEqual(operation.storage_hit_count, 4)
                    self.assertEqual(operation.hash_value, hashes[:1])
                    self.assertEqual(operation.host_indices.numel(), 4)
                    self.assertFalse(operation.is_terminated())
                    self.assertEqual(c.prefetch_tokens_occupied, 4)
                    if rank == 0:
                        self.assertIs(operation, result.operation)
                        freed = c.mem_pool_host.free.call_args
                        self.assertEqual(freed.kwargs, {"pool": PoolName.KV})
                        self.assertTrue(torch.equal(freed.args[0], torch.arange(4, 8)))
                        self.assertEqual(c.mem_pool_host.free.call_count, 1)
                    else:
                        c.mem_pool_host.free.assert_not_called()
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
                            self.assertEqual(tensor.numel(), 1 + len(_SCHEMA))
                            local_votes.append(tensor[0].item())
                            kv_index = 1 + _SCHEMA.index(PoolName.KV)
                            self.assertEqual(
                                tensor[kv_index].item(), 32 if rank == 0 else 8
                            )
                            peer_vote = 2048 if rank == 0 else 4096
                            tensor[0] = min(tensor[0].item(), peer_vote)
                            tensor[kv_index] = min(tensor[kv_index].item(), 8)

                        c._all_reduce.side_effect = reduce_hits
                        self.run_commands(ticket)
                        operation = c.prefetch_buffer.get_nowait()
                        if rank == 0:
                            self.assertIs(operation, result.operation)
                        self.assertEqual(operation.storage_hit_count, 2048)
                        self.assertEqual(local_votes, [local_vote])
                        c.mem_pool_host.alloc.assert_called_once_with(
                            local_vote, pool=PoolName.KV
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
                        if rank == 0:
                            self.assertEqual(c.mem_pool_host.free.call_count, 1)
                            freed = c.mem_pool_host.free.call_args
                            self.assertEqual(freed.kwargs, {"pool": PoolName.KV})
                            self.assertTrue(
                                torch.equal(freed.args[0], torch.arange(2048, 4096))
                            )
                        else:
                            c.mem_pool_host.free.assert_not_called()

    def test_checkpoint_vote_stays_within_capacity_capped_hit(self):
        """A stage whose hit is capped to free host space must not vote for
        restorable checkpoints past that cap, or the agreed hit outgrows the
        buffers it allocated."""
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

        operation = c.prefetch_buffer.get_nowait()
        self.assertEqual(votes[0][: stored_pages + 1], [1, 1, 1] + [0] * 6)
        self.assertEqual(operation.host_indices.numel(), free_pages * 4)
        self.assertEqual(operation.storage_hit_count, free_pages * 4)
        self.assertEqual(len(operation.hash_value), free_pages)
        self.assertFalse(operation.is_terminated())

    def test_local_miss_or_query_error_completes_before_next_ticket(self):
        """A stage miss/error must complete admission without breaking ACK order."""
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
                    self.assertEqual(tensor.numel(), 1 + len(_SCHEMA))
                    local_votes.append(tensor[0].item())
                    if tensor[0].item():
                        self.assertEqual(c.mem_pool_host.alloc.call_args.args, (8,))
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
                self.assertEqual(following.host_indices.numel(), 8)
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
                if name == "peer_miss":
                    self.assertEqual(
                        c.mem_pool_host.alloc.call_args_list,
                        [call(8, pool=PoolName.KV)] * 2,
                    )
                    self.assertEqual(c.mem_pool_host.free.call_count, 1)
                    self.assertTrue(
                        torch.equal(
                            c.mem_pool_host.free.call_args.args[0], torch.arange(8)
                        )
                    )
                    self.assertEqual(
                        c.mem_pool_host.free.call_args.kwargs, {"pool": PoolName.KV}
                    )
                else:
                    c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
                    c.mem_pool_host.free.assert_not_called()
                if name == "query_error":
                    self.assertEqual(failed.query_pool_hit_pages, {})
                self.assertEqual(c.prefetch_tokens_occupied, occupied_before + 8)

                c._all_reduce.side_effect = None
                c.page_get_func = Mock(return_value=1)
                acks = self.run_io(2)
                self.assertEqual(
                    [ack.rid for ack in acks], [failed_rid] + [next_rid] * 3
                )
                self.assertTrue(acks[0].completed_req)
                self.assertEqual(
                    [ack.completed_tokens for ack in acks[1:]], [4, 8, None]
                )
                self.assertEqual(c.page_get_func.call_count, 2)
                self.sync_acks(*acks)
                handle = CacheRequestHandle(failed_rid, 3)
                self.assertTrue(c.is_pp_prefetch_ready(failed_rid))
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[handle], 0)
                self.assertNotIn(handle, cache.ongoing_prefetch)
                cache._handle_prefetch_result.assert_not_called()
                self.assertTrue(cache.check_prefetch_progress(handle))
                self.assertFalse(c.release_pp_prefetch(failed_rid))
                self.assertTrue(
                    cache.check_prefetch_progress(CacheRequestHandle(next_rid, 0))
                )
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
        operation = c.prefetch_buffer.get_nowait()
        self.assertEqual(operation.handle, handle)
        self.assertEqual(
            operation.hash_value, get_storage_hash_str(ticket.prefetch_key, page_size=4)
        )
        self.assertTrue(operation.token_ids.is_bigram)
        self.assertEqual(len(operation.token_ids), 8)
        c._storage_hit_query.assert_called_once_with(operation)
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

    def test_poll_exchanges_readiness_once_for_all_tickets(self):
        c, cache = self.c, self.cache
        tickets = [CacheRequestHandle(f"t{i}", 0) for i in range(6)]
        local = CacheRequestHandle("local", 0)
        for handle in tickets:
            c.pp_prefetch_decisions[handle.rid] = PPPrefetchDecision.TICKETED
        # Finishing a ticket clears its decision, as take_ready_pp_prefetch does.
        cache._finish_pp_prefetch = Mock(
            side_effect=lambda handle: c.pp_prefetch_decisions.pop(handle.rid)
        )
        cache.check_prefetch_progress = Mock()
        queue = [tickets[0], local, *tickets[1:]]

        cache.pp_rank = 0
        with patch.object(
            c, "is_pp_prefetch_ready", side_effect={"t2", "t4"}.__contains__
        ):
            cache.poll_prefetch_progress(queue)
        cache._all_reduce.assert_called_once()
        ready, op = cache._all_reduce.call_args.args
        self.assertEqual(
            (ready.tolist(), op), ([0, 0, 1, 0, 1, 0], torch.distributed.ReduceOp.MAX)
        )
        self.assertEqual(
            [call.args[0] for call in cache._finish_pp_prefetch.call_args_list],
            [tickets[2], tickets[4]],
        )
        # Only the unticketed request takes the local path; finished tickets do not.
        cache.check_prefetch_progress.assert_called_once_with(local)

        # A downstream stage follows PP0's verdict from the same single exchange.
        cache.pp_rank = 1
        cache._all_reduce.reset_mock()
        cache._finish_pp_prefetch.reset_mock()
        cache._all_reduce.side_effect = lambda tensor, _: tensor.__setitem__(3, 1)
        for handle in (tickets[2], tickets[4]):
            c.pp_prefetch_decisions[handle.rid] = PPPrefetchDecision.TICKETED
        cache.poll_prefetch_progress(queue)
        cache._all_reduce.assert_called_once()
        cache._finish_pp_prefetch.assert_called_once_with(tickets[3])

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
        c.storage_config.pp_rank = 1
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
        next_operation = c.pp_prefetch_states["next"].operation
        self.assertTrue(failed.is_terminated())
        self.assertEqual(failed.storage_hit_count, 0)
        self.assertEqual(failed.hash_value, [])
        self.assertEqual(failed.host_indices.numel(), 0)
        self.assertEqual(
            c._storage_hit_query.call_args_list, [call(failed), call(next_operation)]
        )
        self.assertEqual(c._all_reduce.call_count, 2)
        self.assertEqual(
            [
                invocation.args[0][0].item()
                for invocation in c._all_reduce.call_args_list
            ],
            [0, 8],
        )
        c.mem_pool_host.free.assert_called_once_with(kv, pool=PoolName.KV)
        c.page_get_func = Mock(return_value=1)
        c.storage_backend = Mock()
        c.storage_backend.batch_get_v2.return_value = {"swa": [True]}
        with (
            patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1),
            _stop_after_transfers(c, 2),
        ):
            c.prefetch_io_aux_func()
        acks = [c.prefetch_sync_queue.get_nowait() for _ in range(6)]
        self.assertEqual([a.rid for a in acks], ["hit"] * 2 + ["next"] * 4)
        self.assertEqual(
            [a.completed_tokens for a in acks], [None, None, 4, 8, None, None]
        )
        self.assertEqual((acks[0].pool_hits, acks[4].pool_hits), ({}, {"swa": 1}))
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
        c.storage_config.pp_rank = 1
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
                c.storage_config.pp_rank = rank
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
        self.assertEqual(operation.hash_value, [])
        self.assertEqual(operation.storage_hit_count, 0)
        c._storage_hit_query.assert_called_once_with(operation)
        c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
        c._all_reduce.assert_called_once()
        packed, op, groups = c._all_reduce.call_args.args
        self.assertEqual(packed[0].item(), 0)
        self.assertEqual(packed.numel(), 1 + len(_SCHEMA))
        self.assertEqual(op, torch.distributed.ReduceOp.MIN)
        self.assertEqual(groups, c.prefetch_hits_sync_groups)
        c.mem_pool_host.free.assert_not_called()
        self.sync_acks(
            PrefetchAck("hit", operation, completed_tokens=0, completed_req=True)
        )
        c.take_ready_pp_prefetch("hit")
        self.assertIsNone(borrowed.host_indices)
        self.assertEqual(c.mem_pool_host.free.call_args.kwargs["pool"], PoolName.SWA)

    def test_peer_vote_trims_kv_alias_and_defers_sidecar_free(self):
        """A peer's zero allocation vote must release KV without freeing a sidecar."""
        c = self.c
        for peer_vote in (4, 0):
            with self.subTest(peer_vote=peer_vote):
                rid = f"peer_{peer_vote}"
                c._storage_hit_query.reset_mock()
                c._all_reduce.reset_mock()
                c.mem_pool_host.alloc.reset_mock()
                c.mem_pool_host.free.reset_mock()
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
                    self.assertEqual(tensor.numel(), 1 + len(_SCHEMA))
                    self.assertEqual(op, torch.distributed.ReduceOp.MIN)
                    self.assertEqual(groups, c.prefetch_hits_sync_groups)
                    c.mem_pool_host.alloc.assert_called_once_with(8, pool=PoolName.KV)
                    tensor[0] = peer_vote

                c._all_reduce.side_effect = reduce_hits
                self.run_commands()
                c._storage_hit_query.assert_called_once_with(operation)
                c._all_reduce.assert_called_once()
                swa, indexer = operation.pool_transfers
                self.assertEqual(operation.host_indices.numel(), peer_vote)
                self.assertEqual(operation.storage_hit_count, peer_vote)
                self.assertEqual(operation.is_terminated(), peer_vote == 0)
                self.assertIs(indexer.host_indices, operation.host_indices)
                self.assertIs(indexer.keys, operation.hash_value)
                self.assertIs(swa.host_indices, borrowed_indices)
                self.assertEqual(c.prefetch_tokens_occupied, peer_vote)
                freed = c.mem_pool_host.free.call_args
                self.assertEqual(c.mem_pool_host.free.call_count, 1)
                self.assertEqual(freed.kwargs, {"pool": PoolName.KV})
                self.assertTrue(torch.equal(freed.args[0], torch.arange(peer_vote, 8)))

                self.assertTrue(c.release_pp_prefetch(rid))
                self.assertIs(swa.host_indices, borrowed_indices)
                self.assertEqual(c.mem_pool_host.free.call_count, 1)
                c._all_reduce.side_effect = None
                self.sync_acks(
                    PrefetchAck(rid, operation, completed_tokens=peer_vote),
                    PrefetchAck(rid, operation, completed_req=True),
                )
                self.assertIsNone(swa.host_indices)
                self.assertNotIn(rid, c.pp_prefetch_states)
                self.assertEqual(c.prefetch_tokens_occupied, 0)
                sidecar_frees = [
                    invocation
                    for invocation in c.mem_pool_host.free.call_args_list
                    if invocation.kwargs["pool"] == PoolName.SWA
                ]
                self.assertEqual(len(sidecar_frees), 1)
                self.assertIs(sidecar_frees[0].args[0], borrowed_indices)

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
        self.assertIs(c.prefetch_buffer.get_nowait(), operation)
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_called_once()

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
        self.assertEqual(c.prefetch_buffer.get_nowait().request_id, "hit")
        self.assertTrue(c.prefetch_buffer.empty())
        c.mem_pool_host.alloc.assert_called_once()
        task_done.assert_not_called()


class _StageStore:
    load_mode = "prefetch"
    prefetch_policy = "wait_complete"

    def __init__(self, rank):
        self.pools = {}
        self.num_completed_offloads = Mock(return_value=0)
        self.offload = Mock(return_value=False)
        self.reset = Mock()
        self.close = Mock()
        self.rank = rank
        self.first_lookup = threading.Event()
        self.reads = []
        self.requests_by_first_page = {}
        self.slow_read = None

    def lookup(self, rid, transfers):
        self.requests_by_first_page[transfers[0].keys[0]] = rid
        if rid == "hit":
            self.first_lookup.set()
            return [1, 3, 5] if self.rank == 0 else [2, 3, 5]
        if rid == "miss" and self.rank == 1:
            return []
        return [2]

    def prefetch(self, transfers):
        keys = transfers[0].keys
        rid = self.requests_by_first_page[keys[0]]
        self.reads.append((rid, len(keys)))
        done = Future()
        if rid == "slow":
            done.set_running_or_notify_cancel()
            self.slow_read = done
        else:
            done.set_result(rid != "failed-get" or self.rank == 0)
        return done


def _request(rid, first_token, size=7):
    return Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", range(first_token, first_token + size)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _drain_until(cache, condition):
    deadline = time.monotonic() + 20
    while True:
        cache.check_hicache_events()
        ready = torch.tensor(int(condition()))
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if ready.item():
            return
        assert time.monotonic() < deadline, "stage did not reach the expected state"


def _check_overlapping_reads(cache, store, allocator):
    """A running GET must leave the next ticket free to start and survive a skip."""
    cache.prefetch_stop_policy = "timeout"
    cache.prefetch_timeout_base = cache.prefetch_timeout_per_page = 0
    slow, fast = _request("slow", 0), _request("fast", 50)
    # Reuse the earlier hit so the running GET must keep a non-root anchor.
    prefix = fixtures.match_prefix(cache, slow.origin_input_ids[:-1])
    assert prefix.device_prefix_len == 3

    def anchor_refs():
        return fixtures.node_device_lock_ref(
            cache, prefix.last_device_node, ComponentType.FULL
        )

    baseline = anchor_refs()
    tickets = cache.linker.pp_tickets
    assert tickets.bind(slow, expected=True)
    assert anchor_refs() == baseline + 1
    assert tickets.bind(fast, expected=True)
    _drain_until(cache, lambda: ("fast", 2) in store.reads)
    assert store.slow_read.running()
    assert anchor_refs() == baseline + 1
    if store.rank == 0:
        assert not tickets.can_admit(slow.cache_request_handle)
    cache.check_hicache_events()
    assert tickets.can_admit(slow.cache_request_handle)
    assert store.slow_read.running()
    if store.rank == 0:
        store.slow_read.set_result(True)
    dist.barrier()
    cache.check_hicache_events()
    assert anchor_refs() == baseline + 1
    assert allocator.available_size() == 32 - 5 - 2 - 2
    if store.rank == 1:
        store.slow_read.set_result(True)
    _drain_until(cache, lambda: not tickets.pending)
    assert anchor_refs() == baseline
    assert not cache.linker.prefetches
    for req, expected in ((slow, 3), (fast, 2)):
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids[:-1]))
        )
        assert match.device_prefix_len == expected
    assert allocator.available_size() == 32 - 5 - 2


def _check_blocked_follower_lookup(cache, store):
    tickets = cache.linker.pp_tickets
    for policy in ("best_effort", "timeout"):
        cache.prefetch_stop_policy = policy
        cache.prefetch_timeout_base = cache.prefetch_timeout_per_page = 0
        entered, resume, timed_out = (threading.Event() for _ in range(3))
        lookup = store.lookup

        def blocked_lookup(rid, transfers):
            result = lookup(rid, transfers)
            entered.set()
            if store.rank == 1 and not resume.wait(10):
                timed_out.set()
            return result

        with patch.object(store, "lookup", blocked_lookup):
            req = _request(policy, 60)
            handle = req.cache_request_handle
            assert tickets.bind(req, expected=True)
            try:
                assert entered.wait(10)
                dist.barrier()
                # PP0 has voted its local query ready. The follower must still
                # return from its scheduler poll while its own query is blocked.
                cache.check_hicache_events()
                if store.rank == 0:
                    assert not tickets.can_admit(handle)
                # PP1's candidate loop did not reach the blocked request. It
                # still applies PP0's stop decision in the next regular tick.
                cache.check_hicache_events()
                assert tickets.can_admit(handle)
            finally:
                resume.set()
            blocked = torch.tensor(int(timed_out.is_set()))
            dist.all_reduce(blocked, op=dist.ReduceOp.MAX)
            assert not blocked.item(), "scheduler waited on a follower's storage query"
            _drain_until(cache, lambda: not tickets.pending)
            assert all(rid != policy for rid, _ in store.reads)


def _check_layer_wise(cache, allocator, rank):
    store = _StageStore(rank)
    store.load_mode = "layer_wise"
    queries = []
    probed_suffix = threading.Event()

    def lookup(rid, transfers):
        pages = len(transfers[0].keys)
        queries.append((rid, pages))
        if pages == 4:
            probed_suffix.set()
            return [2, 4]  # Queried after PP0's two local pages.
        return [] if rid == "gap" else [1, 3, 4]  # From this stage's root.

    store.lookup = lookup
    cache.init_cache_linker(store)
    tickets = cache.linker.pp_tickets
    gap, hit = _request("gap", 70), _request("layer-hit", 70)
    if rank == 0:
        cache.insert(
            InsertParams(
                key=RadixKey(gap.origin_input_ids[:2]), value=allocator.alloc(2)
            )
        )
    with patch.object(store, "prefetch", wraps=store.prefetch) as full_get:
        if rank == 1:
            assert probed_suffix.wait(10), "early lookup did not use PP0's L1 boundary"
        for req in (gap, hit):
            assert tickets.bind(req, expected=True)
        _drain_until(cache, lambda: not tickets.pending)
        assert gap.external_cache_hit_length == 0
        assert hit.external_cache_hit_length == 4
        for req, expected in ((gap, 0), (hit, 2 if rank == 0 else 4)):
            req.init_next_round_input(cache, cow_mamba=False)
            assert req.host_hit_length == expected, (rank, req.rid, req.host_hit_length)
            tickets.release(req.cache_request_handle)
        if rank == 1:
            assert [pages for rid, pages in queries if rid == "gap"] == [4, 6]
        full_get.assert_not_called()

    def make_adder():
        return PrefillAdder(
            page_size=1,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=100,
            rem_chunk_tokens=None,
        )

    # Both stages can budget this request; only PP1's physical take fails.
    # The next request has room for one copy of its hit but not two.
    store.load = Mock(return_value=True)
    store.start_layer_wise_loading = Mock(return_value=0)
    store.finish_layer_wise_loading = Mock(return_value=True)
    for failed in (True, False):
        req = _request(f"take-{failed}", 95)
        free = allocator.available_size()
        held = allocator.alloc(free - 10)
        assert tickets.bind(req, expected=True)
        with (
            patch.object(type(cache.kv_storage()), "take_units", return_value=None)
            if failed and rank == 1
            else nullcontext()
        ):
            _drain_until(cache, lambda: not tickets.pending)
        req.init_next_round_input(cache, cow_mamba=False)
        adder = make_adder()
        # The physical destinations must already be held through admission.
        with patch.object(
            type(cache.kv_storage()), "take_units", side_effect=AssertionError
        ):
            adder.add_one_req(req, False, None)
        assert adder.can_run_list == [req]
        assert (req.prefix_len, req.extend_len) == ((0, 7) if failed else (4, 3))
        if not failed:
            slots = cache.linker.private_load_indices[req.rid]
            cache.linker.start_layer_wise_loading()
            assert not cache.linker.finish_loads()
            allocator.free(slots)
        cache.unlock(req.lock)
        tickets.release(req.cache_request_handle)
        allocator.free(held)
        assert allocator.available_size() == free

    # A full queue can defer every layer-wise admission. At most one ticket
    # holds private rows, then its scheduling turn expires and frees them.
    free = allocator.available_size()
    held = allocator.alloc(free - 10)
    waiting = [_request(f"queued-{i}", 110 + 10 * i) for i in range(8)]
    for req in waiting:
        assert tickets.bind(req, expected=True)

    def deferred():
        assert len(cache.linker.prefetches) <= 1
        assert allocator.available_size() >= 4
        return not tickets.pending

    _drain_until(cache, deferred)
    assert all(req.external_cache_hit_length == 0 for req in waiting[:-1])
    assert waiting[-1].external_cache_hit_length == 4
    cache.check_hicache_events()
    assert all(req.external_cache_hit_length == 0 for req in waiting)
    assert allocator.available_size() == 10
    assert tickets.layer_admission is None

    req = waiting[0]
    req.init_next_round_input(cache, cow_mamba=False)
    adder = make_adder()
    adder.add_one_req(req, False, None)
    assert adder.can_run_list == [req] and (req.prefix_len, req.extend_len) == (0, 7)
    cache.unlock(req.lock)

    req = _request("cancel-ready", 200)
    assert tickets.bind(req, expected=True)
    _drain_until(cache, lambda: not tickets.pending)
    assert req.external_cache_hit_length == 4
    cache.linker.release_request(req.cache_request_handle)
    assert not cache.linker.prefetches and tickets.layer_admission is None
    assert allocator.available_size() == 10
    allocator.free(held)

    # A later request hits the entire pool only on PP1. Its queued ticket must
    # leave those pages evictable until the earlier miss has entered the batch.
    cache.evict(EvictParams(num_tokens=allocator.size))
    cold = _request("gap", 230, size=6)
    later = _request("all-local-on-pp1", 300, size=allocator.size + 1)
    if rank == 1:
        cache.insert(
            InsertParams(
                key=RadixKey(later.origin_input_ids[:-1]),
                value=allocator.alloc(allocator.size),
            )
        )
    for req in (cold, later):
        assert tickets.bind(req, expected=True)
    _drain_until(cache, lambda: tickets.can_admit(cold.cache_request_handle))
    assert later.cache_request_handle in tickets.pending
    cold.init_next_round_input(cache, cow_mamba=False)
    adder = make_adder()
    adder.add_one_req(cold, False, None)
    assert adder.can_run_list == [cold] and (cold.prefix_len, cold.extend_len) == (0, 6)
    cache.unlock(cold.lock)
    tickets.release(later.cache_request_handle)
    _drain_until(cache, lambda: not tickets.pending)


def _check_unbound_and_retracted_tickets(cache, store):
    tickets = cache.linker.pp_tickets
    req = _request("retry", 80)
    rejected = req.cache_request_handle
    if store.rank == 1:
        tickets.skip_unbound(req.rid)
    dist.barrier()  # The follower rejected the relay before receiving the ticket.
    if store.rank == 0:
        assert tickets.bind(req, expected=False)
    empty = _request("empty", 90, size=1)
    for retracted in (empty, req):
        retracted.reset_for_retract()
        retracted.advance_cache_request_handle()
        cache.linker.prefetch(retracted)

    if store.rank == 0:
        assert not tickets.can_admit(rejected)
    # PP1 already rejected the request and never polls it for admission.
    # The ordered exchange carries three tickets despite that asymmetry.
    cache.check_hicache_events()
    assert tickets.can_admit(rejected)
    reads_before = len(store.reads)
    _drain_until(cache, lambda: not tickets.pending)
    assert store.reads[reads_before:] == [("retry", 2)]
    assert rejected not in cache.prefetch_loaded_tokens_by_reqid
    assert empty.cache_request_handle not in cache.prefetch_loaded_tokens_by_reqid
    assert cache.prefetch_loaded_tokens_by_reqid[req.cache_request_handle] == 2

    # Stop with a locked L1 anchor on PP0 and no matching request on PP1.
    protected = cache.full_protected_size()
    if store.rank == 0:
        assert tickets.bind(_request("pending-reset", 80), expected=False)
        assert cache.full_protected_size() > protected
    dist.barrier()
    cache.linker.reset()
    assert cache.full_protected_size() == protected
    active = cache.linker.pp_tickets
    following = _request("after-reset", 100)
    assert active.bind(following, expected=True)
    _drain_until(cache, lambda: not active.pending)
    assert cache.prefetch_loaded_tokens_by_reqid[following.cache_request_handle] == 2


def _check_layer_load_verdict(cache, rank):
    """Use the real launch/retirement paths and move their wire payloads over Gloo."""
    from collections import deque
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import (
        SchedulerPPMixin,
        _pp_can_skip_output_comm,
    )
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    scheduler = SchedulerPPMixin()
    scheduler.tree_cache = cache
    scheduler.pp_group = SimpleNamespace(
        is_first_rank=rank == 0, is_last_rank=rank == 1
    )
    scheduler.forward_stream_ctx = nullcontext()
    scheduler._pp_wait_forward_dependencies = lambda: None
    scheduler.device_module = Mock()
    scheduler.future_map = Mock()
    scheduler.tp_worker = Mock()
    scheduler.enable_unified_cache_external_linker = True
    for failing_stage, rejected, middle in (
        (1, False, True),
        (0, False, False),
        (1, True, False),
        (None, False, True),
    ):
        req = _request("layer-restore", 90)
        req.pp_linker_verdict_end = req.extend_end = 4
        healthy = _request("unrelated-request", 120)
        scheduler.chunked_req = req if middle else None
        batch = ScheduleBatch(
            reqs=[req] if middle else [req, healthy],
            chunked_req=scheduler.chunked_req,
            contains_last_prefill_chunk=not middle,
            forward_mode=ForwardMode.EXTEND,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        linker = cache.linker
        # Only one stage needs storage; the other has the entire boundary in L1.
        reads_here = rank == (failing_stage if failing_stage is not None else 1)
        linker.inflight_load_rids = [req.rid] if reads_here else []
        linker.inflight_load_index = 0 if reads_here and not rejected else -1
        if reads_here and rejected:
            linker.rejected_pp_loads.add(req.rid)
        linker.cache_linker.finish_layer_wise_loading = Mock(
            return_value=failing_stage is None
        )

        def forward(batch, proxy):
            assert proxy is None or "pp_failed_load_rids" not in proxy.tensors
            result = GenerationBatchResult(
                next_token_ids=torch.zeros(len(batch.reqs), dtype=torch.int64),
                pp_hidden_states_proxy_tensors=PPProxyTensors(
                    {"hidden_states": torch.ones(1)}
                ),
            )
            result.pp_failed_load_rids = tuple(linker.finish_loads(batch.reqs))
            return result

        scheduler.run_batch = forward
        metadata, outputs = [None], deque()
        payload = [None]
        if rank == 1:
            dist.recv_object_list(payload, src=0)
        result, _ = scheduler._pp_launch_batch(
            0, batch, PPProxyTensors(payload[0]) if rank else None, metadata, outputs
        )
        with envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(True):
            assert batch.pp_linker_load and not _pp_can_skip_output_comm(batch)
        if middle:
            assert scheduler._pp_linker_chunk_batch is batch
            # The chunk cannot be consumed by the prefill planner during the
            # return trip. Other decode work and the output ring remain runnable.
            assert Scheduler._get_new_batch_prefill_raw(scheduler, None, batch) == (
                None,
                batch,
            )
        if rank == 0:
            scheduler.launch_event = None
            scheduler._pp_send_dict_to_next_stage = lambda tensors, **kwargs: (
                dist.send_object_list([tensors], dst=1)
            )
            scheduler._pp_send_proxy_to_next_stage(result)
            assert (
                "pp_failed_load_rids"
                not in result.pp_hidden_states_proxy_tensors.tensors
            )
        else:
            payload[0] = outputs.popleft()[1].tensors
        dist.broadcast_object_list(payload, src=1)
        if rejected:
            # Speculative PP can filter the live rows before its output returns.
            metadata[0].fwd_batch = batch.copy()
            batch.reqs = [healthy]
        scheduler._pp_prep_batch_result(batch, metadata[0], PPProxyTensors(payload[0]))
        assert scheduler._pp_linker_chunk_batch is None
        assert bool(req.skip_radix_cache_insert) == (failing_stage is not None)
        assert (
            healthy.discard_output_reason is None
            and not healthy.skip_radix_cache_insert
        )
        if failing_stage is None:
            assert req.discard_output_reason is None
        else:
            assert req.discard_output_reason.status_code == 503
            if middle:
                assert scheduler._pending_chunked_abort_req is req
        dist.barrier()


def _run_linker_stage(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=40),
    )
    with patch.object(fixtures, "get_device", return_value="cpu"):
        cache, allocator, _ = fixtures.build_fixture(
            CacheConfig(page_size=1, num_layers=1, kv_size=32),
            tree_core_backend="python",
        )
    cache.pp_group = dist.group.WORLD
    cache.pp_size, cache.pp_rank = 2, rank
    store = _StageStore(rank)
    cache.init_cache_linker(store)
    tickets = cache.linker.pp_tickets
    try:
        # Stage 1 can reserve four slots. Although both stores offer page 5,
        # only page 3 is both restorable and backed by local reservations.
        held = allocator.alloc(28) if rank == 1 else None
        requests = [
            _request(rid, i * 10)
            for i, rid in enumerate(("hit", "miss", "failed-get", "next"))
        ]
        if rank == 1:
            # Hold the real request relay until this stage has queried its store.
            assert store.first_lookup.wait(20), "ticket did not precede request relay"
            assert not store.reads
        for req in requests:
            assert tickets.bind(req, expected=rank == 1)
            assert not tickets.can_admit(req.cache_request_handle)

        deadline = time.monotonic() + 30
        while not tickets.can_admit(requests[-1].cache_request_handle):
            cache.check_hicache_events()
            if held is not None and tickets.can_admit(requests[0].cache_request_handle):
                allocator.free(held)
                held = None
            assert time.monotonic() < deadline, "ticket queue did not drain"

        assert store.reads == [("hit", 3), ("failed-get", 2), ("next", 2)]
        for req, expected in zip(requests, (3, 0, 0, 2)):
            assert tickets.can_admit(req.cache_request_handle)
            result = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(req.origin_input_ids[:-1]))
            )
            assert result.device_prefix_len == expected, (rank, req.rid, result)
            tickets.release(req.cache_request_handle)
        assert not cache.linker.prefetches
        assert allocator.available_size() == 32 - 3 - 2
        _check_overlapping_reads(cache, store, allocator)
        _check_blocked_follower_lookup(cache, store)
        _check_unbound_and_retracted_tickets(cache, store)
        cache.linker.close()
        _check_layer_wise(cache, allocator, rank)
        _check_layer_load_verdict(cache, rank)
    finally:
        if store.slow_read is not None and not store.slow_read.done():
            store.slow_read.set_result(False)
        cache.linker.close()
        dist.destroy_process_group()


class TestLinkerPPTickets(CustomTestCase):
    def test_prefetch_and_layer_wise_admission_across_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / "init").as_uri()
            mp.spawn(_run_linker_stage, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
