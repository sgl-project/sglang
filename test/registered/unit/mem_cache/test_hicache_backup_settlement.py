"""Storage backup acks settle at the rank-agreed written prefix: beliefs cover
only what was written, the tail is left for a later write-through to cover.
CPU only: the tree, the controller, its backup thread and the file backend are
real; the D2H transfer engine is faked (no device streams)."""

import contextlib
import json
import os
import shutil
import tempfile
import time
import unittest
from array import array
from datetime import timedelta
from unittest import mock

import torch

import sglang.test.kits.unified_radix_cache.fixtures as _fixtures
from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageFailureInjector,
    PoolName,
)
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    drop_hicache_atexit_pin,
    node_host_lock_ref,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

# Three write batches (128, 128, 44 pages); the failing one is the second.
_PAGES = 2 * STORAGE_BATCH_SIZE + 44
_TAIL = _PAGES - STORAGE_BATCH_SIZE
_CFG = CacheConfig(
    page_size=1,
    components=(ComponentType.FULL,),
    kv_size=1024,
    max_context_len=1024,
)
# Multi-token pages: the tail is sliced by pages * page_size and the agreed
# count must stay page-aligned.
_CFG_PAGED = CacheConfig(
    page_size=4,
    components=(ComponentType.FULL,),
    kv_size=2048,
    max_context_len=2048,
)
_CACHE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


class _FakeEvent:
    def record(self, *args, **kwargs):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def wait(self, *args, **kwargs):
        pass


class _FakeL2Engine:
    """Completes every D2H submission at once, without device streams."""

    def submit_device_to_host(self, transfers):
        return TransferCompletion(_FakeEvent(), _FakeEvent(), False)


class _FailingBatchBackend(HiCacheStorageFailureInjector):
    """The real backend, failing the write batch that carries ``fail_keys``
    for the first ``budget`` such batches (None: every time)."""

    def __init__(self, backend, fail_keys, budget):
        super().__init__(backend, 0.0, 0.0)
        self._fail_keys = set(fail_keys)
        self._budget = budget
        self.failed_batches = 0

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        if any(key in self._fail_keys for key in keys) and (
            self._budget is None or self._budget > 0
        ):
            if self._budget is not None:
                self._budget -= 1
            self.failed_batches += 1
            return False
        return self._backend.batch_set(keys, values, target_locations, target_sizes)


def _build_cache(
    cleanup: contextlib.ExitStack,
    *,
    host_memory_mode: str,
    storage_dir=None,
    cfg: CacheConfig = _CFG,
):
    """A real UnifiedRadixCache + HybridCacheController on CPU, with the file
    L3 backend when ``storage_dir`` is given and a faked D2H engine. The
    parallel topology stays single-rank; a two-rank test installs the gloo
    world as the cache's TP group itself."""
    import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

    # The kit resolves its device through get_device(), which on a CUDA-less
    # runner raises unless the CPU engine is selected; these tests only need
    # CPU tensors, so pin the device without touching the engine selection.
    with mock.patch.object(_fixtures, "get_device", return_value="cpu"):
        cache, allocator, req_to_token_pool = build_fixture(cfg)
    # Host pools pin through CUDA; the CPU fixture keeps them pageable.
    pinned_cls = assembler.get_mha_host_pool_cls

    def unpinned_cls(device_pool):
        host_pool_cls = pinned_cls(device_pool)

        def make(*args, **kwargs):
            kwargs["pin_memory"] = False
            return host_pool_cls(*args, **kwargs)

        return make

    cleanup.enter_context(
        mock.patch.object(assembler, "get_mha_host_pool_cls", side_effect=unpinned_cls)
    )
    cleanup.enter_context(
        get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1)
    )
    extra_config = None
    if storage_dir is not None:
        cleanup.enter_context(
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(storage_dir)
        )
        extra_config = json.dumps({"prefetch_threshold": cfg.page_size})
    server_args = ServerArgs(
        model_path="dummy",
        page_size=cfg.page_size,
        hicache_io_backend="direct",
        hicache_mem_layout="page_first_direct",
        hicache_write_policy="write_through",
        hicache_storage_backend=None if storage_dir is None else "file",
        hicache_storage_backend_extra_config=extra_config,
        hicache_storage_prefetch_policy="wait_complete",
        hicache_host_memory_mode=host_memory_mode,
    )
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, cfg.page_size)
    set_global_server_args_for_scheduler(server_args)
    cache.init_hicache(server_args, cache.cache_init_params)
    cleanup.callback(drop_hicache_atexit_pin, cache)
    cleanup.callback(cache.release_host_resources)
    cc = cache.cache_controller
    if storage_dir is not None:
        cleanup.callback(cc._stop_storage_threads)
    cache.write_through_threshold = 1 << 30
    cache.load_back_threshold = 0
    cc.l2_transfer_engine = _FakeL2Engine()
    return cache, allocator, req_to_token_pool


def _insert_span(cache, allocator, num_tokens):
    tokens = array("q", range(1, 1 + num_tokens))
    value = allocator.alloc(num_tokens)
    cache.insert(InsertParams(key=RadixKey(tokens), value=value))
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens))).last_device_node


def _backup_to_host(cache, node_id):
    """Write-through D2H of ``node_id`` through to its ack, which enqueues
    the L3 write in cache mode."""
    written = cache._execute_and_commit_kv_backup(
        cache.tree_core.build_backup_kv_action(node_id)
    )
    assert written > 0
    cache.writing_check(write_back=True)


def _tick_until(cache, predicate, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        cache.check_hicache_events()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("the HiCache tick did not reach the expected state")


def _spy_storage_writes(cc):
    writes = mock.Mock(side_effect=cc.write_storage)
    cc.write_storage = writes
    return writes


def _write_spans(writes):
    return [(len(c.args[0]), len(c.args[2])) for c in writes.call_args_list]


class _SettlementCase(unittest.TestCase):
    """Buffer mode on ``cfg``: a real cache with the file backend, one span of
    _PAGES pages inserted, storage writes spied."""

    cfg = _CFG

    def setUp(self):
        self._cleanup = contextlib.ExitStack()
        self.addCleanup(self._cleanup.close)
        self.storage_dir = tempfile.mkdtemp(prefix="hicache-backup-settle-")
        self.addCleanup(shutil.rmtree, self.storage_dir, ignore_errors=True)
        self.cache, allocator, _ = _build_cache(
            self._cleanup,
            host_memory_mode="buffer_only",
            storage_dir=self.storage_dir,
            cfg=self.cfg,
        )
        self.cc = self.cache.cache_controller
        self.pipeline = self.cache.buffer_pipeline
        self.leaf = _insert_span(self.cache, allocator, _PAGES * self.cfg.page_size)
        self.chain = self.cache.tree_core.get_hash_values(self.leaf)
        self.assertEqual(len(self.chain), _PAGES)
        self.beliefs = self.cache.storage_existence_cache.pool(PoolName.KV)
        self.collector = self.cache.storage_metrics_collector = mock.Mock()
        self.cache.enable_storage_metrics = True
        self.writes = _spy_storage_writes(self.cc)

    def _fail_second_batch(self, budget):
        backend = _FailingBatchBackend(
            self.cc.storage_backend,
            self.chain[STORAGE_BATCH_SIZE : 2 * STORAGE_BATCH_SIZE],
            budget,
        )
        self.cc.storage_backend = backend
        return backend

    def _backuped_tokens(self):
        return [c.args[0] for c in self.collector.log_backuped_tokens.call_args_list]

    def _settle_short_write(self, budget):
        """One backup whose second batch fails: settled at the first batch,
        the tail believed absent, nothing retried, nothing held."""
        page_size = self.cfg.page_size
        backend = self._fail_second_batch(budget)
        self.cache.request_buffer_backup(self.leaf)
        with self.assertLogs(_CACHE_LOGGER, level="WARNING") as logs:
            _tick_until(self.cache, self.pipeline.is_idle)
        self.assertEqual(sum("tail stays un-backed" in line for line in logs.output), 1)

        self.assertTrue(self.beliefs.contains_all(self.chain[:STORAGE_BATCH_SIZE]))
        self.assertFalse(self.beliefs.contains(self.chain[STORAGE_BATCH_SIZE]))
        self.assertFalse(self.beliefs.covered(self.chain))
        self.assertFalse(self.pipeline.backup_pending(self.leaf))
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.assertEqual(self.pipeline.ongoing_backup, {})
        self.assertEqual(_write_spans(self.writes), [(_PAGES * page_size, _PAGES)])
        self.assertEqual(backend.failed_batches, 1)
        self.assertEqual(backend.batch_exists(self.chain), STORAGE_BATCH_SIZE)
        self.assertEqual(self._backuped_tokens(), [STORAGE_BATCH_SIZE * page_size])
        self.collector.log_storage_backup_failed_tokens.assert_called_once_with(
            _TAIL * page_size
        )
        # Nothing is queued past the settlement.
        for _ in range(3):
            self.cache.check_hicache_events()
        self.assertEqual(self.writes.call_count, 1)
        self.assertEqual(self.cc.backup_queue.qsize(), 0)
        self.cache.sanity_check()
        return backend


class TestStorageBackupSettlement(_SettlementCase):
    """Beliefs follow the agreed written prefix; the tail is left for the
    next write-through through the path, which writes it."""

    def test_short_write_settles_the_prefix_and_the_next_write_through_covers_the_tail(
        self,
    ):
        backend = self._settle_short_write(budget=1)

        # The next write-through through the path (a later hit re-triggers
        # the node's backup) finds the tail uncovered and writes it.
        self.cache.request_buffer_backup(self.leaf)
        _tick_until(self.cache, self.pipeline.is_idle)
        self.assertEqual(self.writes.call_count, 2)
        self.assertTrue(
            set(self.chain[STORAGE_BATCH_SIZE:])
            <= set(self.writes.call_args_list[1].args[2])
        )
        self.assertTrue(self.beliefs.contains_all(self.chain))
        self.assertEqual(backend.batch_exists(self.chain), _PAGES)
        self.assertEqual(backend.failed_batches, 1)
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.collector.log_storage_backup_failed_tokens.assert_called_once_with(_TAIL)
        self.cache.sanity_check()

    def test_persistent_failure_counts_once_per_settlement(self):
        self._settle_short_write(budget=None)
        self.cache.request_buffer_backup(self.leaf)
        _tick_until(self.cache, self.pipeline.is_idle)
        # Settled short again: counted again, still nothing retried or held.
        self.assertEqual(self.writes.call_count, 2)
        self.assertEqual(
            [
                c.args[0]
                for c in self.collector.log_storage_backup_failed_tokens.call_args_list
            ],
            [_TAIL, _TAIL],
        )
        self.assertFalse(self.beliefs.covered(self.chain))
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.cache.sanity_check()

    def test_the_drain_reduces_only_when_it_acked_backups(self):
        with mock.patch.object(
            self.cache,
            "_all_reduce_storage_drain",
            wraps=self.cache._all_reduce_storage_drain,
        ) as reduce:
            for _ in range(3):
                self.cache.check_hicache_events()
            reduce.assert_not_called()
            self.cache.request_buffer_backup(self.leaf)
            _tick_until(self.cache, self.pipeline.is_idle)
            self.assertEqual(reduce.call_count, 1)
            self.assertEqual(reduce.call_args.args[0].tolist(), [_PAGES])
        self.assertTrue(self.beliefs.contains_all(self.chain))
        self.collector.log_storage_backup_failed_tokens.assert_not_called()


class TestStorageBackupSettlementPagedKeys(_SettlementCase):
    """page_size 4: the agreed token count is page-aligned and the believed
    prefix is sliced in pages of the hash chain."""

    cfg = _CFG_PAGED

    def test_short_write_settles_in_pages(self):
        backend = self._settle_short_write(budget=1)
        self.cache.request_buffer_backup(self.leaf)
        _tick_until(self.cache, self.pipeline.is_idle)
        self.assertTrue(self.beliefs.contains_all(self.chain))
        self.assertEqual(backend.batch_exists(self.chain), _PAGES)
        self.cache.sanity_check()


class TestCacheModeShortSettlement(unittest.TestCase):
    """Cache mode: a short write releases the host lock as before and is
    counted; the span is sent again by the existing write-through cycle once
    the host copy is evicted and the node is hit again."""

    def setUp(self):
        self._cleanup = contextlib.ExitStack()
        self.addCleanup(self._cleanup.close)
        storage_dir = tempfile.mkdtemp(prefix="hicache-backup-settle-")
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        self.cache, self.allocator, _ = _build_cache(
            self._cleanup, host_memory_mode="cache", storage_dir=storage_dir
        )
        self.cc = self.cache.cache_controller
        self.leaf = _insert_span(self.cache, self.allocator, _PAGES)
        self.chain = self.cache.tree_core.get_hash_values(self.leaf)
        self.collector = self.cache.storage_metrics_collector = mock.Mock()
        self.cache.enable_storage_metrics = True
        self.writes = _spy_storage_writes(self.cc)

    def test_short_write_settles_and_releases_the_host_lock(self):
        backend = _FailingBatchBackend(
            self.cc.storage_backend,
            self.chain[STORAGE_BATCH_SIZE : 2 * STORAGE_BATCH_SIZE],
            budget=None,
        )
        self.cc.storage_backend = backend
        _backup_to_host(self.cache, self.leaf)
        self.assertEqual(len(self.cache.ongoing_backup), 1)
        self.assertEqual(len(self.cache._backup_requested_tokens), 1)
        self.assertEqual(
            node_host_lock_ref(self.cache, self.leaf, ComponentType.FULL), 1
        )

        with self.assertLogs(_CACHE_LOGGER, level="WARNING") as logs:
            _tick_until(self.cache, lambda: not self.cache.ongoing_backup)
        self.assertEqual(sum("tail stays un-backed" in line for line in logs.output), 1)
        self.assertEqual(self.cache._backup_requested_tokens, {})
        self.assertEqual(
            node_host_lock_ref(self.cache, self.leaf, ComponentType.FULL), 0
        )
        self.assertEqual(self.writes.call_count, 1)
        self.assertEqual(backend.failed_batches, 1)
        self.assertEqual(backend.batch_exists(self.chain), STORAGE_BATCH_SIZE)
        self.assertEqual(
            [c.args[0] for c in self.collector.log_backuped_tokens.call_args_list],
            [STORAGE_BATCH_SIZE],
        )
        self.collector.log_storage_backup_failed_tokens.assert_called_once_with(_TAIL)
        # The host copy stays; the next write-through of the node (after its
        # host eviction) sends the span again.
        self.assertTrue(self.cache.tree_core.is_backuped(self.leaf))
        self.cache.sanity_check()


def _two_rank_settlement(rank, rendezvous_dir, skip_rank):
    """One rank's backend fails the second batch. With ``skip_rank`` rank 1
    is not the KV storage writer (replicated MLA KV: it writes nothing and
    acks with completed_tokens 0) and rank 0 is the failing writer; otherwise
    rank 1 fails its own write while rank 0 writes everything. Both ranks
    must settle the same prefix and count the same shortfall."""
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_dir}/gloo",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    cleanup = contextlib.ExitStack()
    try:
        storage_dir = os.path.join(rendezvous_dir, f"rank{rank}")
        os.makedirs(storage_dir)
        cache, allocator, _ = _build_cache(
            cleanup, host_memory_mode="cache", storage_dir=storage_dir
        )
        cache.tp_group = torch.distributed.group.WORLD
        cache.tp_world_size = 2
        cache.storage_metrics_collector = mock.Mock()
        cache.enable_storage_metrics = True
        cc = cache.cache_controller
        leaf = _insert_span(cache, allocator, _PAGES)
        chain = cache.tree_core.get_hash_values(leaf)
        failing_rank = 0 if skip_rank else 1
        backend = _FailingBatchBackend(
            cc.storage_backend,
            (
                chain[STORAGE_BATCH_SIZE : 2 * STORAGE_BATCH_SIZE]
                if rank == failing_rank
                else ()
            ),
            budget=None,
        )
        cc.storage_backend = backend
        writer = not (skip_rank and rank == 1)
        if not writer:
            cleanup.enter_context(
                mock.patch.object(
                    HiCacheStorageConfig,
                    "is_storage_writer",
                    new_callable=mock.PropertyMock,
                    return_value=False,
                )
            )
            assert cc.backup_skip
        writes = _spy_storage_writes(cc)
        _backup_to_host(cache, leaf)
        assert len(cache.ongoing_backup) == 1
        assert node_host_lock_ref(cache, leaf, ComponentType.FULL) == 1

        for step in range(500):
            cache.check_hicache_events()
            # The in-flight operation ids and every enqueued span must match
            # on both ranks after every tick: the ack drain pairs operations
            # by position.
            snapshot = (sorted(cache.ongoing_backup), _write_spans(writes))
            peers = [None, None]
            torch.distributed.all_gather_object(peers, snapshot)
            assert peers[0] == peers[1], (step, peers)
            settled = torch.tensor([int(not cache.ongoing_backup)])
            torch.distributed.all_reduce(settled, op=torch.distributed.ReduceOp.MIN)
            if settled.item():
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the storage backups did not settle")

        collector = cache.storage_metrics_collector
        outcome = (
            _write_spans(writes),
            [
                c.args[0]
                for c in collector.log_storage_backup_failed_tokens.call_args_list
            ],
            node_host_lock_ref(cache, leaf, ComponentType.FULL),
            dict(cache._backup_requested_tokens),
        )
        peers = [None, None]
        torch.distributed.all_gather_object(peers, outcome)
        # Same settlement on both ranks: nothing re-enqueued, the shortfall
        # counted once from the agreed prefix, the lock released.
        assert peers[0] == peers[1], peers
        assert outcome == ([(_PAGES, _PAGES)], [_TAIL], 0, {}), outcome
        # Backed-up tokens are what this rank wrote of the agreed prefix: the
        # non-writer wrote nothing, every writer the agreed first batch.
        assert [c.args[0] for c in collector.log_backuped_tokens.call_args_list] == [
            STORAGE_BATCH_SIZE if writer else 0
        ]
        assert backend.failed_batches == (1 if rank == failing_rank else 0)
        if not writer:
            assert backend.batch_exists(chain) == 0
        elif rank == failing_rank:
            assert backend.batch_exists(chain) == STORAGE_BATCH_SIZE
        else:
            # The healthy writer wrote everything; it still settles at the
            # agreed prefix.
            assert backend.batch_exists(chain) == _PAGES
        cache.sanity_check()
    finally:
        cleanup.close()
        torch.distributed.destroy_process_group()


class TestTwoRankSettlement(unittest.TestCase):
    def _spawn(self, skip_rank):
        with tempfile.TemporaryDirectory(prefix="hicache-backup-settle-") as directory:
            torch.multiprocessing.spawn(
                _two_rank_settlement, args=(directory, skip_rank), nprocs=2, join=True
            )

    def test_rank_local_failure_settles_both_ranks_at_the_agreed_prefix(self):
        self._spawn(skip_rank=False)

    def test_non_writer_rank_vouches_for_the_span(self):
        self._spawn(skip_rank=True)


if __name__ == "__main__":
    unittest.main()
