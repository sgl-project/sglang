"""Storage backup acks settle at the rank-agreed written prefix and retry the
unwritten tail; write-through backups short on host staging retry from the
tick. CPU only: the tree, the controller, its backup thread and the file
backend are real; the D2H transfer engine is faked (no device streams)."""

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
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import STORAGE_BACKUP_MAX_RETRIES
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    drop_hicache_atexit_pin,
    node_device_lock_ref,
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


class _TailRetryCase(unittest.TestCase):
    """Buffer mode on ``cfg``: a real cache with the file backend, one span of
    _PAGES pages inserted, storage writes spied."""

    cfg = _CFG

    def setUp(self):
        self._cleanup = contextlib.ExitStack()
        self.addCleanup(self._cleanup.close)
        self.storage_dir = tempfile.mkdtemp(prefix="hicache-backup-retry-")
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


class TestStorageBackupTailRetry(_TailRetryCase):
    """Beliefs follow the agreed written prefix; the tail is re-enqueued up
    to the bound, then given up and counted."""

    def test_failed_batch_feeds_prefix_beliefs_then_retries_tail(self):
        backend = self._fail_second_batch(budget=1)
        self.cache.request_buffer_backup(self.leaf)

        # First ack: only the written prefix is believed; the tail is in
        # flight again as one new operation and the staging stays.
        _tick_until(
            self.cache,
            lambda: any(e.attempts == 1 for e in self.pipeline.ongoing_backup.values()),
        )
        self.assertTrue(self.beliefs.contains_all(self.chain[:STORAGE_BATCH_SIZE]))
        self.assertFalse(self.beliefs.contains(self.chain[STORAGE_BATCH_SIZE]))
        self.assertTrue(self.beliefs.covered(self.chain))
        self.assertTrue(self.pipeline.backup_pending(self.leaf))
        self.assertEqual(self.pipeline.write_staged_tokens_, _PAGES)
        self.assertEqual(_write_spans(self.writes), [(_PAGES, _PAGES), (_TAIL, _TAIL)])
        self.assertEqual(
            self.writes.call_args_list[1].args[2], self.chain[STORAGE_BATCH_SIZE:]
        )

        _tick_until(self.cache, self.pipeline.is_idle)
        self.assertTrue(self.beliefs.contains_all(self.chain))
        self.assertFalse(self.pipeline.backup_pending(self.leaf))
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.assertEqual(self.writes.call_count, 2)
        self.assertEqual(backend.failed_batches, 1)
        self.assertEqual(backend.batch_exists(self.chain), _PAGES)
        self.assertEqual(self._backuped_tokens(), [STORAGE_BATCH_SIZE, _TAIL])
        self.collector.log_storage_backup_failed_tokens.assert_not_called()
        self.cache.sanity_check()

    def test_retries_exhausted_counts_terminal_failure(self):
        backend = self._fail_second_batch(budget=None)
        self.cache.request_buffer_backup(self.leaf)

        with self.assertLogs(_CACHE_LOGGER, level="ERROR") as logs:
            _tick_until(self.cache, self.pipeline.is_idle)
        self.assertEqual(sum("gave up" in line for line in logs.output), 1)
        self.assertEqual(
            _write_spans(self.writes),
            [(_PAGES, _PAGES)] + [(_TAIL, _TAIL)] * STORAGE_BACKUP_MAX_RETRIES,
        )
        self.assertEqual(backend.failed_batches, 1 + STORAGE_BACKUP_MAX_RETRIES)
        # The prefix is believed; the tail is neither believed nor in flight,
        # so the next insert of this span writes it again.
        self.assertTrue(self.beliefs.contains_all(self.chain[:STORAGE_BATCH_SIZE]))
        self.assertFalse(self.beliefs.covered(self.chain))
        self.assertFalse(self.pipeline.backup_pending(self.leaf))
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.assertEqual(backend.batch_exists(self.chain), STORAGE_BATCH_SIZE)
        self.collector.log_storage_backup_failed_tokens.assert_called_once_with(_TAIL)
        self.assertEqual(self._backuped_tokens(), [STORAGE_BATCH_SIZE, 0, 0])
        # No further enqueue past the bound.
        for _ in range(3):
            self.cache.check_hicache_events()
        self.assertEqual(self.writes.call_count, 1 + STORAGE_BACKUP_MAX_RETRIES)
        self.assertEqual(self.cc.backup_queue.qsize(), 0)
        self.cache.sanity_check()


class TestStorageBackupTailRetryPagedKeys(_TailRetryCase):
    """page_size 4: the re-enqueued tail starts at the agreed page count times
    the page size in host indices and token ids, and at the page count in the
    hash chain; the agreed token count stays page-aligned through the drain."""

    cfg = _CFG_PAGED

    def test_tail_is_sliced_by_pages(self):
        page_size = self.cfg.page_size
        written_tokens = STORAGE_BATCH_SIZE * page_size
        backend = self._fail_second_batch(budget=1)
        self.cache.request_buffer_backup(self.leaf)

        _tick_until(
            self.cache,
            lambda: any(e.attempts == 1 for e in self.pipeline.ongoing_backup.values()),
        )
        self.assertTrue(self.beliefs.contains_all(self.chain[:STORAGE_BATCH_SIZE]))
        self.assertFalse(self.beliefs.contains(self.chain[STORAGE_BATCH_SIZE]))
        self.assertEqual(self.pipeline.write_staged_tokens_, _PAGES * page_size)
        self.assertEqual(
            _write_spans(self.writes),
            [(_PAGES * page_size, _PAGES), (_TAIL * page_size, _TAIL)],
        )
        first, retry = self.writes.call_args_list
        self.assertTrue(torch.equal(retry.args[0], first.args[0][written_tokens:]))
        self.assertEqual(list(retry.args[1]), list(first.args[1][written_tokens:]))
        self.assertEqual(retry.args[2], self.chain[STORAGE_BATCH_SIZE:])

        _tick_until(self.cache, self.pipeline.is_idle)
        self.assertTrue(self.beliefs.contains_all(self.chain))
        self.assertEqual(self.pipeline.write_staged_tokens_, 0)
        self.assertEqual(backend.failed_batches, 1)
        self.assertEqual(backend.batch_exists(self.chain), _PAGES)
        self.assertEqual(self._backuped_tokens(), [written_tokens, _TAIL * page_size])
        self.collector.log_storage_backup_failed_tokens.assert_not_called()
        self.cache.sanity_check()


class TestHostCapacityBackupRetry(unittest.TestCase):
    """Cache mode: a write-through node whose host staging allocation failed
    is retried from the tick and backed up once capacity returns."""

    def setUp(self):
        self._cleanup = contextlib.ExitStack()
        self.addCleanup(self._cleanup.close)
        storage_dir = tempfile.mkdtemp(prefix="hicache-backup-retry-")
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        self.cache, self.allocator, _ = _build_cache(
            self._cleanup, host_memory_mode="cache", storage_dir=storage_dir
        )
        self.cc = self.cache.cache_controller
        self.leaf = _insert_span(self.cache, self.allocator, _PAGES)
        host_pool = self.cc.mem_pool_host
        # Leave less than the span free, with nothing host-evictable.
        self.held = host_pool.alloc(host_pool.size - _PAGES // 2)
        self.assertIsNotNone(self.held)

    def _parked(self):
        """Parked node -> tick retries spent; a parked node is device-locked."""
        parked = self.cache._host_capacity_backup_retries
        for node_id in parked:
            self.assertEqual(
                node_device_lock_ref(self.cache, node_id, ComponentType.FULL),
                self.base_lock_ref + 1,
            )
        return {node_id: entry.attempts for node_id, entry in parked.items()}

    def _trigger_backup(self):
        self.base_lock_ref = node_device_lock_ref(
            self.cache, self.leaf, ComponentType.FULL
        )
        written = self.cache._execute_and_commit_kv_backup(
            self.cache.tree_core.build_backup_kv_action(self.leaf)
        )
        self.assertEqual(written, 0)
        self.assertEqual(self._parked(), {self.leaf: 0})
        self.assertFalse(self.cache.tree_core.is_backuped(self.leaf))

    def test_retried_from_tick_once_capacity_returns(self):
        collector = StorageMetricsCollector(labels={"fixture": "host_capacity"})
        self.cache.storage_metrics_collector = collector
        self.cache.enable_storage_metrics = True
        self._trigger_backup()

        self.cache.check_hicache_events()
        self.assertEqual(self._parked(), {self.leaf: 1})
        self.assertFalse(self.cache.tree_core.is_backuped(self.leaf))

        self.cc.mem_pool_host.free(self.held)
        self.cache.check_hicache_events()
        self.assertTrue(self.cache.tree_core.is_backuped(self.leaf))
        self.assertEqual(self._parked(), {})
        self.assertEqual(self.cache.ongoing_write_through, {})
        # The write-through took and released its own lock; the park lock is gone.
        self.assertEqual(
            node_device_lock_ref(self.cache, self.leaf, ComponentType.FULL),
            self.base_lock_ref,
        )
        retries = collector.storage_backup_host_capacity_retries_total.labels(
            fixture="host_capacity"
        )
        self.assertEqual(retries._value.get(), 2)
        # The host ack chained the L3 write as usual.
        _tick_until(self.cache, lambda: not self.cache.ongoing_backup)
        chain = self.cache.tree_core.get_hash_values(self.leaf)
        self.assertEqual(self.cc.storage_backend.batch_exists(chain), _PAGES)
        self.cache.sanity_check()

    def test_bound_gives_up_with_one_error(self):
        self.cache.storage_metrics_collector = mock.Mock()
        self.cache.enable_storage_metrics = True
        self._trigger_backup()

        with self.assertLogs(_CACHE_LOGGER, level="ERROR") as logs:
            for _ in range(STORAGE_BACKUP_MAX_RETRIES):
                self.cache.check_hicache_events()
        self.assertEqual(sum("gave up" in line for line in logs.output), 1)
        self.assertEqual(self._parked(), {})
        self.assertEqual(
            node_device_lock_ref(self.cache, self.leaf, ComponentType.FULL),
            self.base_lock_ref,
        )
        self.assertEqual(
            self.cache.storage_metrics_collector.log_storage_backup_host_capacity_retry.call_count,
            STORAGE_BACKUP_MAX_RETRIES,
        )
        # Past the bound the node waits for its next hit, not for the tick.
        self.cc.mem_pool_host.free(self.held)
        self.cache.check_hicache_events()
        self.assertFalse(self.cache.tree_core.is_backuped(self.leaf))
        self.cache.sanity_check()

    def test_hit_while_parked_keeps_one_lock_and_one_entry(self):
        self.cache.storage_metrics_collector = mock.Mock()
        self.cache.enable_storage_metrics = True
        self._trigger_backup()

        # A later insert of the span hits the parked leaf; it is still not
        # backuped, so the hit fires its own BackupKV through the insert
        # path, and that attempt fails on host staging again. The park is
        # idempotent: one entry, one device lock, the retry count untouched.
        self.cache.write_through_threshold = 1
        with mock.patch.object(
            self.cache, "_execute_kv_backup", wraps=self.cache._execute_kv_backup
        ) as attempts:
            hit = _insert_span(self.cache, self.allocator, _PAGES)
        self.assertEqual(hit, self.leaf)
        self.assertEqual(attempts.call_count, 1)
        self.assertFalse(self.cache.tree_core.is_backuped(self.leaf))
        self.assertEqual(self._parked(), {self.leaf: 0})

        self.cache.check_hicache_events()
        self.assertEqual(self._parked(), {self.leaf: 1})

        # The retry runs once capacity returns; the single park lock is gone.
        self.cc.mem_pool_host.free(self.held)
        self.cache.check_hicache_events()
        self.assertTrue(self.cache.tree_core.is_backuped(self.leaf))
        self.assertEqual(self._parked(), {})
        self.assertEqual(
            node_device_lock_ref(self.cache, self.leaf, ComponentType.FULL),
            self.base_lock_ref,
        )
        self.assertEqual(
            self.cache.storage_metrics_collector.log_storage_backup_host_capacity_retry.call_count,
            2,
        )
        _tick_until(self.cache, lambda: not self.cache.ongoing_backup)
        self.cache.sanity_check()


def _two_rank_tail_retry(rank, rendezvous_dir, skip_rank):
    """One rank's backend fails the second batch once. With ``skip_rank``
    rank 1 is not the KV storage writer (replicated MLA KV: it writes nothing
    and acks with completed_tokens 0) and rank 0 is the failing writer;
    otherwise rank 1 fails its own write."""
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
            budget=1,
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
            # The in-flight operation ids, their attempts and every enqueued
            # span must match on both ranks after every tick: the ack drain
            # pairs operations by position.
            snapshot = (
                sorted(cache.ongoing_backup),
                sorted(cache._backup_write_specs),
                [spec.attempts for spec in cache._backup_write_specs.values()],
                _write_spans(writes),
            )
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

        # Both ranks re-enqueued the same tail although only one rank failed;
        # a non-writer's empty ack never shortened the agreed prefix.
        assert _write_spans(writes) == [(_PAGES, _PAGES), (_TAIL, _TAIL)], _write_spans(
            writes
        )
        assert backend.failed_batches == (1 if rank == failing_rank else 0)
        assert backend.batch_exists(chain) == (_PAGES if writer else 0)
        assert node_host_lock_ref(cache, leaf, ComponentType.FULL) == 0
        assert not cache._backup_write_specs
        cache.sanity_check()
    finally:
        cleanup.close()
        torch.distributed.destroy_process_group()


class TestTwoRankTailRetry(unittest.TestCase):
    def _spawn(self, skip_rank):
        with tempfile.TemporaryDirectory(prefix="hicache-backup-retry-") as directory:
            torch.multiprocessing.spawn(
                _two_rank_tail_retry, args=(directory, skip_rank), nprocs=2, join=True
            )

    def test_rank_local_failure_re_enqueues_the_tail_on_both_ranks(self):
        self._spawn(skip_rank=False)

    def test_non_writer_rank_vouches_for_the_span(self):
        self._spawn(skip_rank=True)


if __name__ == "__main__":
    unittest.main()
