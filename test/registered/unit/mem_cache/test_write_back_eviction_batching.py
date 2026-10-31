"""Write-back device eviction: one wait per batch of copies.

Drives a real UnifiedRadixCache (both tree cores) on CPU with a fake
write-back controller whose D->H copies complete immediately: one fenced
submission and one wait per batch, no device page a copy reads freed before
that wait, host room made per victim, and the load-back chain check.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    BATCH_TOKENS,
    PAGE,
    SEQ_TOKENS,
    SEQS_PER_BATCH,
    InstantWriteBackController,
    batch_seqs,
    hybrid_write_back_cache,
    insert_hybrid_seq,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FencedController(InstantWriteBackController):
    """Submits D->H copies through the controller's own start_writing, so a
    copy goes behind its write fence, onto streams and events that log what
    happens: ("fence",) when the D->H stream waits on the forward stream,
    ("submit", n) for the n-th submission and ("sync", n) when its finish
    event is waited on. ``submitted[n]`` maps each pool to the device pages
    submission n reads."""

    start_writing = HybridCacheController.start_writing
    _l2_transfers = HybridCacheController._l2_transfers
    _num_tokens_by_pool = HybridCacheController._num_tokens_by_pool

    def __init__(self, controller, pool_names):
        super().__init__(controller.mem_pool_host, controller.mem_pool_device_allocator)
        self.log = []
        self.submitted = []
        self._pool_names = pool_names
        forward = SimpleNamespace()
        self.write_fence_stream = forward

        def wait_stream(stream):
            assert stream is forward
            self.log.append(("fence",))

        self.l2_transfer_engine = SimpleNamespace(
            batch_direct=False,
            device_to_host_stream=SimpleNamespace(wait_stream=wait_stream),
            submit_device_to_host=self._submit,
        )

    def _move_write_operation(self, op):
        return op.host_indices, op.device_indices, op.pool_transfers

    def _transfer_num_bytes(self, op):
        return 0

    def _submit(self, transfers):
        n = len(self.submitted)
        pages = {}
        for transfer in transfers:
            pool = self._pool_names[id(transfer.host_pool)]
            pages.setdefault(pool, set()).update(
                (transfer.device_indices // PAGE).tolist()
            )
        self.submitted.append(pages)
        self.log.append(("submit", n))
        finish = SimpleNamespace(
            synchronize=lambda: self.log.append(("sync", n)), query=lambda: True
        )
        return SimpleNamespace(
            start_event=None, finish_event=finish, timing_enabled=False
        )

    def queued_pages(self, pool):
        """Device pages the queued, not yet submitted, copies read in ``pool``."""
        pages = set()
        for op in self.write_queue:
            if pool == "kv":
                pages.update((op.device_indices // PAGE).tolist())
            for transfer in op.pool_transfers or ():
                if transfer.name.value == pool:
                    pages.update((transfer.device_indices // PAGE).tolist())
        return pages

    def record_frees(self, pool, allocator):
        """Log ("free", pool, pages) for each of ``allocator``'s page frees,
        after checking that no queued copy still reads those pages."""
        release_page_ids = allocator._release_page_ids

        def free(*page_ids):
            pages = {page for ids in page_ids for page in ids.tolist()}
            assert not pages & self.queued_pages(pool), (
                f"freed {pool} pages a queued copy reads"
            )
            self.log.append(("free", pool, pages))
            return release_page_ids(*page_ids)

        return mock.patch.object(allocator, "_release_page_ids", side_effect=free)

    def assert_copies_behind_the_fence(self, test):
        """Every submission follows a fence wait, and a page a submission reads
        is freed only after its finish event was waited on."""
        synced = set()
        for i, event in enumerate(self.log):
            if event[0] == "submit":
                test.assertEqual(self.log[i - 1], ("fence",))
            elif event[0] == "sync":
                synced.add(event[1])
            elif event[0] == "free":
                _, pool, pages = event
                for n, read in enumerate(self.submitted):
                    if pages & read.get(pool, set()):
                        test.assertIn(n, synced, f"freed {pool} pages copy {n} reads")


class _WriteBackBatchingSuite:
    tree_core_backend: str

    def _fill_host_with_one_batch(self):
        """Back up one batch to host, filling the host pool exactly."""
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=BATCH_TOKENS,
        )
        for seq in batch_seqs(0):
            insert_seq(cache, allocator, seq)
        cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
        self.assertEqual(host_pool.available_size(), 0)
        for seq in batch_seqs(0):
            self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
        return cache, allocator, host_pool

    def _record_host_work(self, cache, host_pool):
        """Record each host eviction's and host allocation's size."""
        evict_host_requests, host_allocs = [], []
        evict_host, alloc = cache.evict_host, host_pool.alloc

        def record_evict(num_tokens, component_type=ComponentType.FULL, **labels):
            evict_host_requests.append(num_tokens)
            return evict_host(num_tokens, component_type, **labels)

        def record_alloc(need_size):
            host_allocs.append(need_size)
            return alloc(need_size)

        patches = (
            mock.patch.object(cache, "evict_host", side_effect=record_evict),
            mock.patch.object(host_pool, "alloc", side_effect=record_alloc),
        )
        return evict_host_requests, host_allocs, patches

    def _count_waits(self, cache):
        """Record how many queued copies each blocking wait drains."""
        waits = []
        writing_check = cache.writing_check

        def count_waits(write_back=False, finish_count=None):
            if write_back and cache.ongoing_write_through:
                waits.append(len(cache.ongoing_write_through))
            return writing_check(write_back=write_back, finish_count=finish_count)

        return waits, mock.patch.object(cache, "writing_check", side_effect=count_waits)

    def test_batch_copies_behind_one_wait(self):
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        for seq in batch_seqs(1):
            insert_seq(cache, allocator, seq)

        waits, patch = self._count_waits(cache)
        with patch:
            result = cache.evict(EvictParams(num_tokens=BATCH_TOKENS))

        # One wait for the batch's eight copies, not one per victim.
        self.assertEqual(waits, [SEQS_PER_BATCH])
        self.assertEqual(result.num_tokens_evicted, BATCH_TOKENS)
        # The new batch replaced the old one on host, nothing more.
        self.assertEqual(host_pool.available_size(), 0)
        for seq in batch_seqs(0):
            self.assertEqual(match_seq(cache, seq).host_hit_length, 0)
        for seq in batch_seqs(1):
            match = match_seq(cache, seq)
            self.assertEqual(match.device_prefix_len, 0)
            self.assertEqual(match.host_hit_length, SEQ_TOKENS)
        cache.sanity_check()

    def test_batch_copies_go_behind_the_write_fence_in_one_submission(self):
        """The batch's copies are submitted together, behind the forward
        stream's write fence, and its victims' device pages are freed only
        after that submission is waited on."""
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        for seq in batch_seqs(1):
            insert_seq(cache, allocator, seq)
        victim_pages = {
            page
            for seq in batch_seqs(1)
            for page in (
                cache.path_device_indices(match_seq(cache, seq).last_device_node)
                // PAGE
            ).tolist()
        }
        cc = _FencedController(cache.cache_controller, {id(host_pool): "kv"})
        cache.cache_controller = cc

        with cc.record_frees("kv", allocator):
            result = cache.evict(EvictParams(num_tokens=BATCH_TOKENS))

        self.assertEqual(result.num_tokens_evicted, BATCH_TOKENS)
        self.assertEqual(cc.submitted, [{"kv": victim_pages}])
        self.assertTrue(any(event[0] == "free" for event in cc.log))
        cc.assert_copies_behind_the_fence(self)
        cache.sanity_check()

    def test_swa_window_copies_go_behind_the_write_fence(self):
        """A chain's SWA window copy also reads its parent's SWA slots: the
        batch, parents included, goes behind the fence in one submission, and
        no Full or SWA page a copy reads is freed before that is waited on."""
        cache, allocator, kv_host, swa_host = hybrid_write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            swa_device_tokens=4 * BATCH_TOKENS,
            host_tokens=4 * BATCH_TOKENS,
            swa_host_tokens=4 * BATCH_TOKENS,
            sliding_window=3 * PAGE,
        )
        chains = batch_seqs(0)[:2]
        for seq in chains:
            insert_hybrid_seq(cache, allocator, seq[:PAGE])
            insert_hybrid_seq(cache, allocator, seq)
        cc = _FencedController(
            cache.cache_controller, {id(kv_host): "kv", id(swa_host): "swa"}
        )
        cache.cache_controller = cc

        with (
            cc.record_frees("kv", allocator.full_attn_allocator),
            cc.record_frees("swa", allocator.swa_attn_allocator),
        ):
            result = cache.evict(EvictParams(num_tokens=2 * SEQ_TOKENS))

        self.assertEqual(result.num_tokens_evicted, 2 * SEQ_TOKENS)
        self.assertEqual(len(cc.submitted), 1)
        self.assertEqual(len(cc.submitted[0]["kv"]), 2 * SEQ_TOKENS // PAGE)
        self.assertTrue(cc.submitted[0]["swa"])
        self.assertTrue(
            any(event[1] == "swa" for event in cc.log if event[0] == "free")
        )
        cc.assert_copies_behind_the_fence(self)
        for seq in chains:
            self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
        cache.sanity_check()

    def _evict_until_reclaimed(self, *, batched):
        """evict_for_alloc with an allocator's reclaim predicate, met three
        victims into a quota of a whole batch; returns the tokens evicted and
        which sequences stay on device."""
        cache, allocator, _ = self._fill_host_with_one_batch()
        for seq in batch_seqs(1):
            insert_seq(cache, allocator, seq)
        if not batched:
            cache._write_back_batch_tokens = lambda *args: 0
        target = allocator.available_size() + 3 * SEQ_TOKENS
        result = cache.evict_for_alloc(
            EvictParams(num_tokens=BATCH_TOKENS),
            allocation_reclaim_satisfied=lambda: allocator.available_size() >= target,
        )
        cache.sanity_check()
        on_device = [
            match_seq(cache, seq).device_prefix_len == SEQ_TOKENS
            for seq in batch_seqs(1)
        ]
        return result.num_tokens_evicted, on_device

    def test_a_reclaim_predicate_stops_the_walk_at_the_per_victim_victim(self):
        """An allocator's reclaim predicate reads its free space, which only a
        demote changes, so the walk demotes each victim before the next."""
        batched = self._evict_until_reclaimed(batched=True)
        self.assertEqual(batched, self._evict_until_reclaimed(batched=False))
        self.assertEqual(batched[0], 3 * SEQ_TOKENS)

    def test_a_batch_that_ends_early_evicts_host_for_its_victims_only(self):
        """Host room is made per victim, so a batch that meets its request
        early leaves the rest of the host pool alone."""
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        for seq in batch_seqs(1)[:2]:
            insert_seq(cache, allocator, seq)

        result = cache.evict(EvictParams(num_tokens=2 * SEQ_TOKENS))
        self.assertEqual(result.num_tokens_evicted, 2 * SEQ_TOKENS)
        self.assertEqual(host_pool.available_size(), 0)
        on_host = sum(
            match_seq(cache, seq).host_hit_length == SEQ_TOKENS for seq in batch_seqs(0)
        )
        self.assertEqual(on_host, SEQS_PER_BATCH - 2)
        cache.sanity_check()

    def test_need_beyond_the_evictable_device_keeps_the_rest_on_host(self):
        """A batch makes host room only for what the device can still evict."""
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        for seq in batch_seqs(1)[:2]:
            insert_seq(cache, allocator, seq)

        result = cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
        self.assertEqual(result.num_tokens_evicted, 2 * SEQ_TOKENS)
        self.assertEqual(host_pool.available_size(), 0)
        on_host = sum(
            match_seq(cache, seq).host_hit_length == SEQ_TOKENS for seq in batch_seqs(0)
        )
        self.assertEqual(on_host, SEQS_PER_BATCH - 2)
        cache.sanity_check()

    def test_chain_parents_join_their_leaves_batch(self):
        """A parent whose device children are all queued follows them into
        their batch, as it follows them when each victim is demoted before the
        walk advances: one wait covers the chain."""
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        chains = batch_seqs(1)[:2]
        for seq in chains:
            insert_seq(cache, allocator, seq[:PAGE])
            insert_seq(cache, allocator, seq)

        waits, patch = self._count_waits(cache)
        with patch:
            result = cache.evict(EvictParams(num_tokens=2 * SEQ_TOKENS))

        self.assertEqual(result.num_tokens_evicted, 2 * SEQ_TOKENS)
        self.assertEqual(waits, [2 * len(chains)])
        self.assertEqual(host_pool.available_size(), 0)
        for seq in chains:
            self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
        cache.sanity_check()

    def test_a_failed_backup_falls_back_to_the_drop_within_a_batch(self):
        """A victim whose write fails falls back to the drop, whose frees make
        host room for the next victim of the same batch."""
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=BATCH_TOKENS,
        )
        # Unbacked parent over a host-only child and grandchild.
        parent = [1, 2, 3, 4]
        grandchild = parent + [5, 6, 7, 8] + [9, 10, 11, 12]
        insert_seq(cache, allocator, parent)
        insert_seq(cache, allocator, grandchild[: 2 * PAGE])
        insert_seq(cache, allocator, grandchild)
        cache.evict(EvictParams(num_tokens=2 * PAGE))
        self.assertEqual(match_seq(cache, grandchild).host_hit_length, 2 * PAGE)
        # A younger unbacked leaf, and no free host slot left.
        younger = [100 + t for t in range(2 * PAGE)]
        insert_seq(cache, allocator, younger)
        self.assertIsNotNone(host_pool.alloc(host_pool.available_size()))

        assert cache.cache_controller is not None
        real_write = cache.cache_controller.write
        writes = []

        def fail_first(*args, **kwargs):
            writes.append(args)
            if len(writes) == 1:
                return None
            return real_write(*args, **kwargs)

        with mock.patch.object(cache.cache_controller, "write", side_effect=fail_first):
            result = cache.evict(EvictParams(num_tokens=3 * PAGE))

        self.assertGreaterEqual(result.num_tokens_evicted, 3 * PAGE)
        self.assertEqual(len(writes), 2)
        self.assertEqual(match_seq(cache, parent).host_hit_length, 0)
        match = match_seq(cache, younger)
        self.assertEqual(match.device_prefix_len, 0)
        self.assertEqual(match.host_hit_length, 2 * PAGE)
        cache.sanity_check()

    def test_steady_churn_keeps_host_and_device_accounting(self):
        cache, allocator, host_pool = self._fill_host_with_one_batch()
        for batch in range(1, 6):
            for seq in batch_seqs(batch):
                insert_seq(cache, allocator, seq)
            cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
            self.assertEqual(host_pool.available_size(), 0)
            for seq in batch_seqs(batch):
                self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
            cache.sanity_check()

    def _evict_past_pinned_copies(self, *, walk_order, num_tokens, batched):
        """Evict unbacked leaves and backed-up leaves whose host copies are
        host-locked, so host eviction cannot reclaim them, with a host pool a
        quarter free and otherwise holding host-only leaves. ``walk_order``
        lists (count, pinned) groups, least recently used first. Returns how
        many host-only leaves stay on host and the host slots left free."""
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=2 * BATCH_TOKENS,
        )
        host_only = batch_seqs(0)
        for seq in host_only:
            insert_seq(cache, allocator, seq)
        cache.evict(EvictParams(num_tokens=BATCH_TOKENS))
        unbacked, pinned = iter(batch_seqs(1)), iter(batch_seqs(2))
        pinned_seqs, locks = [], []
        for count, is_pinned in walk_order:
            for _ in range(count):
                seq = next(pinned if is_pinned else unbacked)
                insert_seq(cache, allocator, seq)
                if is_pinned:
                    node = match_seq(cache, seq).last_device_node
                    self.assertTrue(cache.backup_node_for_write_back(node))
                    locks.append((node, cache.inc_host_lock_ref(node).to_dec_params()))
                    pinned_seqs.append(seq)
        self.assertEqual(host_pool.available_size(), BATCH_TOKENS // 2)
        if not batched:
            # The per-victim path: copy and demote each victim on its own.
            cache._write_back_batch_tokens = lambda *args: 0
        result = cache.evict(EvictParams(num_tokens=num_tokens))
        self.assertGreaterEqual(result.num_tokens_evicted, num_tokens)
        for seq in pinned_seqs:
            self.assertEqual(match_seq(cache, seq).host_hit_length, SEQ_TOKENS)
        for node, params in locks:
            cache.dec_host_lock_ref(node, params)
        cache.sanity_check()
        host_only_left = sum(
            match_seq(cache, seq).host_hit_length == SEQ_TOKENS for seq in host_only
        )
        return host_only_left, host_pool.available_size()

    def test_pinned_copies_cost_no_host_only_leaves(self):
        """A pinned victim demotes without a backup, so a batch must not
        evict host-only leaves for it, wherever the walk meets it."""
        half = SEQS_PER_BATCH // 2
        for walk_order, num_tokens, host_only_left in (
            # The unbacked victims' backups need 64 slots and 32 are free.
            (((SEQS_PER_BATCH, False), (half, True)), 3 * half * SEQ_TOKENS, half),
            # The pinned victims cover half the need, the free slots the rest;
            # the batch reserves before the walk reaches them.
            (((half, False), (half, True), (half, False)), BATCH_TOKENS, 2 * half),
        ):
            with self.subTest(walk_order=walk_order):
                per_victim = self._evict_past_pinned_copies(
                    walk_order=walk_order, num_tokens=num_tokens, batched=False
                )
                batched = self._evict_past_pinned_copies(
                    walk_order=walk_order, num_tokens=num_tokens, batched=True
                )
                self.assertEqual(batched, per_victim)
                self.assertEqual(batched, (host_only_left, 0))

    def _backed_history(self):
        """A two-node history chain whose host copies coexist with its device
        copies, as a load-back leaves them; the host pool otherwise holds
        host-only leaves and one sequence's free slots. Returns the cache, its
        host pool, the host-only sequences, and the chain's key."""
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=BATCH_TOKENS,
        )
        host_only = batch_seqs(0)[: SEQS_PER_BATCH - 2]
        for seq in host_only:
            insert_seq(cache, allocator, seq)
        cache.evict(EvictParams(num_tokens=len(host_only) * SEQ_TOKENS))
        history = batch_seqs(1)[0]
        for prefix in (history[:PAGE], history):
            insert_seq(cache, allocator, prefix)
            node = match_seq(cache, prefix).last_device_node
            self.assertTrue(cache.backup_node_for_write_back(node))
        self.assertEqual(host_pool.available_size(), SEQ_TOKENS)
        return cache, host_pool, host_only, history

    def _evict_recording_host_work(self, cache, host_pool, num_tokens):
        """Evict ``num_tokens``; returns the result, each host eviction's and
        host allocation's size, and each D->H write's Full token count."""
        evict_host_requests, host_allocs, patches = self._record_host_work(
            cache, host_pool
        )
        written = []
        assert cache.cache_controller is not None
        real_write = cache.cache_controller.write

        def record_write(device_value, **kwargs):
            written.append(len(device_value))
            return real_write(device_value, **kwargs)

        with (
            patches[0],
            patches[1],
            mock.patch.object(
                cache.cache_controller, "write", side_effect=record_write
            ),
        ):
            result = cache.evict(EvictParams(num_tokens=num_tokens))
        self.assertEqual(result.num_tokens_evicted, num_tokens)
        return evict_host_requests, host_allocs, written

    def _assert_on_host(self, cache, seqs):
        for seq in seqs:
            match = match_seq(cache, seq)
            self.assertEqual(match.device_prefix_len, 0)
            self.assertEqual(match.host_hit_length, len(seq))

    def test_backed_chain_demotes_without_reserving_or_reclaiming(self):
        cache, host_pool, host_only, chain = self._backed_history()
        work = self._evict_recording_host_work(cache, host_pool, len(chain))
        self.assertEqual(work, ([], [], []))
        self.assertEqual(host_pool.available_size(), SEQ_TOKENS)
        self._assert_on_host(cache, host_only + [chain])
        cache.sanity_check()

    def test_demote_fails_while_the_backup_reads_the_device_slots(self):
        cache, allocator, _ = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=BATCH_TOKENS,
        )
        (seq,) = batch_seqs(0)[:1]
        insert_seq(cache, allocator, seq)
        node = match_seq(cache, seq).last_device_node
        # Queued, not acknowledged: the host value is already committed.
        written = cache._execute_and_commit_kv_backup(
            BackupKV(node_ids=[node]), write_back=True
        )
        self.assertEqual(written, SEQ_TOKENS)
        self.assertTrue(cache.is_backuped(node))
        with self.assertRaisesRegex(
            (AssertionError, RuntimeError), "still reads its device slots"
        ):
            cache.tree_core.demote(node)

    def test_load_back_commit_rejects_indices_that_miss_the_chain(self):
        for device_tokens in (2 * SEQ_TOKENS, SEQ_TOKENS - PAGE):
            with self.subTest(device_tokens=device_tokens):
                cache, allocator, _ = write_back_cache(
                    self.tree_core_backend,
                    device_tokens=4 * BATCH_TOKENS,
                    host_tokens=BATCH_TOKENS,
                )
                (seq,) = batch_seqs(0)[:1]
                insert_seq(cache, allocator, seq)
                cache.evict(EvictParams(num_tokens=SEQ_TOKENS))
                node = match_seq(cache, seq).last_host_node
                kv_xfer, comp_xfers = cache.tree_core.build_load_back_spec(node)
                assert kv_xfer.host_indices is not None
                self.assertEqual(len(kv_xfer.host_indices), SEQ_TOKENS)
                device_indices = allocator.alloc(device_tokens)
                assert device_indices is not None
                # The Python core asserts; the Rust core's panic surfaces as a
                # RuntimeError at the binding.
                with self.assertRaisesRegex(
                    (AssertionError, RuntimeError),
                    f"load-back filled {device_tokens} device slots for a chain "
                    f"of {SEQ_TOKENS} host tokens",
                ):
                    cache.tree_core.commit_load_back(
                        node, device_indices, kv_xfer, comp_xfers
                    )


class TestWriteBackBatchingRustCore(_WriteBackBatchingSuite, CustomTestCase):
    tree_core_backend = "rust"


class TestWriteBackBatchingPythonCore(_WriteBackBatchingSuite, CustomTestCase):
    tree_core_backend = "python"


if __name__ == "__main__":
    unittest.main()
