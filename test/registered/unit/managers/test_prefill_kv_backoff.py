"""Prefill admission must not admit KV that eviction cannot free.

Under HiCache write_back with a full host pool, a device chain whose only leaf
can be neither backed up nor dropped (its host-only child is host-locked, as by
an in-flight storage prefetch) is counted by `evictable_size()` but cannot be
freed. Admission must defer instead of letting `alloc_paged_token_slots_extend`
raise "Prefill out of memory".
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    prefill_kv_backoff_stats,
)
from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    PAGE,
    insert_seq,
    match_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POOL = 256
LOCKED_TOKENS = 176  # held by running requests
CHAIN_TOKENS = 32  # the unlocked chain: two 16-token device nodes
CHILD_TOKENS = 2 * PAGE  # its host-only child; fills the host pool
FREE_TOKENS = 2 * PAGE


def _seq(start: int, num_tokens: int) -> list[int]:
    return list(range(start, start + num_tokens))


class _PrefillKvBackoffSuite:
    tree_core_backend: str

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        stats = prefill_kv_backoff_stats()
        self._stats_before = (stats.deferred, stats.shrunk_chunks, stats.skipped_chunks)

    def _stats_delta(self):
        stats = prefill_kv_backoff_stats()
        now = (stats.deferred, stats.shrunk_chunks, stats.skipped_chunks)
        return tuple(a - b for a, b in zip(now, self._stats_before))

    def _cache(self, host_tokens: int = CHILD_TOKENS):
        cache, allocator, host_pool = write_back_cache(
            self.tree_core_backend, device_tokens=POOL, host_tokens=host_tokens
        )
        return cache, allocator, host_pool

    def _stuck_state(self):
        """A locked majority, one unlocked chain whose leaf can be neither
        backed up nor dropped, and two free pages."""
        cache, allocator, host_pool = self._cache()
        parent = _seq(1, CHAIN_TOKENS // 2)
        chain = parent + _seq(500, CHAIN_TOKENS // 2)
        child_seq = chain + _seq(1000, CHILD_TOKENS)
        for seq in (parent, chain, child_seq):
            insert_seq(cache, allocator, seq)
        child = match_seq(cache, child_seq).last_device_node
        # write_back backs the child up alone, filling the host pool.
        cache.evict(EvictParams(num_tokens=CHILD_TOKENS))
        self.assertEqual(host_pool.available_size(), 0)
        self.assertEqual(match_seq(cache, child_seq).host_hit_length, CHILD_TOKENS)
        # An in-flight storage prefetch anchors on the host-only child.
        anchor = cache.inc_host_lock_ref(child).to_dec_params()

        locked = _seq(10_000, LOCKED_TOKENS)
        insert_seq(cache, allocator, locked)
        lock = cache.lock(match_seq(cache, locked).last_device_node)
        self.addCleanup(cache.unlock, lock)
        # The other free pages are running requests' private rows.
        self.assertIsNotNone(allocator.alloc(allocator.available_size() - FREE_TOKENS))
        self.assertEqual(cache.evictable_size(), CHAIN_TOKENS)
        return cache, allocator, child, anchor, chain

    def _req(self, rid, num_tokens, cache):
        req = mock.MagicMock(spec=Req)
        req.kv = ReqKvInfo()
        req.rid = str(rid)
        req.cache_request_handle = CacheRequestHandle(req.rid, 0)
        req.priority = 0
        req.prefix_len = 0
        type(req).extend_len = Req.extend_len
        req.extend_end = None
        req.full_untruncated_fill_ids = list(range(num_tokens))
        req.output_ids = []
        req.sampling_params = SimpleNamespace(max_new_tokens=1, ignore_eos=False)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.window_host_hit_lengths = {}
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        req.last_node = cache.root_node_handle()
        req.lock = None
        return req

    def _adder(self, cache, allocator, *, rem_chunk_tokens=None):
        return PrefillAdder(
            page_size=PAGE,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=SimpleNamespace(reqs=[]),
            new_token_ratio=1.0,
            rem_input_tokens=1024,
            rem_chunk_tokens=rem_chunk_tokens,
        )

    def _add(self, adder, req):
        return adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    def _alloc_batch(self, cache, reqs):
        """The KV allocation prepare_for_extend runs for an admitted batch."""
        prefix = torch.tensor([r.prefix_len for r in reqs], dtype=torch.int64)
        seq = torch.tensor([r.extend_end for r in reqs], dtype=torch.int64)
        return alloc_paged_token_slots_extend(
            tree_cache=cache,
            prefix_lens=prefix,
            prefix_lens_cpu=prefix,
            seq_lens=seq,
            seq_lens_cpu=seq,
            last_loc=torch.full((len(reqs),), -1, dtype=torch.int64),
            extend_num_tokens=int((seq - prefix).sum()),
        )

    def test_counted_evictable_chain_cannot_be_freed(self):
        cache, allocator, _, _, chain = self._stuck_state()
        evict_from_tree_cache(cache, FREE_TOKENS + CHAIN_TOKENS)
        self.assertEqual(allocator.available_size(), FREE_TOKENS)
        self.assertEqual(cache.evictable_size(), CHAIN_TOKENS)
        self.assertEqual(match_seq(cache, chain).device_prefix_len, CHAIN_TOKENS)

    def test_without_backoff_the_admitted_batch_raises(self):
        cache, allocator, _, _, _ = self._stuck_state()
        with envs.SGLANG_ENABLE_PREFILL_KV_BACKOFF.override(False):
            adder = self._adder(cache, allocator)
        req = self._req(1, 16, cache)
        self._add(adder, req)
        # The budget trusts available + evictable (8 + 32) and admits.
        self.assertEqual(adder.can_run_list, [req])
        with self.assertRaisesRegex(RuntimeError, "Prefill out of memory"):
            self._alloc_batch(cache, adder.can_run_list)
        cache.unlock(req.lock)

    def test_backoff_defers_then_admits_once_freeable(self):
        cache, allocator, child, anchor, chain = self._stuck_state()
        protected = cache.protected_size()
        req = self._req(1, 16, cache)
        adder = self._adder(cache, allocator)
        self.assertEqual(self._add(adder, req), AddReqResult.NO_TOKEN)
        # Deferred with nothing committed: no lock, no extend, no KV taken.
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(req.lock)
        self.assertIsNone(req.extend_end)
        self.assertEqual(cache.protected_size(), protected)
        self.assertEqual(allocator.available_size(), FREE_TOKENS)
        self.assertEqual(self._stats_delta(), (1, 0, 0))

        # The prefetch releases its anchor: the chain drops next pass, and the
        # same request is admitted and allocated.
        cache.dec_host_lock_ref(child, anchor)
        adder = self._adder(cache, allocator)
        self._add(adder, req)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(match_seq(cache, chain).device_prefix_len, 0)
        self.assertEqual(len(self._alloc_batch(cache, adder.can_run_list)), 16)
        cache.unlock(req.lock)

    def test_continuing_chunk_shrinks_then_waits(self):
        cache, allocator, _, _, _ = self._stuck_state()
        chunked = self._req(1, 64, cache)
        adder = self._adder(cache, allocator, rem_chunk_tokens=32)
        # Two free pages, one kept for page rounding: a one-page chunk.
        self.assertIs(adder.add_chunked_req(chunked), chunked)
        self.assertEqual(adder.can_run_list, [chunked])
        self.assertEqual(chunked.extend_end, PAGE)
        self._alloc_batch(cache, adder.can_run_list)
        self.assertEqual(self._stats_delta(), (0, 1, 0))

        # Not one page left: the continuation waits, and no new request starts.
        chunked.prefix_len, chunked.extend_end = chunked.extend_end, None
        adder = self._adder(cache, allocator, rem_chunk_tokens=32)
        self.assertIs(adder.add_chunked_req(chunked), chunked)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(
            self._add(adder, self._req(2, 4, cache)), AddReqResult.NO_TOKEN
        )
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(self._stats_delta(), (0, 1, 1))

    def test_no_shortfall_admits_the_same_batch_without_evicting(self):
        batches = []
        for enabled in (True, False):
            cache, allocator, _ = self._cache()
            with envs.SGLANG_ENABLE_PREFILL_KV_BACKOFF.override(enabled):
                adder = self._adder(cache, allocator, rem_chunk_tokens=64)
            reqs = [self._req(i, 24, cache) for i in range(3)]
            results = [self._add(adder, r) for r in reqs]
            self._alloc_batch(cache, adder.can_run_list)
            batches.append((results, [r.extend_end for r in adder.can_run_list]))
            self.assertEqual(cache.evictable_size(), 0)
            for r in adder.can_run_list:
                cache.unlock(r.lock)
        self.assertEqual(batches[0], batches[1])

    def test_one_reclaim_walk_per_pass(self):
        """The first shortfall reclaims for the whole pass, and the batch
        allocation then needs no walk of its own."""
        cache, allocator, _ = self._cache(host_tokens=POOL)
        for i in range(POOL // 32):
            insert_seq(cache, allocator, _seq(20_000 + 100 * i, 32))
        self.assertEqual(allocator.available_size(), 0)
        reqs = [self._req(i, 16, cache) for i in range(3)]
        with mock.patch.object(
            cache, "evict_for_alloc", wraps=cache.evict_for_alloc
        ) as evict:
            adder = self._adder(cache, allocator, rem_chunk_tokens=64)
            for r in reqs:
                self._add(adder, r)
            self.assertEqual(adder.can_run_list, reqs)
            self._alloc_batch(cache, adder.can_run_list)
            self.assertEqual(evict.call_count, 1)
        for r in reqs:
            cache.unlock(r.lock)

    def test_verdicts_match_on_replicated_stages(self):
        """PP stages replicate the tree and pool and run the same pass, so
        their back-off verdicts and batches must be identical."""
        outcomes = []
        for _stage in range(2):
            cache, allocator, child, anchor, _ = self._stuck_state()
            stage = []
            for round_id in range(2):
                if round_id == 1:
                    cache.dec_host_lock_ref(child, anchor)
                adder = self._adder(cache, allocator, rem_chunk_tokens=32)
                reqs = [self._req(i, 12, cache) for i in range(3)]
                results = [self._add(adder, r).name for r in reqs]
                stage.append(
                    (
                        results,
                        [r.rid for r in adder.can_run_list],
                        allocator.available_size(),
                    )
                )
                for r in adder.can_run_list:
                    cache.unlock(r.lock)
            outcomes.append(stage)
        self.assertEqual(outcomes[0], outcomes[1])
        # Round 0 defers everything; round 1 admits.
        self.assertEqual(outcomes[0][0][1], [])
        self.assertTrue(outcomes[0][1][1])


class TestPrefillKvBackoffPythonCore(_PrefillKvBackoffSuite, CustomTestCase):
    tree_core_backend = "python"


class TestPrefillKvBackoffRustCore(_PrefillKvBackoffSuite, CustomTestCase):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
