"""Test steps for SWA KV kept only for the window before each cached Mamba checkpoint."""

import random
from dataclasses import replace

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InitLoadBackParams
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    match_prefix,
    node_device_value,
    node_key_length,
    node_parent,
)
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures


class SWACheckpointWindowFixtures(UnifiedRadixCacheFixtures):
    def _build(self, write_policy=None, swa_kv_size=None):
        cache, allocator, pool = build_fixture(
            replace(self.cfg, swa_kv_size=swa_kv_size)
        )
        if write_policy is not None:
            self._init_hicache(cache, write_policy=write_policy)
            if write_policy == "write_through":
                cache.write_through_threshold = 1
        self.page = self.cfg.page_size
        self.window = -(-self.cfg.sliding_window_size // self.page) * self.page
        return cache, allocator, pool

    def _seq(self, start, num_tokens):
        return self._make_seq(start, num_tokens // self.page)

    def _insert_seq(self, cache, allocator, pool, seq):
        return self._insert(cache, allocator, pool, seq).last_device_node

    def _path(self, cache, node_id, end):
        """(start, end, has SWA KV, has checkpoint) of each node from the root."""
        root, spans = cache.root_node_handle(), []
        while node_id != root:
            start = end - node_key_length(cache, node_id)
            spans.append(
                (
                    start,
                    end,
                    node_device_value(cache, node_id, ComponentType.SWA) is not None,
                    node_device_value(cache, node_id, ComponentType.MAMBA) is not None,
                )
            )
            node_id, end = node_parent(cache, node_id), start
        return spans[::-1]

    def _two_overlapping_checkpoints(self, cache, allocator, pool):
        seq_a = self._seq(1, 2 * self.window + self.page)
        seq_b = seq_a + self._seq(1000, self.page)
        self._insert_seq(cache, allocator, pool, seq_a)
        return len(seq_a), len(seq_b), self._insert_seq(cache, allocator, pool, seq_b)

    def _load_back(self, cache, pool, seq):
        match = match_prefix(cache, seq)
        req = self._make_req(pool)
        self._apply_match_to_req(req, match)
        loaded, last_node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        if loaded:
            self._finish_pending_loads(cache)
        self._release_ongoing_load_back_locks(cache)
        return match, loaded, last_node

    def _make_room(self, cache, allocator, need):
        full_short = need - allocator.full_available_size()
        swa_short = need - allocator.swa_available_size()
        if full_short > 0 or swa_short > 0:
            cache.evict_for_alloc(
                EvictParams(
                    num_tokens=max(full_short, 0), swa_num_tokens=max(swa_short, 0)
                )
            )
        return (
            allocator.full_available_size() >= need
            and allocator.swa_available_size() >= need
        )

    def _run_random_operations(self, write_policy, seed):
        # A small SWA pool, so allocations alone run it out and trigger eviction.
        cache, allocator, pool = self._build(
            write_policy, swa_kv_size=self.cfg.kv_size // 4
        )
        rng = random.Random(seed)
        seqs, locks = [], []
        next_token, restored = 1, 0
        for _ in range(300):
            op = rng.random()
            if op < 0.4 or not seqs:
                base = rng.choice(seqs) if seqs and rng.random() < 0.7 else []
                keep = rng.randrange(len(base) // self.page + 1) * self.page
                extra = rng.randint(1, 3 * self.window // self.page) * self.page
                seq = base[:keep] + list(range(next_token, next_token + extra))
                next_token += extra
                if self._make_room(cache, allocator, len(seq)):
                    self._insert(cache, allocator, pool, seq)
                    seqs.append(seq)
            elif op < 0.55:
                pages = rng.randint(1, 16) * self.page
                cache.evict(
                    rng.choice(
                        [
                            EvictParams(num_tokens=pages),
                            EvictParams(num_tokens=0, swa_num_tokens=pages),
                            EvictParams(num_tokens=0, mamba_num=1),
                        ]
                    )
                )
            elif op < 0.65:
                component = rng.choice(
                    [ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA]
                )
                cache.evict_host(
                    1 if component == ComponentType.MAMBA else self.page, component
                )
            elif op < 0.75:
                seq = rng.choice(seqs)
                match = match_prefix(cache, seq)
                if match.host_hit_length and self._make_room(
                    cache, allocator, len(seq)
                ):
                    restored += self._load_back(cache, pool, seq)[1] > 0
            elif op < 0.87:
                match = match_prefix(cache, rng.choice(seqs))
                if match.last_device_node != cache.root_node_handle():
                    locks.append(cache.lock(match.last_device_node))
            elif locks:
                lock = locks.pop(rng.randrange(len(locks)))
                if not lock.swa_released and rng.random() < 0.3:
                    cache.release_swa(lock)
                    locks.append(lock)
                else:
                    cache.unlock(lock)
            cache.writing_check(write_back=True)
            cache.sanity_check()
        self.assertGreater(restored, 0)
        for lock in locks:
            cache.unlock(lock)
        swa = ComponentType.SWA
        cache.evict_for_alloc(
            EvictParams(
                num_tokens=0,
                swa_num_tokens=cache.tree_core.component_evictable_size(swa),
            )
        )
        self.assertEqual(cache.tree_core.component_evictable_size(swa), 0)
        cache.sanity_check()
