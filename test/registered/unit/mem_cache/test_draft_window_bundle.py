"""DFLASH draft KV kept as the windows of Mamba checkpoints (draft_window.py)."""

import random
import unittest
from array import array
from dataclasses import replace
from unittest import mock

import torch
from test_unified_radix_cache_unittest import (
    _TREE_CORE_REGISTRY,
    CacheConfig,
    UnifiedRadixCacheSuite,
    _device_value,
    _host_value,
    build_fixture,
)
from unified_tree_core_inspector import UnifiedTreeCoreInspector

from sglang.srt.mem_cache.allocator.swa import DraftSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.components.draft_window import (
    DraftWindowComponent,
    DraftWindowMambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

FULL, SWA, MAMBA = ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA
DRAFT_LAYERS = 2


def build_draft_bundle_fixture(cfg: CacheConfig, draft_size: int):
    """A Full + Mamba target whose allocator holds the draft pool as its SWA side."""
    target_cfg = replace(cfg, components=(FULL, MAMBA), sliding_window_size=None)
    _, target_allocator, req_to_token_pool = build_fixture(target_cfg)
    device = get_device()
    draft_pool = SWAKVPool(
        size=cfg.kv_size,
        size_swa=draft_size,
        page_size=cfg.page_size,
        dtype=cfg.dtype,
        head_num=cfg.head_num,
        head_dim=cfg.head_dim,
        swa_attention_layer_ids=list(range(DRAFT_LAYERS)),
        full_attention_layer_ids=[],
        device=device,
    )
    allocator = DraftSWATokenToKVPoolAllocator(
        cfg.kv_size,
        draft_size,
        page_size=cfg.page_size,
        dtype=cfg.dtype,
        device=device,
        kvcache=target_allocator.get_kvcache(),
        need_sort=False,
    )
    allocator.attach_draft_kv_pool(draft_pool)
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=cfg.page_size,
        disable=False,
        sliding_window_size=cfg.sliding_window_size,
        tree_components=(FULL, SWA, MAMBA),
        enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
        eviction_policy=cfg.eviction_policy,
        component_registry_override={
            SWA: DraftWindowComponent,
            MAMBA: DraftWindowMambaComponent,
        },
    )
    with mock.patch.dict(
        _TREE_CORE_REGISTRY,
        {
            "python": lambda params, components: UnifiedTreeCoreInspector(
                params, components
            )
        },
    ):
        cache = UnifiedRadixCache(params=params)
    cache.cache_init_params = params
    return cache, allocator, req_to_token_pool


class _DraftWindowBundleSuite:
    cfg: CacheConfig
    _rid = 0
    _alloc = UnifiedRadixCacheSuite._alloc
    _insert = UnifiedRadixCacheSuite._insert
    _make_req = UnifiedRadixCacheSuite._make_req
    _make_seq = UnifiedRadixCacheSuite._make_seq
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _apply_match_to_req = UnifiedRadixCacheSuite._apply_match_to_req
    _finish_pending_loads = UnifiedRadixCacheSuite._finish_pending_loads
    _release_ongoing_load_back_locks = (
        UnifiedRadixCacheSuite._release_ongoing_load_back_locks
    )

    def _build(self, write_policy=None, draft_size=None):
        cache, allocator, pool = build_draft_bundle_fixture(
            self.cfg, draft_size=draft_size or self.cfg.kv_size
        )
        if write_policy is not None:
            self._init_hicache(cache, write_policy=write_policy)
            if write_policy == "write_through":
                cache.write_through_threshold = 1
        window = cache.tree_core.components_by_type[SWA]
        self.page = self.cfg.page_size
        self.window = window.window_tokens
        return cache, allocator, pool, window

    def _seq(self, start, num_tokens):
        return self._make_seq(start, num_tokens // self.page)

    def _insert_seq(self, cache, allocator, pool, seq):
        return self._insert(cache, allocator, pool, seq).last_device_node

    def _path(self, cache, node_id, end):
        """(start, end, cover_ref, has draft KV, has checkpoint) from the root."""
        tree = cache.tree_core
        node, spans = tree.node_by_id(node_id), []
        while node is not tree.root_node:
            start = end - len(node.key)
            spans.append(
                (
                    start,
                    end,
                    DraftWindowComponent.cover_ref(node),
                    node.component_data[SWA].value is not None,
                    node.component_data[MAMBA].value is not None,
                )
            )
            node, end = node.parent, start
        return spans[::-1]

    def _two_overlapping_bundles(self, cache, allocator, pool):
        seq_a = self._seq(1, 2 * self.window + self.page)
        seq_b = seq_a + self._seq(1000, self.page)
        self._insert_seq(cache, allocator, pool, seq_a)
        return len(seq_a), len(seq_b), self._insert_seq(cache, allocator, pool, seq_b)

    def test_overlapping_windows_share_draft_kv(self):
        cache, allocator, pool, window = self._build()
        a, b, leaf = self._two_overlapping_bundles(cache, allocator, pool)
        w = self.window
        self.assertEqual(
            self._path(cache, leaf, b),
            [
                (0, a - w, 0, True, False),
                (a - w, b - w, 1, True, False),
                (b - w, a, 2, True, True),
                (a, b, 1, True, True),
            ],
        )
        # Draft KV outside every window is reclaimable.
        first = cache.tree_core.root_node.children
        self.assertTrue(
            any(n.id in window._uncovered[EvictLayer.DEVICE] for n in first.values())
        )
        cache.sanity_check()

    def test_evicting_a_checkpoint_frees_only_its_unshared_window(self):
        cache, allocator, pool, _ = self._build()
        a, b, leaf = self._two_overlapping_bundles(cache, allocator, pool)

        # The older, internal checkpoint is the Mamba LRU victim.
        result = cache.evict(EvictParams(num_tokens=0, mamba_num=1))

        w = self.window
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            self._path(cache, leaf, b)[1:],
            [
                (a - w, b - w, 0, False, False),
                (b - w, a, 1, True, False),
                (a, b, 1, True, True),
            ],
        )
        cache.sanity_check()

    def test_draft_pressure_reclaims_uncovered_kv_then_evicts_bundles(self):
        cache, allocator, pool, window = self._build()
        leaves = [
            self._insert_seq(
                cache, allocator, pool, self._seq(1000 * i, 2 * self.window + self.page)
            )
            for i in range(1, 4)
        ]
        uncovered = 3 * (self.window + self.page)
        available = allocator.swa_available_size()

        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=uncovered))
        self.assertEqual(allocator.swa_available_size(), available + uncovered)
        self.assertFalse(window._uncovered[EvictLayer.DEVICE])
        for leaf in leaves:
            self.assertIsNotNone(_device_value(cache, leaf, MAMBA))

        # Beyond the uncovered KV, the least recent bundle goes as a whole.
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=self.page))
        self.assertFalse(cache.tree_core.contains_node(leaves[0]))
        for leaf in leaves[1:]:
            self.assertIsNotNone(_device_value(cache, leaf, MAMBA))
            self.assertIsNotNone(_device_value(cache, leaf, SWA))
        cache.sanity_check()

    def _load_back(self, cache, pool, seq):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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

    def test_write_back_restores_a_whole_bundle_from_host(self):
        cache, allocator, pool, _ = self._build("write_back")
        seq = self._seq(1, 2 * self.window + self.page)
        leaf = self._insert_seq(cache, allocator, pool, seq)
        draft = allocator.draft_kv_pool.swa_kv_pool
        slots = _device_value(cache, leaf, SWA)
        written = []
        for layer in range(DRAFT_LAYERS):
            k_buffer = draft.get_key_buffer(layer)
            written.append(torch.randn_like(k_buffer[slots], dtype=torch.float32))
            k_buffer[slots] = written[-1].to(k_buffer.dtype)

        cache.evict(EvictParams(num_tokens=len(seq)))
        node = cache.tree_core.node_by_id(leaf)
        for component in (FULL, MAMBA, SWA):
            self.assertIsNone(_device_value(cache, leaf, component))
            self.assertIsNotNone(_host_value(cache, leaf, component))
        # Only the window gets a host copy.
        self.assertIsNone(node.parent.component_data[SWA].host_value)
        for layer in range(DRAFT_LAYERS):
            draft.get_key_buffer(layer).zero_()

        match, loaded, last_node = self._load_back(cache, pool, seq)
        self.assertEqual(last_node, leaf)
        self.assertEqual(match.device_prefix_len + loaded, len(seq))
        slots = _device_value(cache, leaf, SWA)
        for layer, values in enumerate(written):
            k_buffer = draft.get_key_buffer(layer)
            self.assertTrue(torch.equal(k_buffer[slots], values.to(k_buffer.dtype)))
        cache.sanity_check()

    def test_random_operations_keep_bundle_invariants(self):
        """sanity_check recomputes every window count after each step."""
        for write_policy in ("write_back", "write_through"):
            for seed in range(3):
                with self.subTest(write_policy=write_policy, seed=seed):
                    self._run_random_operations(write_policy, seed)

    def _run_random_operations(self, write_policy, seed):
        # A small draft pool, so allocations themselves run it short.
        cache, allocator, pool, window = self._build(
            write_policy, draft_size=self.cfg.kv_size // 4
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
                component = rng.choice([FULL, SWA, MAMBA])
                cache.evict_host(1 if component == MAMBA else self.page, component)
            elif op < 0.75:
                seq = rng.choice(seqs)
                match = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq)))
                )
                if match.host_hit_length and self._make_room(
                    cache, allocator, len(seq)
                ):
                    restored += self._load_back(cache, pool, seq)[1] > 0
            elif op < 0.87:
                match = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", rng.choice(seqs))))
                )
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
        # With every lock gone, draft pressure can empty the pool.
        cache.evict(
            EvictParams(
                num_tokens=0,
                swa_num_tokens=cache.tree_core.component_evictable_size(SWA),
            )
        )
        self.assertEqual(cache.tree_core.component_evictable_size(SWA), 0)
        self.assertFalse(window._uncovered[EvictLayer.DEVICE])
        cache.sanity_check()

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


_CFG = dict(
    components=(FULL, SWA, MAMBA), kv_size=1024, max_num_reqs=400, mamba_cache_size=400
)


class TestDraftWindowBundlePage1(_DraftWindowBundleSuite, CustomTestCase):
    cfg = CacheConfig(page_size=1, sliding_window_size=4, **_CFG)


class TestDraftWindowBundlePage4(_DraftWindowBundleSuite, CustomTestCase):
    # A 6-token window rounds up to two pages.
    cfg = CacheConfig(
        page_size=4, sliding_window_size=6, enable_mamba_extra_buffer=True, **_CFG
    )


if __name__ == "__main__":
    unittest.main()
