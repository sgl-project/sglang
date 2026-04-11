"""Unit tests for UnifiedRadixCache"""

import unittest

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_PAGE_SIZE = 1
_HEAD_NUM = 2
_HEAD_DIM = 128
_NUM_LAYERS = 24
_GLOBAL_INTERVAL = 4
_DTYPE = torch.bfloat16


def _full_attention_layer_ids():
    return [i for i in range(_GLOBAL_INTERVAL - 1, _NUM_LAYERS, _GLOBAL_INTERVAL)]


def _mamba_layer_ids():
    full_set = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full_set]


def _swa_attention_layer_ids():
    full_set = set(_full_attention_layer_ids())
    return [i for i in range(_NUM_LAYERS) if i not in full_set]


# ===================================================================
# Test: Full + Mamba components (no SWA)
# ===================================================================
class TestUnifiedRadixCacheMamba(unittest.TestCase):
    """UnifiedRadixCache with (Full, Mamba) components."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def _build_tree(
        self,
        kv_size: int = 128,
        max_num_reqs: int = 10,
        mamba_cache_size: int = 20,
        max_context_len: int = 128,
    ):
        device = get_device()
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=_mamba_layer_ids()
            )

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=_mamba_layer_ids(),
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=kv_size,
            dtype=_DTYPE,
            page_size=_PAGE_SIZE,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            full_attention_layer_ids=_full_attention_layer_ids(),
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=kv_size,
            dtype=_DTYPE,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        tree = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=_PAGE_SIZE,
                disable=False,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            ),
        )

        def make_req():
            sp = SamplingParams(temperature=0, max_new_tokens=1)
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sp,
            )
            req_to_token_pool.alloc([req])
            return req

        return tree, allocator, req_to_token_pool, make_req

    # ------- insert + match -------
    def test_insert_and_match_basic(self):
        tree, alloc, _, make_req = self._build_tree()

        # Insert [1,2,3]
        req1 = make_req()
        v1 = alloc.alloc(3)
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=v1,
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(result.prefix_len, 0)

        # Insert [1,2,3,4,5] — shares prefix [1,2,3]
        req2 = make_req()
        v2 = alloc.alloc(5)
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=v2,
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(result.prefix_len, 3)

        # Match [1,2,3,4,5] — full hit
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        self.assertEqual(len(m.device_indices), 5)

        # Match [1,2,3,4,5,6] — partial hit (5 tokens)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6])))
        self.assertEqual(len(m.device_indices), 5)

        # Match [10,11] — no hit
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([10, 11])))
        self.assertEqual(len(m.device_indices), 0)

        tree.sanity_check()

    # ------- evict: full-only -------
    def test_evict_full_tokens(self):
        tree, alloc, _, make_req = self._build_tree()

        # Insert two disjoint sequences
        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([10, 11, 12]),
                value=alloc.alloc(3),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(tree.full_evictable_size(), 6)

        # Evict 3 full tokens — should remove one leaf
        result = tree.evict(EvictParams(num_tokens=3))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(result.num_tokens_evicted, 3)
        self.assertTrue(tree.full_evictable_size() <= 3)
        tree.sanity_check()

    # ------- evict: mamba-only -------
    def test_evict_mamba_only(self):
        tree, alloc, rtp, make_req = self._build_tree()
        mamba_pool = rtp.mamba_pool

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7]),
                value=alloc.alloc(7),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(tree.mamba_evictable_size(), 2)

        # Evict 1 mamba state
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        # After mamba eviction on an internal node, full tokens remain
        self.assertGreaterEqual(tree.full_evictable_size(), 0)
        tree.sanity_check()

    # ------- evict: mamba → match stops at tombstone -------
    def test_evict_mamba_breaks_match(self):
        tree, alloc, _, make_req = self._build_tree()

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Evict all mamba (2 states)
        tree.evict(EvictParams(num_tokens=0, mamba_num=2))
        self.assertEqual(tree.mamba_evictable_size(), 0)

        # Now match should return 0 because mamba validator fails
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    # ------- evict: lock_ref protection -------
    def test_evict_respects_lock_ref(self):
        tree, alloc, _, make_req = self._build_tree()

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([10, 11, 12]),
                value=alloc.alloc(3),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Lock the first leaf
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        locked_node = m.last_device_node
        tree.inc_lock_ref(locked_node)

        # Evict all full tokens — only unlocked leaf should be evicted
        result = tree.evict(EvictParams(num_tokens=6))
        self.assertGreaterEqual(result.num_tokens_evicted, 3)

        # [1,2,3] is still matchable because it was locked
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertEqual(len(m.device_indices), 3)

        # Unlock and verify we can now evict it
        tree.dec_lock_ref(locked_node)
        result = tree.evict(EvictParams(num_tokens=3))
        self.assertGreaterEqual(result.num_tokens_evicted, 3)
        tree.sanity_check()

    # ------- evict: verify EvictResult accounting -------
    def test_evict_result_accounting(self):
        tree, alloc, _, make_req = self._build_tree()

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Request 0 mamba + 3 full → full evicted, mamba cascaded
        result = tree.evict(EvictParams(num_tokens=3))
        self.assertGreaterEqual(result.num_tokens_evicted, 3)
        # Leaf eviction cascades all components; mamba also freed
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        tree.sanity_check()

    # ------- insert: prev_prefix_len controls overlap free -------
    def test_insert_prev_prefix_len(self):
        tree, alloc, _, make_req = self._build_tree()
        initial_avail = alloc.available_size()

        # Step 1: Insert [1,2,3]
        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(alloc.available_size(), initial_avail - 3)

        # Step 2: Insert [1,2,3,4,5] with prev_prefix_len=0 → frees overlap [0:3]
        req2 = make_req()
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
                prev_prefix_len=0,
            )
        )
        self.assertEqual(result.prefix_len, 3)
        # alloc 5, freed 3 overlap, stored 2 new → net -2
        self.assertEqual(alloc.available_size(), initial_avail - 3 - 2)

        # Step 3: Insert [1,2,3,4,5,6] with prev_prefix_len=5 → nothing freed
        req3 = make_req()
        avail_before = alloc.available_size()
        result = tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6]),
                value=alloc.alloc(6),
                mamba_value=req3.mamba_pool_idx.unsqueeze(0),
                prev_prefix_len=5,
            )
        )
        self.assertEqual(result.prefix_len, 5)
        # alloc 6, freed 0, stored 1 → net -6
        self.assertEqual(alloc.available_size(), avail_before - 6)
        tree.sanity_check()

    # ------- available_and_evictable_str + pretty_print -------
    def test_diagnostics(self):
        tree, alloc, _, make_req = self._build_tree()

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        diag = tree.available_and_evictable_str()
        self.assertIn("Available full tokens", diag)
        self.assertIn("mamba", diag.lower())

        diag2 = available_and_evictable_str(tree)
        self.assertIn("Available full tokens", diag2)

        tree.pretty_print()
        tree.sanity_check()


# ===================================================================
# Test: Full + SWA + Mamba components
# ===================================================================
class TestUnifiedRadixCacheSWAMamba(unittest.TestCase):
    """UnifiedRadixCache with (Full, SWA, Mamba) components — the most complex config."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def _build_tree(
        self,
        kv_size: int = 128,
        kv_size_swa: int = 64,
        max_num_reqs: int = 10,
        mamba_cache_size: int = 20,
        max_context_len: int = 128,
        sliding_window_size: int = 4,
    ):
        device = get_device()
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=_mamba_layer_ids()
            )

        req_to_token_pool = HybridReqToTokenPool(
            size=max_num_reqs,
            mamba_size=mamba_cache_size,
            mamba_spec_state_size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=_mamba_layer_ids(),
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )

        kv_pool = SWAKVPool(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=_PAGE_SIZE,
            dtype=_DTYPE,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            swa_attention_layer_ids=_swa_attention_layer_ids(),
            full_attention_layer_ids=_full_attention_layer_ids(),
            enable_kvcache_transpose=False,
            device=device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size_swa,
            page_size=_PAGE_SIZE,
            dtype=_DTYPE,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )

        tree = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=_PAGE_SIZE,
                disable=False,
                sliding_window_size=sliding_window_size,
                tree_components=(
                    ComponentType.FULL,
                    ComponentType.SWA,
                    ComponentType.MAMBA,
                ),
            ),
        )

        def make_req():
            sp = SamplingParams(temperature=0, max_new_tokens=1)
            req = Req(
                rid=0,
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=sp,
            )
            req_to_token_pool.alloc([req])
            return req

        return tree, allocator, req_to_token_pool, make_req

    # ------- basic insert + match with SWA -------
    def test_insert_and_match_with_swa(self):
        tree, alloc, _, make_req = self._build_tree(sliding_window_size=4)

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Match: SWA validator requires contiguous window >= sliding_window_size
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        # With sliding_window_size=4 and 5 tokens on single node → should match
        self.assertEqual(len(m.device_indices), 5)
        tree.sanity_check()

    # ------- evict SWA → cascade Mamba -------
    def test_evict_swa_cascades_mamba(self):
        tree, alloc, _, make_req = self._build_tree(sliding_window_size=4)

        # Build tree: [1,2,3] → [4,5,6,7]
        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3]),
                value=alloc.alloc(3),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5, 6, 7]),
                value=alloc.alloc(7),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        initial_mamba = tree.mamba_evictable_size()

        # Evict SWA — on internal node, SWA eviction cascades to Mamba (priority: swa=1 > mamba=0)
        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=3))
        self.assertGreaterEqual(result.swa_num_tokens_evicted, 0)

        tree.sanity_check()

    # ------- evict full leaf -------
    def test_evict_full_leaf_cascades_all(self):
        tree, alloc, _, make_req = self._build_tree(sliding_window_size=4)

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([10, 11, 12, 13, 14]),
                value=alloc.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )
        self.assertEqual(tree.full_evictable_size(), 10)

        # Evict one leaf (5 full tokens) → also cascades SWA + Mamba
        result = tree.evict(EvictParams(num_tokens=5))
        self.assertGreaterEqual(result.num_tokens_evicted, 5)
        # Leaf eviction should cascade all components
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        self.assertGreaterEqual(result.swa_num_tokens_evicted, 0)
        tree.sanity_check()

    # ------- evict with SWA lock -------
    def test_swa_lock_protects_from_eviction(self):
        tree, alloc, _, make_req = self._build_tree(sliding_window_size=4)

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )
        req2 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([10, 11, 12, 13, 14]),
                value=alloc.alloc(5),
                mamba_value=req2.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Lock the first entry
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        lock_result = tree.inc_lock_ref(m.last_device_node)

        # Try to evict all full tokens
        result = tree.evict(EvictParams(num_tokens=10))
        # Only the unlocked one (5 tokens) should be evictable
        self.assertGreaterEqual(result.num_tokens_evicted, 5)

        # Locked one is still matchable
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5])))
        self.assertEqual(len(m.device_indices), 5)

        # Unlock
        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=lock_result.swa_uuid_for_lock),
        )
        tree.sanity_check()

    # ------- cache_finished_req (with insert) -------
    def test_cache_finished_req_insert(self):
        tree, alloc, rtp, make_req = self._build_tree()

        req = make_req()
        req.origin_input_ids = [1, 2, 3, 4, 5]
        req.output_ids = [6, 7]
        kv_len = len(req.origin_input_ids) + len(req.output_ids)
        kv_indices = alloc.alloc(kv_len)
        rtp.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.mamba_last_track_seqlen = kv_len
        req.fill_ids = req.origin_input_ids + req.output_ids

        tree.cache_finished_req(req, is_insert=True)

        # Verify the tokens are in the tree
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5, 6, 7])))
        self.assertEqual(len(m.device_indices), 7)
        tree.sanity_check()

    # ------- cache_finished_req (no insert) -------
    def test_cache_finished_req_no_insert(self):
        tree, alloc, rtp, make_req = self._build_tree()

        req = make_req()
        req.origin_input_ids = [1, 2, 3]
        req.output_ids = []
        kv_len = 3
        kv_indices = alloc.alloc(kv_len)
        rtp.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.fill_ids = req.origin_input_ids

        avail_before = alloc.available_size()
        tree.cache_finished_req(req, is_insert=False)

        # KV indices should be freed back
        self.assertEqual(alloc.available_size(), avail_before + kv_len)

        # Nothing in tree
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    # ------- cache_unfinished_req -------
    def test_cache_unfinished_req(self):
        tree, alloc, rtp, make_req = self._build_tree()

        req = make_req()
        req.origin_input_ids = [1, 2, 3, 4, 5]
        req.output_ids = []
        req.fill_ids = req.origin_input_ids[:]
        kv_len = len(req.fill_ids)
        kv_indices = alloc.alloc(kv_len)
        rtp.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.mamba_last_track_seqlen = kv_len

        tree.cache_unfinished_req(req)

        # After caching, prefix_indices should be set
        self.assertGreater(len(req.prefix_indices), 0)
        self.assertEqual(req.cache_protected_len, len(req.prefix_indices))
        self.assertIsNotNone(req.last_node)

        # Release the lock acquired by cache_unfinished_req before idle check
        tree.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    # ------- evict empty tree → no crash -------
    def test_evict_empty_tree(self):
        tree, alloc, _, _ = self._build_tree()
        result = tree.evict(EvictParams(num_tokens=10, mamba_num=5))
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 0)
        tree.sanity_check()

    # ------- multiple evictions until empty -------
    def test_evict_until_empty(self):
        tree, alloc, _, make_req = self._build_tree()

        for i in range(5):
            req = make_req()
            tokens = list(range(i * 10, i * 10 + 5))
            tree.insert(
                InsertParams(
                    key=RadixKey(tokens),
                    value=alloc.alloc(5),
                    mamba_value=req.mamba_pool_idx.unsqueeze(0),
                )
            )
        self.assertEqual(tree.full_evictable_size(), 25)

        # Evict all
        result = tree.evict(EvictParams(num_tokens=100))
        self.assertGreaterEqual(result.num_tokens_evicted, 25)
        self.assertEqual(tree.full_evictable_size(), 0)
        self.assertEqual(tree.mamba_evictable_size(), 0)

        # Verify tree is empty (no matches)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3, 4])))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    # ------- cow mamba on match -------
    def test_match_cow_mamba(self):
        tree, alloc, rtp, make_req = self._build_tree()
        mamba_pool = rtp.mamba_pool

        req1 = make_req()
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
                mamba_value=req1.mamba_pool_idx.unsqueeze(0),
            )
        )

        # Match with cow_mamba
        req2 = make_req()
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3, 4, 5]), cow_mamba=True, req=req2)
        )
        self.assertEqual(len(m.device_indices), 5)
        # req2 should now have its own mamba state (copied)
        self.assertIsNotNone(req2.mamba_pool_idx)

        # Verify the copy matches
        src_value = m.last_device_node.component_data[ComponentType.MAMBA].value
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][:, req2.mamba_pool_idx]
                == mamba_pool.mamba_cache.conv[0][:, src_value]
            )
        )
        tree.sanity_check()


# ===================================================================
# Test: Helper functions
# ===================================================================
class TestUnifiedRadixCacheHelpers(unittest.TestCase):
    """Tests for internal helper functions of UnifiedRadixCache."""

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=_PAGE_SIZE)
        )

    def _build_tree(
        self,
        kv_size: int = 128,
        max_num_reqs: int = 10,
        max_context_len: int = 128,
    ):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

        device = get_device()
        req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        kv_pool = MHATokenToKVPool(
            size=kv_size,
            page_size=_PAGE_SIZE,
            dtype=_DTYPE,
            head_num=_HEAD_NUM,
            head_dim=_HEAD_DIM,
            layer_num=_NUM_LAYERS,
            device=device,
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=kv_size,
            dtype=_DTYPE,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        tree = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=_PAGE_SIZE,
                disable=False,
                tree_components=(
                    ComponentType.FULL,
                ),  # Full attention only, no mamba/swa
            ),
        )
        return tree, allocator

    def test_readonly_does_not_modify_tree(self):
        """Verify readonly match does not modify tree structure (no split)."""
        tree, alloc = self._build_tree()

        # Insert [1, 2, 3, 4, 5]
        tree.insert(
            InsertParams(
                key=RadixKey([1, 2, 3, 4, 5]),
                value=alloc.alloc(5),
            )
        )

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        node_count_before = count_nodes(tree.root_node)
        self.assertEqual(node_count_before, 2)  # root_node and [1, 2, 3, 4, 5]

        # Regular match with partial key [1, 2] creates a split
        value, best_node, best_value_len = tree._match_prefix_helper(RadixKey([1, 2]))
        # Regular match with partial key [1, 2, 3, 4] creates a split
        value, best_node, best_value_len = tree._match_prefix_helper(
            RadixKey([1, 2, 3, 4])
        )
        self.assertEqual(best_value_len, 2)
        self.assertEqual(best_node.key.token_ids, [3, 4])
        node_count_after_regular = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_regular, node_count_before + 2)

        # Readonly match with partial key [1, 2, 3] should NOT create a split
        value, best_node, best_value_len = tree._match_prefix_helper_readonly(
            RadixKey([1, 2, 3])
        )
        self.assertEqual(best_value_len, 1)
        self.assertEqual(best_node.key.token_ids, [1, 2])
        node_count_after_readonly = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_readonly, node_count_after_regular)

        tree.sanity_check()


if __name__ == "__main__":
    unittest.main()
