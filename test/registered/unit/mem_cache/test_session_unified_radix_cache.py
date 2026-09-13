"""Tests for session references on UnifiedRadixCache."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase


def make_params(enable_session: bool) -> CacheInitParams:
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=64,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        eviction_policy="lru",
        enable_session_radix_cache=enable_session,
        tree_components=(ComponentType.FULL,),
    )


def insert(cache, token_ids):
    """Insert and return the tail node; the cache boundary hands back a NodeId."""
    indices = cache.token_to_kv_pool_allocator.alloc(len(token_ids))
    node_id = cache.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=indices.to(torch.int64),
        )
    ).last_device_node
    return cache.tree_core.node_by_id(node_id)


def match_len(cache, token_ids) -> int:
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", token_ids)))
        ).device_indices
    )


def register(cache, token_ids, session_id, generation=None):
    if generation is None:
        generation = cache.ensure_session_generation(session_id)
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=generation,
            session=None,
            last_node=cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", token_ids)))
            ).last_device_node,
            origin_input_ids=array("q", token_ids),
            output_ids=array("q"),
            extra_key=None,
        )
    )


class TestRadixCacheSessionRemoval(CustomTestCase):
    def test_plain_radix_cache_does_not_enable_session_references(self):
        cache = RadixCache(make_params(enable_session=True))

        self.assertFalse(hasattr(cache, "enable_session_radix_cache"))
        self.assertFalse(hasattr(cache, "register_session_ref"))
        self.assertFalse(hasattr(cache, "open_radix_session"))


class TestSessionUnifiedRadixCache(CustomTestCase):
    def setUp(self):
        self.cache = UnifiedRadixCache(make_params(enable_session=True))
        self.full = self.cache.components[ComponentType.FULL]

    def test_register_and_release_update_full_component_reference(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        generation = self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", generation)
        self.assertEqual(self.full.session_ref(leaf), 1)

        self.cache.release_radix_session("s1")
        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_reopen_rejects_stale_generation(self):
        leaf = insert(self.cache, [1, 2, 3, 4])
        old_generation = self.cache.open_radix_session("s1")
        self.cache.release_radix_session("s1")
        self.cache.open_radix_session("s1")

        register(self.cache, [1, 2, 3, 4], "s1", old_generation)

        self.assertEqual(self.full.session_ref(leaf), 0)

    def test_eviction_prefers_unreferenced_full_kv(self):
        referenced = insert(self.cache, [1, 2, 3, 4])
        insert(self.cache, [7, 8, 9])
        register(self.cache, [1, 2, 3, 4], "s1")

        self.cache.evict(EvictParams(num_tokens=3))

        self.assertEqual(match_len(self.cache, [7, 8, 9]), 0)
        self.assertEqual(match_len(self.cache, [1, 2, 3, 4]), 4)
        self.assertEqual(self.full.session_ref(referenced), 1)


class TestSessionTerminalLeaf(CustomTestCase):
    def test_empty_terminal_insert_preserves_prior_hybrid_session_leaf(self):
        import test_unified_radix_cache_unittest as native

        from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH

        cases = (
            ("missing_checkpoint", True, None, True, False, False),
            ("zero_checkpoint", True, 0, True, False, False),
            ("new_checkpoint", True, 256, True, False, False),
            ("no_prefix", False, None, True, False, False),
            ("aborted", True, None, True, True, False),
            ("no_insert", True, None, False, False, False),
            ("stale_generation", True, None, True, False, True),
        )
        for name, prior, checkpoint, insert_tail, aborted, stale in cases:
            with (
                self.subTest(case=name),
                patch.object(native, "_TREE_CORE_TEST_BACKEND", "python"),
            ):
                cfg = native.CacheConfig(
                    components=(ComponentType.FULL, ComponentType.MAMBA),
                    page_size=64,
                    kv_size=2048,
                    max_context_len=1024,
                    enable_mamba_extra_buffer=True,
                    mamba_cache_size=60,
                )
                cache, allocator, pool = native.build_fixture(
                    cfg, enable_session_radix_cache=True
                )
                helper = native.UnifiedRadixCacheSuite()
                helper.cfg = cfg
                tokens = array("q", range(256))
                if prior:
                    helper._insert(cache, allocator, pool, tokens)
                req = helper._make_req(pool)
                match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
                helper._apply_match_to_req(req, match)
                cache.inc_lock_ref(req.last_node)
                req.kv.cache_protected_len = len(match.device_indices)
                req.origin_input_ids = tokens
                req.output_ids = array("q", range(2000, 2008))
                length = len(tokens) + len(req.output_ids)
                suffix = helper._alloc(allocator, length - len(match.device_indices))
                indices = torch.cat([match.device_indices, suffix])
                pool.write((req.kv.req_pool_idx, slice(0, length)), indices)
                req.kv.kv_committed_len = req.kv.kv_allocated_len = length
                req.full_untruncated_fill_ids = tokens + req.output_ids
                req.set_extend_range(len(match.device_indices), length)
                req.kv.mamba_last_track_seqlen = checkpoint
                req.session_id = "terminal-leaf"
                req.session_generation = cache.open_radix_session(req.session_id)
                if stale:
                    cache.release_radix_session(req.session_id)
                    cache.open_radix_session(req.session_id)
                req.finished_reason = (
                    FINISH_ABORT() if aborted else FINISH_LENGTH(length=8)
                )
                cache.cache_finished_req(
                    req, is_insert=insert_tail, kv_len_to_handle=length
                )
                expected = int(prior and insert_tail and not aborted and not stale)
                for component in cache.components.values():
                    refs = component._session_leaves.get(req.session_id, ())
                    self.assertEqual(
                        sum(component.session_ref(node) > 0 for node in refs), expected
                    )
                self.assertEqual(match_len(cache, tokens), 256 if prior else 0)
                cache.sanity_check()
                cache.release_radix_session(req.session_id)
                cache.evict(EvictParams(num_tokens=2048, mamba_num=60))
                self.assertEqual(match_len(cache, tokens), 0)
                self.assertEqual(allocator.available_size(), 2048)


if __name__ == "__main__":
    unittest.main()
