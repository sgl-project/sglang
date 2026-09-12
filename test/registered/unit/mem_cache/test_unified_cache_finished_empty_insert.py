"""Regression tests: cache_finished_req must early-return when nothing is cacheable.

A finished hybrid (mamba) request that never reached a track boundary reports
cache_len None -> 0 from MambaComponent.prepare_for_caching_req, making
effective_cache_len <= 0. The finished path must skip the insert, mirroring
cache_unfinished_req's existing early-return: an empty-key insert is a tree
no-op whose guard result would repoint req.last_node onto the (unlocked) root
and register a zero-length session ref. The path must also conserve every
pool slot (KV row, mamba slot, donated mamba value).

Unified-path port of the legacy MambaRadixCache fix in #35821.
"""

import unittest
from array import array

import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestCacheFinishedReqEmptyInsert(CustomTestCase):
    # enable_mamba_extra_buffer=True makes MambaComponent derive cache_len from
    # req.kv.mamba_last_track_seqlen, so None -> 0 deterministically triggers
    # the "no track boundary" case without depending on ReplaySSM write_pos.
    cfg = CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=8,
        kv_size=64,
        max_context_len=64,
    )

    def _make_finished_req(self, req_to_token_pool, tokens, last_track_seqlen):
        req = Req(
            rid=f"empty-insert-{last_track_seqlen}-{len(tokens)}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
        req.kv.kv_committed_len = len(tokens)
        req.kv.kv_allocated_len = len(tokens)
        req.kv.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.swa_prefix_lock_released = False
        req.extra_key = None
        # None == never hit a track boundary (the trigger under test).
        req.kv.mamba_last_track_seqlen = last_track_seqlen
        return req

    def _finish(
        self,
        cache,
        allocator,
        req_to_token_pool,
        tokens,
        track_seqlen,
        last_node=None,
        lock_result=None,
    ):
        req = self._make_finished_req(req_to_token_pool, tokens, track_seqlen)
        if tokens:
            kv_indices = allocator.alloc(len(tokens))
            self.assertIsNotNone(kv_indices)
            req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(0, len(tokens))), kv_indices
            )
        if last_node is None:
            # A request that matched no prefix holds the root as its last_node.
            req.last_node = cache.root_node_handle()
        else:
            # A request that matched a prefix holds a lock on the matched node.
            req.last_node = last_node
            req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
            req.skip_lock_node_ids = lock_result.skip_lock_node_ids
        cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.effective_kv_committed_len()
        )
        return req

    def test_finished_req_without_track_boundary_inserts_nothing(self):
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        tokens = [1, 2, 3]

        # Seed an unrelated prefix so the request can hold a NON-root
        # last_node: the no-op empty-key insert must not repoint it to the
        # (never-locked) root, which is what the tree core's empty-key guard
        # result (last_device_node=root) would do without the early-return.
        # (Repointing only bites when last_node != root before the call, so a
        # plain "last_node stays root" assertion cannot distinguish the paths.)
        seed = [7, 7, 7]
        donor = Req(
            rid="seed-donor",
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([donor])
        seed_value = allocator.alloc(len(seed))
        self.assertIsNotNone(seed_value)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", seed)),
                value=seed_value,
                mamba_value=donor.kv.mamba_pool_idx.unsqueeze(0),
            )
        )
        matched = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seed))))
        self.assertNotEqual(matched.last_device_node, cache.root_node_handle())
        lock_result = cache.inc_lock_ref(matched.last_device_node)

        # Conservation baseline: after the seed insert, before the request
        # allocates anything.
        kv1 = allocator.available_size()
        m1 = req_to_token_pool.mamba_allocator.available_size()

        req = self._finish(
            cache,
            allocator,
            req_to_token_pool,
            tokens,
            None,
            last_node=matched.last_device_node,
            lock_result=lock_result,
        )

        # No ghost node: the request's own sequence is not retrievable, and
        # the tree holds only the seed's mamba state.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(m.device_indices), 0)
        self.assertEqual(cache.mamba_evictable_size(), 1)
        # last_node keeps pointing at the request's matched node (whose lock
        # the early-return released) instead of being repointed to root.
        self.assertEqual(req.last_node, matched.last_device_node)
        # Slot conservation: the request's KV row and mamba slot fully
        # returned; the seed's tree-owned resources untouched.
        self.assertEqual(allocator.available_size(), kv1)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m1)
        cache.sanity_check()

    def test_finished_req_with_track_boundary_still_inserts(self):
        """Contrast case: the fixture inserts normally when a boundary exists,
        so the empty case above is not trivially passing."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        kv0 = allocator.available_size()
        tokens = [1, 2, 3]

        self._finish(cache, allocator, req_to_token_pool, tokens, len(tokens))

        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(m.device_indices), len(tokens))
        self.assertEqual(cache.mamba_evictable_size(), 1)
        self.assertEqual(allocator.available_size(), kv0 - len(tokens))
        cache.sanity_check()

    def test_finished_zero_length_request(self):
        """seqlen=0: the early-return's free range is empty and must be a no-op."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        kv0 = allocator.available_size()
        m0 = req_to_token_pool.mamba_allocator.available_size()

        self._finish(cache, allocator, req_to_token_pool, [], None)

        self.assertEqual(allocator.available_size(), kv0)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m0)
        cache.sanity_check()

    def test_insert_empty_key_is_a_pure_noop(self):
        """UnifiedRadixCache.insert never opens a walk for an empty key and
        reports mamba_exist=True so callers free the unconsumed donation."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        m0 = req_to_token_pool.mamba_allocator.available_size()

        req = self._make_finished_req(req_to_token_pool, [9, 9, 9], None)
        result = cache.insert(
            InsertParams(
                key=RadixKey(array("q", [])),
                value=torch.tensor([], dtype=torch.int64),
                mamba_value=req.kv.mamba_pool_idx.unsqueeze(0),
            )
        )

        self.assertEqual(result.prefix_len, 0)
        self.assertTrue(result.mamba_exist)
        # Documents the no-repoint semantics; relax this assertion only if the
        # insert()-level guard is dropped in favor of the tree core's guard.
        self.assertIsNone(result.last_device_node)
        self.assertEqual(cache.mamba_evictable_size(), 0)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [9, 9, 9]))))
        self.assertEqual(len(m.device_indices), 0)
        # The unconsumed donation is the caller's to release.
        req_to_token_pool.free_mamba_cache(req)
        self.assertEqual(req_to_token_pool.mamba_allocator.available_size(), m0)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
