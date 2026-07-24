"""Regression test: committed prefix ending at a chunk boundary must be backed up.

Chunk-boundary nodes are inserted via cache_unfinished_req(chunked=True), which
skips the write-through trigger. With mamba extra buffer, when the final extend
chunk is shorter than mamba_cache_chunk_size, the finish-time commit collapses
to length 0 (mamba_last_track_seqlen stays None), so no non-chunked insert ever
touches the already-inserted prefix — it silently never reaches host/storage
backup and cross-instance KV reuse always misses for such prompts.

cache_finished_req must leave the request's committed prefix backed up.
"""

import unittest
from array import array

import test_unified_radix_cache_unittest as base

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

_CFG = base.CacheConfig(
    page_size=4,
    components=(ComponentType.FULL, ComponentType.MAMBA),
    enable_mamba_extra_buffer=True,
    mamba_cache_size=60,
    kv_size=2048,
    max_context_len=2048,
)


class TestHiCacheChunkBoundaryBackup(CustomTestCase):
    cfg = _CFG
    _rid = 0

    _init_hicache = base.TestUnifiedRadixCacheKVEvents._init_hicache

    def _make_req(self, req_to_token_pool):
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        type(self)._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _run_chunked_req_with_empty_finish(self, cache, allocator, req_to_token_pool):
        """One chunked request whose finish-time commit is empty.

        Chunk 1 commits [0, kv_len) via cache_unfinished_req(chunked=True); the
        final extend is too short to be tracked (mamba_last_track_seqlen stays
        None), mirroring a prompt whose page-aligned length is an exact
        multiple of chunked_prefill_size.
        """
        ps = self.cfg.page_size
        req = self._make_req(req_to_token_pool)
        tokens = list(range(1, 1 + 3 * ps))
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        kv_len = len(tokens)
        kv_indices = allocator.alloc(kv_len)
        self.assertIsNotNone(kv_indices)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = cache.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.mamba_last_track_seqlen = kv_len

        cache.cache_unfinished_req(req, chunked=True)
        self.assertIsNone(req.mamba_last_track_seqlen)

        boundary_node = req.last_node
        self.assertIsNot(boundary_node, cache.root_node)
        self.assertFalse(boundary_node.backuped)

        output_ids = [2000]
        req.output_ids = array("q", output_ids)
        req.full_untruncated_fill_ids = array("q", tokens + output_ids)
        req.set_extend_range(kv_len, kv_len + 1)
        extra = allocator.alloc(1)
        self.assertIsNotNone(extra)
        req_to_token_pool.write((req.req_pool_idx, slice(kv_len, kv_len + 1)), extra)
        req.kv_committed_len = kv_len + 1

        cache.cache_finished_req(req, is_insert=True)
        return tokens, boundary_node

    def test_chunk_boundary_prefix_backed_up_after_finish(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        tokens, boundary_node = self._run_chunked_req_with_empty_finish(
            cache, allocator, req_to_token_pool
        )

        self.assertTrue(boundary_node.backuped)
        cache.sanity_check()

    def test_backup_finished_prefix_is_idempotent(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        tokens, boundary_node = self._run_chunked_req_with_empty_finish(
            cache, allocator, req_to_token_pool
        )
        self.assertTrue(boundary_node.backuped)
        pending_before = len(cache.ongoing_write_through)

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        cache._backup_finished_prefix(match.last_device_node)

        self.assertEqual(len(cache.ongoing_write_through), pending_before)


if __name__ == "__main__":
    unittest.main()
