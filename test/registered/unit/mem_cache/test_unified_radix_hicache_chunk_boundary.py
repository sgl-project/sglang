"""Regression test: a committed prefix ending at a chunk boundary must be backed up.

Chunk-boundary nodes are inserted via `cache_unfinished_req(chunked=True)`, and
`_inc_hit_count_and_check` deliberately skips their write-through trigger. The
finish-time insert normally repairs that by re-walking the same path with
`chunked=False`, but with mamba extra buffer a final extend chunk shorter than
`mamba_cache_chunk_size` leaves `mamba_last_track_seqlen` unset, so the Mamba
component reports an effective cache length of 0, the finish-time insert becomes
an empty-key no-op, and the already-inserted prefix silently never reaches
host/storage backup — cross-instance KV reuse then always misses for such
prompts.

`cache_finished_req` must leave the request's committed prefix backed up.
"""

import unittest
from array import array

import test_unified_radix_cache_unittest as base

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
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
            rid=str(self._rid),
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        type(self)._rid += 1
        req_to_token_pool.alloc([req])
        req.kv = ReqKvInfo(kv_allocated_len=0, swa_evicted_seqlen=0)
        return req

    def _run_chunked_req_with_empty_finish(
        self, cache, allocator, req_to_token_pool, *, expect_unbacked: bool = True
    ):
        """One chunked request whose finish-time commit is empty.

        The first chunk commits [0, kv_len) via `cache_unfinished_req(
        chunked=True)`; the final extend is then too short to be tracked
        (`mamba_last_track_seqlen` stays None), mirroring a prompt whose
        page-aligned length is an exact multiple of `chunked_prefill_size`.
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
        req.kv.kv_allocated_len = kv_len
        req.last_node = cache.root_node.id
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.mamba_last_track_seqlen = kv_len

        cache.cache_unfinished_req(req, chunked=True)
        # The chunked insert consumed the tracked seqlen; the short final extend
        # below never sets it again, which is what collapses the finish commit.
        self.assertIsNone(req.mamba_last_track_seqlen)

        boundary_node_id = req.last_node
        boundary_node = cache.resolve_node_handle(boundary_node_id)
        self.assertIsNot(boundary_node, cache.root_node)
        if expect_unbacked:
            self.assertFalse(boundary_node.backuped)

        output_ids = [2000]
        req.output_ids = array("q", output_ids)
        req.full_untruncated_fill_ids = array("q", tokens + output_ids)
        req.set_extend_range(kv_len, kv_len + 1)
        extra = allocator.alloc(1)
        self.assertIsNotNone(extra)
        req_to_token_pool.write((req.req_pool_idx, slice(kv_len, kv_len + 1)), extra)
        req.kv_committed_len = kv_len + 1
        req.kv.kv_allocated_len = kv_len + 1

        cache.cache_finished_req(
            req, is_insert=True, kv_len_to_handle=req.effective_kv_committed_len()
        )
        return tokens, boundary_node

    def test_chunk_boundary_prefix_backed_up_after_finish(self):
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        _, boundary_node = self._run_chunked_req_with_empty_finish(
            cache, allocator, req_to_token_pool
        )

        self.assertTrue(boundary_node.backuped)
        cache.sanity_check()

    def test_repeated_request_schedules_no_extra_backup(self):
        """A second request over the same prefix must not re-write it."""
        cache, allocator, req_to_token_pool = base.build_fixture(self.cfg)
        self._init_hicache(cache)
        cache.write_through_threshold = 1

        tokens, boundary_node = self._run_chunked_req_with_empty_finish(
            cache, allocator, req_to_token_pool
        )
        self.assertTrue(boundary_node.backuped)
        pending_before = len(cache.ongoing_write_through)

        # Same token sequence, so the second request lands on the same prefix.
        _, boundary_node_2 = self._run_chunked_req_with_empty_finish(
            cache, allocator, req_to_token_pool, expect_unbacked=False
        )
        self.assertIs(boundary_node_2, boundary_node)
        self.assertEqual(len(cache.ongoing_write_through), pending_before)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
