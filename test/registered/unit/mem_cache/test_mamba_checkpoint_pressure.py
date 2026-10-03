"""A missing optional checkpoint must not abort a request or the scheduler."""

import unittest
from array import array
from unittest.mock import patch

import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMambaCheckpointPressure(unittest.TestCase):
    def fixture(self, extra_buffer):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.MAMBA),
            mamba_cache_size=8,
            enable_mamba_extra_buffer=extra_buffer,
        )
        with patch("test_unified_radix_cache_unittest.get_device", return_value="cpu"):
            cache, allocator, pool = build_fixture(cfg)
        return cache, allocator, pool

    def request(self, cache, pool, rid):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.assertIsNotNone(pool.alloc([req]))
        req.last_node = cache.root_node_handle()
        req.lock_receipt = cache.inc_lock_ref(req.last_node).to_dec_params()
        return req

    def extend(self, req, allocator, pool, length):
        previous = len(req.prefix_indices)
        indices = allocator.alloc(length - previous)
        pool.write((req.kv.req_pool_idx, slice(previous, length)), indices)
        req.origin_input_ids = array("q", range(length))
        req.full_untruncated_fill_ids = array("q", range(length))
        req.set_extend_range(previous, length)
        req.kv.kv_committed_len = length
        req.kv.mamba_last_track_seqlen = length

    def test_exhausted_optional_checkpoint_keeps_request_state_and_next_request_runs(
        self,
    ):
        for extra_buffer in (False, True):
            with self.subTest(extra_buffer=extra_buffer):
                cache, allocator, pool = self.fixture(extra_buffer)
                req = self.request(cache, pool, "first")
                self.extend(req, allocator, pool, 64)
                main_slot = req.kv.mamba_pool_idx.clone()
                buffer = (
                    req.kv.mamba_ping_pong_track_buffer.clone()
                    if extra_buffer
                    else None
                )
                held = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
                cache.cache_unfinished_req(req, chunked=True)
                self.assertEqual(len(req.prefix_indices), 64)
                self.assertEqual(req.kv.cache_protected_len, 0)
                self.assertTrue(torch.equal(req.kv.mamba_pool_idx, main_slot))
                if extra_buffer:
                    self.assertTrue(
                        torch.equal(req.kv.mamba_ping_pong_track_buffer, buffer)
                    )
                self.assertFalse(req.finished())
                cache.cache_finished_req(req, is_insert=False, owned_kv_len=64)
                pool.free(req)
                pool.mamba_allocator.free(held)
                self.assertEqual(pool.mamba_allocator.available_size(), 8)
                self.assertEqual(allocator.available_size(), 256)
                next_req = self.request(cache, pool, "second")
                self.extend(next_req, allocator, pool, 64)
                cache.cache_unfinished_req(next_req, chunked=True)
                self.assertEqual(next_req.kv.cache_protected_len, 64)

    def test_existing_protected_prefix_survives_skipped_checkpoint_and_finish(self):
        cache, allocator, pool = self.fixture(True)
        req = self.request(cache, pool, "protected")
        self.extend(req, allocator, pool, 64)
        cache.cache_unfinished_req(req, chunked=True)
        last_node, receipt = req.last_node, req.lock_receipt
        # Protect the prefix checkpoint even when skip-decode-lock is enabled.
        extra_lock = cache.inc_lock_ref(last_node).to_dec_params()
        held = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
        self.extend(req, allocator, pool, 128)
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(req.last_node, last_node)
        self.assertEqual(req.lock_receipt, receipt)
        self.assertEqual(req.kv.cache_protected_len, 64)
        self.assertEqual(len(req.prefix_indices), 128)
        cache.cache_finished_req(req, is_insert=False, owned_kv_len=128)
        pool.free(req)
        cache.dec_lock_ref(last_node, extra_lock)
        pool.mamba_allocator.free(held)
        self.assertEqual(allocator.available_size(), 192)
        self.assertEqual(pool.mamba_allocator.available_size(), 7)

    def test_caching_resumes_on_next_chunk_after_capacity_returns(self):
        cache, allocator, pool = self.fixture(True)
        req = self.request(cache, pool, "resume")
        self.extend(req, allocator, pool, 64)
        held = pool.mamba_allocator.alloc(pool.mamba_allocator.available_size())
        cache.cache_unfinished_req(req, chunked=True)
        pool.mamba_allocator.free(held)
        self.extend(req, allocator, pool, 128)
        cache.cache_unfinished_req(req, chunked=True)
        self.assertEqual(req.kv.cache_protected_len, 128)
        self.assertFalse(req.finished())


if __name__ == "__main__":
    unittest.main()
