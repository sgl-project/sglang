"""Write-back eviction must wait for the checkpoint's KV producer."""

import unittest
from array import array
from unittest.mock import patch

import torch
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    build_fixture,
)

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHiCacheWriteBackOrdering(unittest.TestCase):
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _make_req = UnifiedRadixCacheSuite._make_req
    _rid = 0

    def _request(self, cache, allocator, req_pool, start, length=8):
        req = self._make_req(req_pool)
        req.origin_input_ids = array("q", range(start, start + 2 * length))
        req.full_untruncated_fill_ids = array("q", range(start, start + length))
        req.output_ids = array("q")
        req.pending_bootstrap = False
        req.extend_end = length
        ids = allocator.alloc(length)
        self.assertIsNotNone(ids)
        req_pool.write((req.kv.req_pool_idx, slice(0, length)), ids)
        req.kv.kv_committed_len = length
        req.last_node = cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        return req

    def test_write_back_internal_swa_waits_for_checkpoint_producer(self):
        self.cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            page_size=1,
            sliding_window_size=4,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=64,
            max_context_len=64,
            head_num=1,
            head_dim=64,
        )
        cache, allocator, req_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy="write_back")
        controller = cache.cache_controller
        forward = torch.cuda.Stream()
        controller.write_fence_stream = forward
        warm = self._request(cache, allocator, req_pool, 100, length=12)
        cache.checkpoint(warm, up_to=12)
        warm_eviction = cache.evict(EvictParams(swa_num_tokens=1))
        self.assertGreaterEqual(warm_eviction.swa_num_tokens_evicted, 1)
        req = self._request(cache, allocator, req_pool, 0, length=12)
        kv = allocator.get_kvcache()
        buffers = [
            buf
            for pool in (kv.full_kv_pool, kv.swa_kv_pool)
            for buf in pool.k_buffer + pool.v_buffer
        ]
        for buf in buffers:
            buf.fill_(-1)
        read_done = torch.cuda.Event()
        read_done.record()
        torch.cuda.synchronize()
        engine = controller.l2_transfer_engine
        submit = engine.submit_device_to_host
        snapshots = []

        def record_copy(transfers):
            completion = submit(transfers)
            completion.finish_event.synchronize()
            for transfer in transfers:
                if transfer.host_indices.numel():
                    snapshots.append(
                        transfer.host_pool.get_data_page(
                            int(transfer.host_indices[0])
                        ).clone()
                    )
            return completion

        with torch.cuda.stream(forward):
            read_done.record()
            torch.cuda._sleep(600_000_000)
            for buf in buffers:
                buf.fill_(42)
        torch.cuda.current_stream().wait_event(read_done)
        try:
            with patch.object(engine, "submit_device_to_host", side_effect=record_copy):
                cache.checkpoint(req, up_to=12)
                self.assertEqual(controller.write_queue, [])
                # The active request pins Full and its final SWA window. An
                # older window from this same forward is still evictable.
                evicted = cache.evict(EvictParams(swa_num_tokens=1))
            self.assertGreaterEqual(evicted.swa_num_tokens_evicted, 1)
            self.assertTrue(snapshots)
            for snapshot in snapshots:
                self.assertTrue(
                    torch.equal(snapshot, torch.full_like(snapshot, 42)),
                    f"expected 42, copied {snapshot.unique().tolist()}",
                )
        finally:
            forward.synchronize()


if __name__ == "__main__":
    unittest.main()
