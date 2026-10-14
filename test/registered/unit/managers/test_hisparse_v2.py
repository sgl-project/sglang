"""HiCache ownership and DSA byte-preservation contracts for HiSparse V2."""

import atexit
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_v2_cache import HiSparseV2Cache
from sglang.srt.managers.hisparse_v2_coordinator import HiSparseV2Coordinator
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_metadata_kit import make_backend
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class TestHiSparseV2(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_hisparse_v2=True,
            enable_hierarchical_cache=True,
            page_size=64,
            hicache_ratio=2,
            hicache_write_policy="write_back",
            hicache_mem_layout="page_first",
        )
        override.install()
        self.addCleanup(override.restore)
        self.live_reqs = []
        self.tree = None

    def _build(self, dtype=torch.bfloat16):
        self.pool = DSATokenToKVPool(
            size=4096,
            page_size=64,
            kv_lora_rank=512,
            dtype=dtype,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=656 if dtype == torch.float8_e4m3fn else 576,
            index_buf_size=16384,
        )
        self.reqs = ReqToTokenPool(
            size=4, max_context_len=8192, device="cuda", enable_memory_saver=False
        )
        self.allocator = PagedTokenToKVPoolAllocator(
            size=4096,
            page_size=64,
            dtype=dtype,
            device="cuda",
            kvcache=self.pool,
            need_sort=False,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.reqs,
            token_to_kv_pool_allocator=self.allocator,
            page_size=64,
            tree_components=(ComponentType.FULL,),
            tree_core_backend="python",
        )
        self.tree = UnifiedRadixCache(params)
        self.tree.init_hicache(get_server_args(), params)
        self.cache = HiSparseV2Cache(
            req_to_token_pool=self.reqs,
            token_to_kv_pool=self.pool,
            top_k=64,
            buffer_tokens=64,
            block_size=256,
            device="cuda",
        )
        self.coordinator = HiSparseV2Coordinator(
            cache=self.cache, allocator=self.allocator
        )
        self.coordinator.attach_tree(self.tree)

    def _prefill(self, rid="a", length=256):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", range(length)),
            sampling_params=SamplingParams(max_new_tokens=16),
        )
        req.refresh_fill_ids()
        req.extend_end = length
        self.assertTrue(self.coordinator.reserve_prefill(req, max_new_tokens=16))
        self.reqs.alloc([req])
        req.kv.kv_committed_len = req.kv.kv_allocated_len = length
        req.last_node = self.tree.root_node
        indices = self.allocator.alloc(length)
        self.reqs.req_to_token[req.kv.req_pool_idx, :length] = indices.int()
        for buffer in self.pool.kv_buffer:
            view = buffer.view(torch.uint8)
            view[indices] = torch.randint(
                256, (length, *view.shape[1:]), dtype=torch.uint8, device="cuda"
            )
        for buffer in self.pool.index_k_with_scale_buffer:
            buffer.random_(0, 255)
        self.tree.checkpoint(req, up_to=length)
        self.live_reqs.append(req)
        return req

    def _finish(self, req):
        req.finished_reason = FINISH_LENGTH(1)
        release_kv_cache(req, self.tree, checkpoint=True)
        self.live_reqs.remove(req)

    def tearDown(self):
        if self.tree is not None:
            for req in list(self.live_reqs):
                self._finish(req)
            self.tree.shutdown()
            atexit.unregister(self.tree.shutdown)
            self.coordinator.destroy()
            self.tree.release_host_resources()

    def _check_eviction_and_reuse(self, dtype):
        self._build(dtype)
        req = self._prefill()
        rid = req.kv.req_pool_idx
        indices = self.reqs.req_to_token[rid, :256].long().clone()
        expected_kv = [
            b.view(torch.uint8)[indices].clone() for b in self.pool.kv_buffer
        ]
        expected_index = [
            b[indices[::64] // 64].clone() for b in self.pool.index_k_with_scale_buffer
        ]
        self.assertTrue(self.coordinator.admit_request(req))
        self.assertEqual(
            self.tree.evict(EvictParams(num_tokens=256)).num_tokens_evicted, 256
        )
        self.assertTrue((self.reqs.req_to_token[rid, :256] == -1).all().item())

        reused = self.allocator.alloc(256)
        self.addCleanup(self.allocator.free, reused)
        for buffer in self.pool.kv_buffer:
            buffer.view(torch.uint8)[reused] = 0
        for buffer in self.pool.index_k_with_scale_buffer:
            buffer[reused[::64] // 64] = 0
        req_ids = torch.tensor([rid], device="cuda")
        pages = self.cache.indexer_page_table(req_pool_indices=req_ids, num_pages=4)
        for layer_id, buffer in enumerate(self.pool.index_k_with_scale_buffer):
            torch.testing.assert_close(
                buffer[pages[0].long()], expected_index[layer_id]
            )

        self.cache.num_real_reqs.fill_(1)
        # The second selection reuses some hot slots and evicts others.
        for start in (0, 32, 192):
            topk = torch.arange(
                start, start + 64, dtype=torch.int32, device="cuda"
            ).view(1, -1)
            for layer_id in range(2):
                out = self.cache.swap_in_selected_pages(
                    req_ids, torch.tensor([256], device="cuda"), topk, layer_id
                )
                actual = self.pool.kv_buffer[layer_id].view(torch.uint8)[out[0].long()]
                torch.testing.assert_close(
                    actual, expected_kv[layer_id][start : start + 64]
                )
        self._finish(req)
        self.assertEqual(self.coordinator.reserved_temp_tokens(), 0)

    def test_bf16_prefix_survives_device_reuse(self):
        self._check_eviction_and_reuse(torch.bfloat16)

    def test_fp8_payload_scale_and_rope_survive_device_reuse(self):
        self._check_eviction_and_reuse(torch.float8_e4m3fn)

    def _metadata_fixture(self):
        self._build()
        reqs = [self._prefill("a"), self._prefill("b", length=320)]
        for req in reqs:
            self.assertTrue(self.coordinator.admit_request(req))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=4,
            req_pool_indices=torch.tensor(
                [req.kv.req_pool_idx for req in reqs] + [0, 0], device="cuda"
            ),
            seq_lens=torch.tensor([256, 320, 0, 0], device="cuda"),
            seq_lens_cpu=torch.tensor([256, 320, 0, 0]),
            seq_lens_sum=576,
            global_num_tokens_cpu=None,
            spec_info=None,
            out_cache_loc=None,
            actual_forward_mode=None,
        )
        backend = make_backend(
            batch.forward_mode,
            batch.seq_lens,
            batch.req_pool_indices,
            index_kpool=1,
            hisparse_v2_cache=self.cache,
        )
        return backend, batch

    def _shared_indexer_table(self, backend, batch):
        table = backend.get_indexer_metadata(0, batch).get_page_table_64()
        for layer_id in range(1, 61):
            self.assertIs(
                backend.get_indexer_metadata(layer_id, batch).get_page_table_64(),
                table,
            )
        return table

    def test_indexer_table_shared_per_forward_and_refreshed_after_eviction(self):
        """All layers reuse one mapping, and the next forward sees demoted pages."""
        backend, batch = self._metadata_fixture()
        expected = (
            self.cache.req_to_token[batch.req_pool_indices[:2], :256:64] // 64
        ).int()
        backend.init_forward_metadata(batch)
        table = self._shared_indexer_table(backend, batch)
        torch.testing.assert_close(table[:2, :4], expected)

        self.assertGreater(
            self.tree.evict(EvictParams(num_tokens=320)).num_tokens_evicted, 0
        )
        batch.req_pool_indices[:2] = batch.req_pool_indices[:2].flip(0)
        batch.seq_lens[:2] = batch.seq_lens[:2].flip(0)
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        backend.init_forward_metadata(batch)
        table = self._shared_indexer_table(backend, batch)
        expected = self.cache.index_table[batch.req_pool_indices[:2], :4].int()
        torch.testing.assert_close(table[:2, :4], expected)

    def test_indexer_graph_replay_refreshes_mapping_without_rebinding(self):
        """Captured layers see eviction, request reorder and batch-size switches."""
        backend, batch = self._metadata_fixture()
        table = self._shared_indexer_table(backend, batch)
        output = torch.empty_like(table[:, :4])
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            output.copy_(
                backend.get_indexer_metadata(0, batch).get_page_table_64()[:, :4]
            )

        self.assertGreater(
            self.tree.evict(EvictParams(num_tokens=320)).num_tokens_evicted, 0
        )
        for bs in (4, 2, 4):
            batch.req_pool_indices[:2] = batch.req_pool_indices[:2].flip(0)
            batch.seq_lens[:2] = batch.seq_lens[:2].flip(0)
            current_batch = SimpleNamespace(**vars(batch))
            current_batch.batch_size = bs
            current_batch.req_pool_indices = batch.req_pool_indices[:bs]
            current_batch.seq_lens = batch.seq_lens[:bs]
            backend.init_forward_metadata_out_graph(current_batch)
            current = self._shared_indexer_table(backend, current_batch)
            expected = self.cache.index_table[batch.req_pool_indices[:2], :4].int()
            torch.testing.assert_close(current[:2, :4], expected)
            if bs == 4:
                self.assertIs(current, table)
                graph.replay()
                torch.testing.assert_close(output[:2], expected)

    def test_split_preserves_shared_host_lock_receipts(self):
        self._build()
        reqs = [self._prefill("a"), self._prefill("b")]
        for req in reqs:
            self.assertTrue(self.coordinator.admit_request(req))
        self.tree.evict(EvictParams(num_tokens=256))
        self.tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(128)))))
        host = self.cache.host_pool
        before = host.available_size()
        self.tree.evict_host(256)
        self.assertEqual(host.available_size(), before)
        self._finish(reqs[0])
        self.tree.evict_host(256)
        self.assertEqual(host.available_size(), before)
        self._finish(reqs[1])
        self.tree.evict_host(256)
        self.assertEqual(host.available_size(), before + 256)
        self.assertEqual(self.allocator.available_size(), self.pool.size)

    def test_host_pressure_cannot_drop_a_live_prefix(self):
        self._build()
        req = self._prefill()
        self.assertTrue(self.coordinator.admit_request(req))
        host = self.cache.host_pool
        held = host.alloc(host.available_size())
        self.addCleanup(host.free, held)
        self.assertEqual(
            self.tree.evict(EvictParams(num_tokens=256)).num_tokens_evicted, 0
        )
        self.assertTrue(
            (self.reqs.req_to_token[req.kv.req_pool_idx, :256] >= 0).all().item()
        )
        self._finish(req)
        self.assertEqual(
            self.tree.evict(EvictParams(num_tokens=256)).num_tokens_evicted, 256
        )

    def test_unfinished_prefill_releases_its_reservation(self):
        self._build()
        req = self._prefill()
        self.assertEqual(self.coordinator.reserved_temp_tokens(), 64)
        self._finish(req)
        self.assertFalse(self.coordinator.pending)
        self.tree.evict(EvictParams(num_tokens=256))
        self.assertEqual(self.allocator.available_size(), self.pool.size)


if __name__ == "__main__":
    unittest.main()
