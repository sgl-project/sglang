"""DSA page addressing and byte-preservation contracts without a scheduler."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAMetadata,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseHiCacheBuffer
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


class TestHiSparseHiCacheBuffer(CustomTestCase):
    def _build(self, dtype=torch.bfloat16, *, attach=True):
        size, page, layers = 1024, 64, 2
        self.pool = DSATokenToKVPool(
            size=size,
            page_size=page,
            kv_lora_rank=512,
            dtype=dtype,
            qk_rope_head_dim=64,
            layer_num=layers,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=656 if dtype == torch.float8_e4m3fn else 576,
            start_layer=5,
            end_layer=7,
            index_buf_size=3 * size + page,
        )
        self.reqs = ReqToTokenPool(
            size=4, max_context_len=512, device="cuda", enable_memory_saver=False
        )
        self.cache = HiSparseHiCacheBuffer(
            req_to_token_pool=self.reqs,
            token_to_kv_pool=self.pool,
            top_k=page,
            buffer_tokens=page,
            block_size=256,
            device="cuda",
        )
        self.req_ids = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        self.seq_lens = torch.full((2,), 256, dtype=torch.int32, device="cuda")
        self.device_tokens = torch.arange(64, 320, device="cuda")
        self.reqs.req_to_token[self.req_ids, :256] = self.device_tokens.int()
        self.cache.protected_lens[self.req_ids] = 256
        for rid in (1, 2):
            self.cache.assign_temp_locations(
                rid, torch.arange(512 + 64 * rid, 576 + 64 * rid, device="cuda")
            )
        self.cache.prepare_forward(
            SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=2)
        )
        self.expected_kv, self.expected_index = [], []
        generator = torch.Generator(device="cuda").manual_seed(73)
        for kv, index in zip(self.pool.kv_buffer, self.cache.index_buffers):
            view = kv.view(torch.uint8)
            view[self.device_tokens] = torch.randint(
                256,
                (256, *view.shape[1:]),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
            pages = self.device_tokens[::page] // page
            index[pages] = torch.randint(
                256,
                (4, index.shape[1]),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )
            self.expected_kv.append(view[self.device_tokens].clone())
            self.expected_index.append(index[pages].clone())
        host_size = 2 * size + page
        row_bytes = self.pool.kv_buffer[0].view(torch.uint8).stride(0)
        # Layer-interleaved rows with padding exercise the runtime host stride.
        self.host_storage = torch.zeros(
            (host_size, layers, row_bytes + 16), dtype=torch.uint8, pin_memory=True
        )
        self.host = SimpleNamespace(
            size=host_size,
            data_refs=[self.host_storage[:, i, :row_bytes] for i in range(layers)],
        )
        starts = torch.tensor([host_size - page, 0, 128, 256], device="cuda")
        self.host_tokens = (
            starts[:, None] + torch.arange(page, device="cuda")
        ).flatten()
        for host, expected in zip(self.host.data_refs, self.expected_kv):
            host[self.host_tokens.cpu()] = expected.cpu().reshape(256, row_bytes)
        if attach:
            self.cache.attach_host_pool(self.host)

    def _demote(self, logical):
        device = self.device_tokens[logical]
        host = self.host_tokens[logical]
        self.cache.preserve_index(device, host)
        for rid in (1, 2):
            self.cache.host_locs[rid, logical] = host
            self.reqs.req_to_token[rid, logical] = -1
        for kv, index in zip(self.pool.kv_buffer, self.cache.index_buffers):
            kv.view(torch.uint8)[device] = 0
            index[device[::64] // 64] = 0

    def _assert_swap(self, topk):
        for layer, expected in enumerate(self.expected_kv):
            selected = (topk + layer * 64) % 256
            locs = self.cache.swap_in_selected_pages(
                self.req_ids, self.seq_lens, selected, self.pool.start_layer + layer
            )
            actual = self.pool.kv_buffer[layer].view(torch.uint8)[locs.long()]
            torch.testing.assert_close(actual, expected[selected.long()])

    def test_shared_indexes_and_kv_survive_device_reuse(self):
        """Two requests share host-addressed indexes after device pages are reused."""
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                self._build(dtype)
                self._demote(torch.arange(256, device="cuda"))
                pages = self.cache.indexer_page_table(
                    req_pool_indices=self.req_ids, num_pages=4
                )
                expected_pages = self.cache.base_pages + self.host_tokens[::64] // 64
                torch.testing.assert_close(pages, expected_pages.int().expand(2, -1))
                self.assertEqual(pages.max().item(), self.cache.index_pages - 1)
                for index, expected in zip(
                    self.cache.index_buffers, self.expected_index
                ):
                    torch.testing.assert_close(
                        index[pages.long()], expected.expand(2, -1, -1)
                    )
                for start in (0, 32, 192):
                    topk = torch.stack(
                        [
                            torch.arange(
                                start, start + 64, device="cuda", dtype=torch.int32
                            ),
                            torch.arange(
                                192 - start,
                                256 - start,
                                device="cuda",
                                dtype=torch.int32,
                            ),
                        ]
                    )
                    self._assert_swap(topk)

    def test_resident_and_host_tokens_share_one_selection(self):
        """Resident selections retain their physical slots beside swapped misses."""
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                self._build(dtype)
                self._demote(torch.arange(64, 192, device="cuda"))
                selected = (
                    torch.cat(
                        [
                            torch.arange(16, device="cuda"),
                            torch.arange(64, 96, device="cuda"),
                            torch.arange(240, 256, device="cuda"),
                        ]
                    )
                    .int()
                    .expand(2, -1)
                    .contiguous()
                )
                resident = (selected < 64) | (selected >= 192)
                for layer, expected in enumerate(self.expected_kv):
                    locs = self.cache.swap_in_selected_pages(
                        self.req_ids,
                        self.seq_lens,
                        selected,
                        self.pool.start_layer + layer,
                    )
                    torch.testing.assert_close(
                        locs[resident].long(),
                        self.device_tokens[selected.long()][resident],
                    )
                    torch.testing.assert_close(
                        self.pool.kv_buffer[layer].view(torch.uint8)[locs.long()],
                        expected[selected.long()],
                    )

    def test_request_reuse_discards_hot_entries_without_resetting_other_requests(self):
        """A reused request ID cannot hit the previous request's hot KV bytes."""
        self._build()
        self._demote(torch.arange(256, device="cuda"))
        selected = (
            torch.arange(64, dtype=torch.int32, device="cuda")
            .expand(2, -1)
            .contiguous()
        )
        self._assert_swap(selected)
        self.cache.release(1)
        new_host = torch.arange(512, 768, device="cuda")
        self.cache.host_locs[1, :256] = new_host
        self.cache.protected_lens[1] = 256
        self.cache.assign_temp_locations(1, torch.arange(576, 640, device="cuda"))
        for layer, expected in enumerate(self.expected_kv):
            replacement = expected ^ 255
            self.host.data_refs[layer][new_host.cpu()] = replacement.cpu().reshape(
                256, -1
            )
            layer_topk = (selected + layer * 64) % 256
            locs = self.cache.swap_in_selected_pages(
                self.req_ids, self.seq_lens, layer_topk, self.pool.start_layer + layer
            )
            actual = self.pool.kv_buffer[layer].view(torch.uint8)[locs.long()]
            torch.testing.assert_close(actual[0], replacement[layer_topk[0].long()])
            torch.testing.assert_close(actual[1], expected[layer_topk[1].long()])

    def _backend(self, indexer_table):
        q = torch.arange(3, dtype=torch.int32, device="cuda")
        k = q * 256
        token_table = self.reqs.req_to_token[self.req_ids, :256]
        metadata = DSAMetadata(
            page_size=64,
            cache_seqlens_int32=self.seq_lens,
            max_seq_len_q=1,
            max_seq_len_k=256,
            cu_seqlens_q=q,
            cu_seqlens_k=k,
            page_table_1=token_table,
            real_page_table=token_table[:, ::64] // 64,
            dsa_cache_seqlens_int32=self.seq_lens,
            dsa_cu_seqlens_q=q,
            dsa_cu_seqlens_k=k,
            dsa_extend_seq_lens_list=[1, 1],
            dsa_seqlens_expanded=self.seq_lens,
            indexer_page_table=indexer_table,
        )
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.forward_metadata = metadata
        backend.use_fused_topk = True
        backend.hisparse_coordinator = None
        backend.dsa_kv_cache_store_fp8 = False
        backend.dsa_prefill_impl = "fa3"
        backend.dsa_shards_index_k = False
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.dsa_packed_extend_topk = False
        backend.attn_dcp_size = 1
        return backend

    def test_graph_replay_sees_late_host_binding_and_independent_index_metadata(self):
        """Captured swaps resolve a later host attachment and remapped index pages."""
        self._build(torch.float8_e4m3fn, attach=False)
        topk = torch.arange(128, dtype=torch.int32, device="cuda").view(2, 64)
        table = torch.empty((2, 4), dtype=torch.int32, device="cuda")
        locations = torch.empty_like(topk)
        backend = self._backend(table)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        wrappers = [backend.get_indexer_metadata(layer, batch) for layer in (5, 6)]
        self.assertIs(wrappers[0].get_page_table_64(), wrappers[1].get_page_table_64())
        original_attention_pages = backend.forward_metadata.real_page_table.clone()

        def forward():
            table.copy_(
                self.cache.indexer_page_table(
                    req_pool_indices=self.req_ids, num_pages=4
                )
            )
            locations.copy_(
                self.cache.swap_in_selected_pages(self.req_ids, self.seq_lens, topk, 5)
            )

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        self.cache.attach_host_pool(self.host)
        self._demote(torch.arange(256, device="cuda"))
        topk[1, -3:] = -1
        graph.replay()
        torch.cuda.synchronize()
        valid = topk >= 0
        torch.testing.assert_close(
            self.pool.kv_buffer[0].view(torch.uint8)[locations[valid].long()],
            self.expected_kv[0][topk[valid].long()],
        )
        self.assertTrue((locations[~valid] == -1).all().item())
        host_pages = (
            (self.cache.base_pages + self.host_tokens[::64] // 64).int().expand(2, -1)
        )
        for wrapper in wrappers:
            torch.testing.assert_close(wrapper.get_page_table_64(), host_pages)
            torch.testing.assert_close(wrapper.get_logical_page_table(), host_pages)
            self.assertIs(
                wrapper.get_page_table_1(), backend.forward_metadata.page_table_1
            )
        torch.testing.assert_close(
            backend.forward_metadata.real_page_table, original_attention_pages
        )
        backend.forward_metadata = replace(
            backend.forward_metadata, indexer_page_table=None
        )
        ordinary = backend.get_indexer_metadata(5, batch)
        self.assertIs(
            ordinary.get_page_table_64(), backend.forward_metadata.real_page_table
        )
        backend.forward_metadata = replace(
            backend.forward_metadata,
            indexer_page_table=table,
            indexer_logical_page_table=original_attention_pages,
        )
        self.assertIs(
            backend.get_indexer_metadata(5, batch).get_logical_page_table(),
            original_attention_pages,
        )

    def test_host_binding_cannot_exceed_index_address_space(self):
        """Host capacity must fit the page-addressed index extension, including padding."""
        self._build()
        too_large = SimpleNamespace(
            size=self.host.size + 64, data_refs=self.host.data_refs
        )
        with self.assertRaisesRegex(ValueError, "cover the HiCache host pool"):
            self.cache.attach_host_pool(too_large)

    def test_multiple_queries_are_rejected_before_swap(self):
        self._build()
        for requests, queries in ((1, 2), (2, 4), (2, 6)):
            with (
                self.subTest(requests=requests, queries=queries),
                self.assertRaisesRegex(ValueError, "query"),
            ):
                self.cache.swap_in_selected_pages(
                    self.req_ids[:requests],
                    self.seq_lens[:requests],
                    torch.zeros((queries, 64), dtype=torch.int32, device="cuda"),
                    5,
                )


if __name__ == "__main__":
    unittest.main()
