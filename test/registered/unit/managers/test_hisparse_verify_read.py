"""Verify-cache ownership tests; hardware copies are covered separately."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiSparseVerifyGather(unittest.TestCase):
    def check_gather(self, batch_size, queries, padded):
        top_k = 4
        req_ids = torch.tensor([3, 1, 4, 2][:batch_size])
        real_count = batch_size - int(padded)
        prefixes = torch.tensor([0, 3, 7, 12, 5])
        row = torch.arange(32).repeat(5, 1) + torch.arange(5)[:, None] * 32
        mapping = torch.zeros(160, dtype=torch.int64)
        cache = torch.full((512, 1, 8), -999.0)
        topk = torch.full((batch_size, queries, top_k), -1, dtype=torch.int32)
        lens = torch.empty((batch_size, queries), dtype=torch.int64)
        expected = torch.empty((batch_size, queries, top_k))
        for b, rid in enumerate(req_ids.tolist()):
            prefix = int(prefixes[rid])
            generated = torch.arange(prefix, prefix + queries)
            locations = 256 + rid * 32 + torch.arange(queries)
            mapping[row[rid, generated]] = locations
            cache[locations] = (rid * 1000 + generated).float()[:, None, None]
            for q in range(queries):
                # Resident token first, then prompt rows including misses.
                tokens = torch.tensor(
                    [
                        prefix + q,
                        q % prefix,
                        (prefix - 1 - q) % prefix,
                        -1 if q % 2 == 0 else prefix + q + 1,
                    ]
                )
                topk[b, q] = tokens
                lens[b, q] = prefix + q + 1
                expected[b, q] = rid * 1000 + tokens

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = coordinator.is_m3_hisparse = False
        coordinator.enable_prefetch = False
        coordinator.top_k = top_k
        coordinator.spec_prefill_lens = prefixes
        coordinator.num_real_reqs = torch.tensor([real_count])
        coordinator.req_to_token_pool = SimpleNamespace(req_to_token=row)
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=[cache], full_to_hisparse_device_index_mapping=mapping
        )
        calls = []

        def swap(indices, swap_lens, tokens, layer):
            calls.append(tokens.clone())
            self.assertEqual(layer, 0)
            torch.testing.assert_close(
                swap_lens, (prefixes[indices] + 1).clamp(min=top_k)
            )
            output = torch.full_like(tokens, -1)
            for b, rid in enumerate(indices.tolist()[:real_count]):
                # Every call overwrites the SAME device slots. Returning views
                # without materializing query KV would corrupt earlier queries.
                for col, token in enumerate(tokens[b].tolist()):
                    if token < 0:
                        continue
                    self.assertLess(token, int(prefixes[rid]))
                    loc = rid * 8 + col
                    cache[loc] = rid * 1000 + token
                    output[b, col] = loc
            return output

        coordinator._run_swap_in_kernel = swap
        before_mapping = mapping.clone()
        materialized, pages = coordinator.gather_for_verify(req_ids, lens, topk, 0)
        self.assertEqual(len(calls), queries)
        self.assertTrue(all(torch.all(call[:, 0] == -1) for call in calls))
        valid = (topk >= 0) & (topk < lens[:, :, None])
        valid[real_count:] = False
        pages = pages.view_as(topk)
        torch.testing.assert_close(pages >= 0, valid)
        actual = materialized[pages.clamp(min=0).long(), 0, 0]
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
        torch.testing.assert_close(mapping, before_mapping)
        # The original resident cache bytes are never redirected or overwritten.
        for b, rid in enumerate(req_ids.tolist()):
            prefix = int(prefixes[rid])
            locs = mapping[row[rid, prefix : prefix + queries]]
            values = rid * 1000 + torch.arange(prefix, prefix + queries).float()
            torch.testing.assert_close(cache[locs, 0, 0], values)

    def test_c1_c4_mtp1_mtp3_mtp5(self):
        for bs in (1, 4):
            for queries in (2, 4, 6):
                for padded in (False, True):
                    with self.subTest(batch=bs, queries=queries, padded=padded):
                        self.check_gather(bs, queries, padded)

    def test_fp8_storage_is_returned_as_the_kv_dtype(self):
        # FP8 pools store uint8 bytes; attention must see float8 rows of the
        # same width, as get_key_buffer() returns them, or it picks the BF16
        # kernel and reads each row at twice its size.
        top_k, width = 4, 656
        storage = torch.arange(64 * width, dtype=torch.int64).remainder(251)
        storage = storage.to(torch.uint8).view(64, 1, width)
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = coordinator.is_m3_hisparse = False
        coordinator.enable_prefetch = False
        coordinator.top_k = top_k
        coordinator.spec_prefill_lens = torch.tensor([0, 2])
        coordinator.num_real_reqs = torch.tensor([1])
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(64).repeat(2, 1)
        )
        mapping = torch.zeros(65, dtype=torch.int64)
        mapping[:8] = torch.arange(8) + 8
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=[storage],
            dtype=torch.float8_e4m3fn,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator._run_swap_in_kernel = lambda idx, lens, tokens, layer: (
            torch.full_like(tokens, -1)
        )
        topk = torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int32)
        lens = torch.tensor([[3]])
        kv, pages = coordinator.gather_for_verify(torch.tensor([1]), lens, topk, 0)
        self.assertEqual(kv.dtype, torch.float8_e4m3fn)
        self.assertEqual(kv.shape, (top_k, 1, width))
        for col in range(3):
            torch.testing.assert_close(
                kv[pages[0, col]].view(torch.uint8), storage[mapping[col]]
            )

    def test_partial_prompt_page_gets_resident_backing(self):
        for prefix in (64, 65, 127, 511, 513, 577, 769):
            with self.subTest(prefix=prefix):
                mapping = torch.zeros(2048, dtype=torch.int64)
                host = torch.arange(1024).reshape(1, -1)
                restored = []
                allocate = Mock(side_effect=AssertionError("unbudgeted allocation"))
                coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
                coordinator.page_size = 64
                coordinator.device_buffer_size = 512
                coordinator.req_device_buffer_size = torch.tensor([576])
                coordinator.req_to_device_buffer = torch.arange(64, 640).view(1, -1)
                coordinator.token_to_kv_pool_allocator = SimpleNamespace(
                    hisparse_attn_allocator=SimpleNamespace(alloc=allocate)
                )
                coordinator.req_to_token_pool = SimpleNamespace(req_to_token=host + 128)
                coordinator.req_to_host_pool = host
                coordinator.mem_pool_device = SimpleNamespace(
                    layer_num=2, full_to_hisparse_device_index_mapping=mapping
                )
                coordinator.mem_pool_host = SimpleNamespace(
                    load_to_device_per_layer=lambda pool, src, dst, layer, **kw: (
                        restored.append((src.clone(), dst.clone(), layer))
                    )
                )
                req = SimpleNamespace(
                    kv=SimpleNamespace(kv_allocated_len=prefix, req_pool_idx=0)
                )
                coordinator._preserve_speculative_partial_page(req)
                allocate.assert_not_called()
                tail = prefix % 64
                if not tail:
                    self.assertFalse(restored)
                else:
                    buffer_start = min(prefix - tail, 512)
                    expected = torch.arange(64 + buffer_start, 64 + buffer_start + tail)
                    self.assertEqual(len(restored), 2)
                    for src, dst, layer in restored:
                        torch.testing.assert_close(
                            src, torch.arange(prefix - tail, prefix)
                        )
                        torch.testing.assert_close(dst, expected)
                    torch.testing.assert_close(
                        mapping[128 + prefix - tail : 128 + prefix],
                        expected,
                    )


class TestHiSparseVerifyEntry(unittest.TestCase):
    def test_verify_dispatch_uses_materialized_cache_and_raw_positions(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            dsa_seqlens_expanded=torch.tensor([321, 322]),
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend.dsa_decode_impl = backend.dsa_prefill_impl = "tilelang"
        backend.dsa_index_kpool = 1
        backend.use_mha = False
        backend.use_fused_topk = True
        backend.dsa_topk_backend = object()
        backend._resolve_kpool_tail_backend = lambda topk, impl: impl
        backend._check_kpool_tail_backend = lambda *args: None
        backend.get_topk_transform_method = lambda mode: "paged"
        materialized = torch.ones((6, 1, 8))
        pages = torch.arange(6, dtype=torch.int32).view(2, 3)
        gather = Mock(return_value=(materialized, pages))
        backend.hisparse_coordinator = SimpleNamespace(gather_for_verify=gather)
        translate = Mock(side_effect=AssertionError("must not reuse cleared mappings"))
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer: torch.zeros((8, 1, 8)),
            translate_loc_to_hisparse_device=translate,
        )
        backend._forward_tilelang = Mock(return_value=materialized)
        mode = SimpleNamespace(
            is_target_verify=lambda: True,
            is_draft_extend_v2=lambda: False,
            is_decode_or_idle=lambda: False,
        )
        batch = SimpleNamespace(
            forward_mode=mode, batch_size=1, req_pool_indices=torch.tensor([3])
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=1,
            v_head_dim=4,
            head_dim=8,
            scaling=1,
        )
        topk = torch.tensor([[0, 3, 319], [3, 70, 200]])
        self.assertTrue(backend.get_indexer_metadata(0, batch).force_unfused_topk)
        result = backend.forward_extend(
            torch.zeros((2, 1, 8)),
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
            topk_indices=topk,
        )
        self.assertIs(result, materialized)
        translate.assert_not_called()
        args = gather.call_args.args
        torch.testing.assert_close(args[0], batch.req_pool_indices)
        torch.testing.assert_close(args[1], torch.tensor([[321, 322]]))
        torch.testing.assert_close(args[2], topk.view(1, 2, 3))
        self.assertIs(
            backend._forward_tilelang.call_args.kwargs["kv_cache"], materialized
        )
        self.assertIs(backend._forward_tilelang.call_args.kwargs["page_table_1"], pages)


if __name__ == "__main__":
    unittest.main()
