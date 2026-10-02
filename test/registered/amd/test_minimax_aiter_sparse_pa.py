"""Real AITER index selection, SHUFFLE stores, and sparse attention contracts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(torch.version.hip, "ROCm AITER operators")
class TestMiniMaxAiterSparsePA(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.aiter_indexer import (
            AiterMiniMaxIndexer,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.aiter_sparse_pa import (
            AiterMiniMaxSparsePA,
            sparse_kv_views,
        )
        from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            self.skipTest("The native M3 indexer requires gfx950")
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in (
            "SGLANG_M3_USE_AITER_INDEXER",
            "SGLANG_M3_USE_AITER_PREFILL_INDEXER",
            "SGLANG_M3_USE_AITER_SPARSE_PA",
        ):
            os.environ.pop(name, None)
        self.backend_type = MiniMaxSparseAttnBackend
        self.pool_type = MiniMaxSparseKVPool
        self.indexer_type = AiterMiniMaxIndexer
        self.pa_type = AiterMiniMaxSparsePA
        self.views = sparse_kv_views
        self.forward_mode = ForwardMode
        torch.manual_seed(9182)
        torch.backends.cuda.matmul.allow_tf32 = False

    def make_backend(
        self,
        heads=1,
        size=256,
        max_context=8192,
        query_len=1,
        main_dtype=torch.float8_e4m3fn,
        index_dtype=torch.float8_e4m3fn,
        sparse_layers=(1,),
        enable_hierarchical_cache=False,
        disaggregation_mode="null",
    ):
        backend = self.backend_type.__new__(self.backend_type)
        backend.kv_pool = self.pool_type(
            size=size,
            page_size=128,
            dtype=main_dtype,
            head_num=heads,
            head_dim=128,
            idx_head_dim=128,
            dense_layer_ids=[0],
            sparse_layer_ids=list(sparse_layers),
            device="cuda",
            disable_value_sparse_layer_ids=list(sparse_layers),
            index_dtype=index_dtype,
            start_layer=0,
            end_layer=max(sparse_layers) + 1,
        )
        backend.hisparse_coordinator = None
        backend.use_dense_sparse_decode = False
        backend.score_type = "max"
        backend.sparse_layer_ids = list(sparse_layers)
        backend.disable_value_layer_ids = set(sparse_layers)
        backend.block_size_k = backend.page_size = 128
        backend.idx_head_dim = 128
        backend.topk_blocks = 16
        backend.init_blocks = 0
        backend.local_blocks = 1
        backend.max_context_len = max_context
        backend.fp8_attn_gemm = False
        backend.is_hip = True
        backend.is_npu = backend.use_msa = backend._use_msa_decode = False
        backend.index_cache_enabled = False
        backend._topk_is_source = {}
        backend.speculative_num_draft_tokens = query_len
        backend._init_aiter_indexer(heads, max_query_len=query_len)
        backend._init_aiter_sparse_pa(
            enable_hierarchical_cache=enable_hierarchical_cache,
            disaggregation_mode=disaggregation_mode,
        )
        return backend

    def exercise(self, q_lens, lengths, heads=1, prefill=False, max_context=8192):
        dev = "cuda"
        num_pages = 2 * (max_context // 128) + 1
        slots = num_pages * 128
        table = torch.zeros((4, max_context), device=dev, dtype=torch.int32)
        ids = torch.tensor([3, 1], device=dev, dtype=torch.int64)
        pages = torch.randperm(num_pages - 1, device=dev) + 1
        table[ids] = (
            (pages[:, None] * 128 + torch.arange(128, device=dev))
            .reshape(2, -1)
            .to(torch.int32)
        )
        backend = self.make_backend(
            heads,
            size=slots - 128,
            max_context=max_context,
            query_len=1 if prefill else q_lens[0],
        )
        pool = backend.kv_pool
        self.assertIsInstance(backend.aiter_indexer, self.indexer_type)
        self.assertIsInstance(backend.aiter_sparse_pa, self.pa_type)
        self.assertTrue(pool.use_aiter_sparse_pa)
        idx_cache = pool.get_index_k_buffer(1)
        idx_cache.copy_(torch.randn(slots, 1, 128, device=dev).to(torch.float8_e4m3fn))
        k = torch.randn(slots, heads, 128, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_cache, v_cache = pool.get_kv_buffer(1)
        ks = torch.tensor([0.7], device=dev)
        vs = torch.tensor([1.3], device=dev)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=1, k_scale=ks, v_scale=vs),
            torch.arange(slots, device=dev),
            k,
            v,
            k_scale=0.7,
            v_scale=1.3,
        )
        expected_k = (k.float() / ks).to(torch.float8_e4m3fn).float()
        expected_v = (v.float() / vs).to(torch.float8_e4m3fn).float()
        kc, vc = self.views(k_cache, v_cache)
        torch.testing.assert_close(
            kc.float().permute(0, 3, 1, 2, 4).reshape_as(expected_k),
            expected_k,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            vc.float().permute(0, 2, 4, 1, 3).reshape_as(expected_v),
            expected_v,
            rtol=0,
            atol=0,
        )

        indexer = backend.aiter_indexer
        attend = backend.aiter_sparse_pa
        iq = torch.randn(sum(q_lens), heads, 128, device=dev, dtype=torch.bfloat16)
        q = torch.randn(sum(q_lens), heads * 16, 128, device=dev, dtype=torch.bfloat16)
        lens = torch.tensor(lengths, device=dev, dtype=torch.int32)
        cu = torch.tensor([0, q_lens[0], sum(q_lens)], device=dev, dtype=torch.int32)
        backend.req_to_token = table
        layer = SimpleNamespace(layer_id=1, scaling=128**-0.5, k_scale=ks, v_scale=vs)
        verifying = not prefill and q_lens[0] > 1
        batch = SimpleNamespace(
            forward_mode=(
                self.forward_mode.TARGET_VERIFY
                if verifying
                else self.forward_mode.DECODE
            ),
            req_pool_indices=ids,
            # The caches above are already populated and checked independently.
            minimax_m3_precached_sparse_layers={1},
        )

        def run():
            if prefill:
                indexer.prepare_prefill(
                    table,
                    ids,
                    lens,
                    cu,
                    total_q=sum(q_lens),
                    max_query_len=max(q_lens),
                    max_seq_len=max_context,
                )
            else:
                batch.seq_lens = lens - q_lens[0] if verifying else lens
                backend.init_forward_metadata_in_graph(batch)
                forward = (
                    backend.forward_extend if verifying else backend.forward_decode
                )
                _, out = forward(
                    q, None, None, layer, batch, idx_q=iq, idx_k=None, idx_v=None
                )
                return backend._aiter_decode_selection, out.reshape_as(q)
            selection = indexer.select(iq, idx_cache)
            out = attend.forward(
                q,
                k_cache,
                v_cache,
                selection,
                softmax_scale=128**-0.5,
                k_scale=ks,
                v_scale=vs,
            )
            return selection, out

        def check(selection, out):
            qo = 0
            for req, qlen, final_len in zip(ids.tolist(), q_lens, lens.tolist()):
                sample_rows = sorted(set([0, qlen // 2, qlen - 1]))
                for local in sample_rows:
                    row = qo + local
                    length = final_len - qlen + local + 1
                    physical = table[req, :length].long()
                    ik = idx_cache[:, 0].float()[physical]
                    logits = iq[row].to(torch.float8_e4m3fn).float() @ ik.T
                    scores = (
                        torch.nn.functional.pad(
                            logits, (0, (-length) % 128), value=-float("inf")
                        )
                        .reshape(heads, -1, 128)
                        .amax(-1)
                    )
                    scores[:, -1] = float("inf")
                    count = min(16, scores.shape[-1])
                    thresholds = scores.topk(count, dim=-1).values[:, -1]
                    for head in range(heads):
                        selected = selection.topk[head, row]
                        blocks = selected[selected >= 0].long()
                        self.assertEqual(blocks.numel(), count)
                        self.assertEqual(blocks.unique().numel(), count)
                        self.assertEqual((selected < 0).sum().item(), 16 - count)
                        self.assertTrue(
                            (scores[head, blocks] >= thresholds[head] - 1e-4)
                            .all()
                            .item()
                        )
                        self.assertIn(scores.shape[-1] - 1, blocks.tolist())
                        positions = (
                            blocks[:, None] * 128 + torch.arange(128, device=dev)
                        ).reshape(-1)
                        positions = positions[positions < length]
                        pp = table[req, positions].long()
                        qq = (
                            q[row, head * 16 : (head + 1) * 16]
                            .to(torch.float8_e4m3fn)
                            .float()
                        )
                        kk = expected_k[pp, head] * ks
                        vv = expected_v[pp, head] * vs
                        ref = ((qq @ kk.T) * (128**-0.5)).softmax(-1) @ vv
                        actual = out[row, head * 16 : (head + 1) * 16].float()
                        # FP8 probability rounding is also present in AITER's
                        # serving kernel; use its .055 absolute envelope plus L2.
                        error = (actual - ref).abs().max().item()
                        relative_l2 = ((actual - ref).norm() / ref.norm()).item()
                        self.assertLess(error, 0.055 * vs.item())
                        self.assertLess(relative_l2, 0.05)
                        self.assertEqual(
                            selection.context_lens[row * heads + head].item(),
                            positions.numel(),
                        )
                        emitted = selection.block_table[row * heads + head]
                        valid_pages = (positions.numel() + 15) // 16
                        expected_pages = sorted(set((pp // 16 * heads + head).tolist()))
                        self.assertEqual(
                            sorted(emitted[:valid_pages].tolist()), expected_pages
                        )
                qo += qlen

        selection, out = run()
        check(selection, out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selection, out = run()
        graph.replay()
        check(selection, out)
        ids.copy_(ids.flip(0))
        lens.copy_(
            torch.tensor(
                [max(q_lens[0], max(lengths) - 3), max(q_lens[1], max(lengths) - 7)],
                device=dev,
            )
        )
        table[3].copy_(table[3].roll(128))
        graph.replay()
        check(selection, out)

    def test_decode_and_verify(self):
        self.exercise([1, 1], [129, 255])
        self.exercise([4, 4], [8192, 4093], heads=2)

    def test_ragged_prefill_and_first_chunk(self):
        self.exercise([33, 7], [33, 7], prefill=True)
        self.exercise([17, 5], [4093, 8192], heads=2, prefill=True)

    def test_long_context(self):
        self.exercise([4, 4], [1048576, 996579], max_context=1048576)

    def test_prefill_reuses_selection_only_within_batch(self):
        backend = self.make_backend(max_context=256, sparse_layers=(1, 2))
        backend.index_cache_enabled = True
        backend._topk_group_of_layer = {1: 0, 2: 0}
        backend._topk_is_source = {1: True, 2: False}
        backend._topk_cache = {}
        backend._topk_cache_owner = None
        backend.block_size_q = 1
        backend._max_seqlen_q = backend._max_seqlen_k = 3
        select = Mock(wraps=backend.aiter_indexer.select)
        backend.aiter_indexer.select = select
        q = torch.randn(5, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(5, 1, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        iq, ik = torch.randn_like(k), torch.randn_like(k)
        ks = torch.tensor([0.7], device="cuda")
        vs = torch.tensor([1.3], device="cuda")
        layer = SimpleNamespace(
            layer_id=1,
            scaling=128**-0.5,
            k_scale=ks,
            v_scale=vs,
            k_scale_float=0.7,
            v_scale_float=1.3,
            idx_k_scale_float=None,
            idx_v_scale_float=None,
        )
        batch = SimpleNamespace(
            forward_mode=self.forward_mode.EXTEND,
            extend_seq_lens_cpu=[3],
            extend_seq_lens=torch.tensor([3], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([3], dtype=torch.int32, device="cuda"),
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device="cuda"),
            minimax_m3_precached_sparse_layers=None,
        )

        def run(batch, layer_id, first_slot, value_offset=0):
            layer.layer_id = layer_id
            batch.out_cache_loc = torch.tensor(
                [first_slot, first_slot + 1, first_slot + 2, 0, 0], device="cuda"
            )
            backend.req_to_token = torch.arange(
                first_slot, first_slot + 256, dtype=torch.int32, device="cuda"
            ).unsqueeze(0)
            cu = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
            backend._prefill_seqblock_meta = (
                batch,
                cu,
                batch.seq_lens,
                torch.zeros_like(batch.seq_lens),
                cu,
                3,
                3,
            )
            backend.init_forward_metadata_in_graph(batch)
            values = v + value_offset
            _, out = backend.forward_extend(
                q, k, values, layer, batch, idx_q=iq, idx_k=ik, idx_v=None
            )
            out = out.reshape_as(q)
            torch.testing.assert_close(out[3:], torch.zeros_like(out[3:]))
            # All three causal rows fit in the selected first block. Compare
            # actual backend writes + attention against a dense FP32 reference.
            kk = (k[:3, 0].float() / ks).to(torch.float8_e4m3fn).float() * ks
            vv = (values[:3, 0].float() / vs).to(torch.float8_e4m3fn).float() * vs
            for row in range(3):
                qq = q[row].to(torch.float8_e4m3fn).float()
                prob = ((qq @ kk[: row + 1].T) * layer.scaling).softmax(-1)
                torch.testing.assert_close(
                    out[row].float(), prob @ vv[: row + 1], atol=0.08, rtol=0.05
                )

        run(batch, 1, 128)
        first_selection = backend._topk_cache[0]
        run(batch, 2, 128)
        self.assertEqual(select.call_count, 1)
        self.assertIs(backend._topk_cache[0], first_selection)

        # Start a new batch on a reuse layer, with different physical pages
        # and values; stale selection metadata would read the previous values.
        run(SimpleNamespace(**vars(batch)), 2, 256, value_offset=1)
        self.assertEqual(select.call_count, 2)
        self.assertIsNot(backend._topk_cache[0], first_selection)

    def assert_nhd_store(self, pool):
        loc = torch.tensor([1, 2], device="cuda")
        k = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=1, k_scale=None, v_scale=None), loc, k, v
        )
        kc, vc = pool.get_kv_buffer(1)
        torch.testing.assert_close(kc.float()[loc], k.to(kc.dtype).float())
        torch.testing.assert_close(vc.float()[loc], v.to(vc.dtype).float())

    def test_cache_transfer_modes_keep_nhd_storage(self):
        for config in (
            {"enable_hierarchical_cache": True},
            {"disaggregation_mode": "prefill"},
            {"disaggregation_mode": "decode"},
        ):
            with self.subTest(config=config):
                backend = self.make_backend(**config)
                self.assertIsNone(backend.aiter_sparse_pa)
                self.assertFalse(backend.kv_pool.use_aiter_sparse_pa)
                self.assert_nhd_store(backend.kv_pool)

    def test_unit_scale_dense_and_sparse_stores(self):
        backend = self.make_backend()
        pool = backend.kv_pool
        loc = torch.tensor([17, 19], device="cuda")
        k = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for layer_id in (0, 1):
            pool.set_kv_buffer(
                # Omitting the scale arguments must still mean unit scale,
                # even if this layer also carries non-unit device scales.
                SimpleNamespace(
                    layer_id=layer_id,
                    k_scale=torch.tensor([0.7], device="cuda"),
                    v_scale=torch.tensor([1.3], device="cuda"),
                ),
                loc,
                k,
                v,
            )
        dense_k, dense_v = pool.get_kv_buffer(0)
        sparse_k, sparse_v = pool.get_kv_buffer(1)
        kc, vc = self.views(sparse_k, sparse_v)
        unpacked_k = kc.float().permute(0, 3, 1, 2, 4).reshape_as(sparse_k)
        unpacked_v = vc.float().permute(0, 2, 4, 1, 3).reshape_as(sparse_v)
        for stored, reference in (
            (dense_k.float()[loc], k),
            (dense_v.float()[loc], v),
            (unpacked_k[loc], k),
            (unpacked_v[loc], v),
        ):
            torch.testing.assert_close(stored, reference.to(sparse_k.dtype).float())

    def test_disabling_dependencies_keeps_nhd_writer(self):
        for name in (
            "SGLANG_M3_USE_AITER_INDEXER",
            "SGLANG_M3_USE_AITER_PREFILL_INDEXER",
            "SGLANG_M3_USE_AITER_SPARSE_PA",
        ):
            with self.subTest(disabled=name):
                with patch.dict(os.environ, {name: "0"}):
                    backend = self.make_backend()
                self.assertIsNone(backend.aiter_sparse_pa)
                pool = backend.kv_pool
                self.assertFalse(pool.use_aiter_sparse_pa)
                self.assert_nhd_store(pool)

    def test_bf16_index_cache_keeps_nhd_storage(self):
        backend = self.make_backend(index_dtype=torch.bfloat16)
        self.assertIsNone(backend.aiter_indexer)
        self.assertIsNone(backend.aiter_sparse_pa)
        self.assertFalse(backend.kv_pool.use_aiter_sparse_pa)

    def test_bf16_main_cache_keeps_nhd_storage(self):
        backend = self.make_backend(main_dtype=torch.bfloat16)
        self.assertIsInstance(backend.aiter_indexer, self.indexer_type)
        self.assertIsNone(backend.aiter_sparse_pa)
        self.assertFalse(backend.kv_pool.use_aiter_sparse_pa)


if __name__ == "__main__":
    unittest.main()
