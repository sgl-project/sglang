"""Native FP8 block selection against independent dense FP32 scoring."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm AITER indexer")
class TestMiniMaxAiterIndexer(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )
        from sglang.srt.layers.attention.minimax_sparse_ops.aiter_indexer import (
            AiterMiniMaxIndexer,
        )

        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            self.skipTest("The native M3 indexer requires gfx950")
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SGLANG_M3_USE_AITER_INDEXER", None)
        os.environ.pop("SGLANG_M3_USE_AITER_PREFILL_INDEXER", None)
        self.backend_type = MiniMaxSparseAttnBackend
        self.indexer_type = AiterMiniMaxIndexer
        torch.manual_seed(9182)
        torch.backends.cuda.matmul.allow_tf32 = False

    def make_backend(self, cache, heads=1, max_context=8192, query_len=1):
        backend = self.backend_type.__new__(self.backend_type)
        backend.kv_pool = SimpleNamespace(
            main_pool=SimpleNamespace(head_num=heads),
            get_index_k_buffer=lambda _: cache,
        )
        backend.hisparse_coordinator = None
        backend.indexer_cp = None
        backend.use_dense_sparse_decode = False
        backend.score_type = "max"
        backend.sparse_layer_ids = [0]
        backend.disable_value_layer_ids = {0}
        backend.block_size_k = backend.page_size = 128
        backend.idx_head_dim = 128
        backend.topk_blocks = 16
        backend.init_blocks = 0
        backend.local_blocks = 1
        backend.max_context_len = max_context
        backend._init_aiter_indexer(heads, max_query_len=query_len)
        return backend

    def exercise(self, q_lens, lengths, heads=1, prefill=False, max_context=8192):
        device = "cuda"
        num_pages = 2 * (max_context // 128) + 1
        table = torch.zeros((4, max_context), device=device, dtype=torch.int32)
        ids = torch.tensor([3, 1], device=device, dtype=torch.int64)
        pages = torch.randperm(num_pages - 1, device=device) + 1
        table[ids] = (
            (pages[:, None] * 128 + torch.arange(128, device=device))
            .reshape(2, -1)
            .to(torch.int32)
        )
        cache = torch.randn(num_pages * 128, 1, 128, device=device).to(
            torch.float8_e4m3fn
        )
        query = torch.randn(
            sum(q_lens), heads, 128, device=device, dtype=torch.bfloat16
        )
        lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        cu_q = torch.tensor(
            [0, q_lens[0], sum(q_lens)], device=device, dtype=torch.int32
        )
        backend = self.make_backend(
            cache, heads, max_context, query_len=1 if prefill else q_lens[0]
        )
        indexer = backend.aiter_indexer
        self.assertIsInstance(indexer, self.indexer_type)
        self.assertTrue(backend.aiter_prefill_indexer)

        def run():
            if prefill:
                indexer.prepare_prefill(
                    table,
                    ids,
                    lens,
                    cu_q,
                    total_q=sum(q_lens),
                    max_query_len=max(q_lens),
                    max_seq_len=max_context,
                )
            else:
                indexer.prepare(table, ids, lens, q_lens[0])
            return indexer.select(query, cache)

        def check(selection):
            query_offset = 0
            for request, q_len, final_len in zip(ids.tolist(), q_lens, lens.tolist()):
                for local in sorted({0, q_len // 2, q_len - 1}):
                    row = query_offset + local
                    length = final_len - q_len + local + 1
                    physical = table[request, :length].long()
                    keys = cache[:, 0].float()[physical]
                    logits = query[row].to(torch.float8_e4m3fn).float() @ keys.T
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
                        self.assertTrue((selected[selected < 0] == -1).all().item())
                        self.assertLess(blocks.max().item(), scores.shape[-1])
                        self.assertTrue(
                            (scores[head, blocks] >= thresholds[head] - 1e-4)
                            .all()
                            .item()
                        )
                        self.assertIn(scores.shape[-1] - 1, blocks.tolist())
                        positions = (
                            blocks[:, None] * 128 + torch.arange(128, device=device)
                        ).reshape(-1)
                        positions = positions[positions < length]
                        physical = table[request, positions].long()
                        self.assertEqual(
                            selection.context_lens[row * heads + head].item(),
                            positions.numel(),
                        )
                        emitted = selection.block_table[row * heads + head]
                        valid_pages = (positions.numel() + 15) // 16
                        expected_pages = sorted(
                            set((physical // 16 * heads + head).tolist())
                        )
                        self.assertEqual(
                            sorted(emitted[:valid_pages].tolist()), expected_pages
                        )
                query_offset += q_len

        check(run())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selection = run()
        graph.replay()
        check(selection)
        ids.copy_(ids.flip(0))
        lens.copy_(
            torch.tensor(
                [max(q_lens[0], max(lengths) - 3), max(q_lens[1], max(lengths) - 7)],
                device=device,
                dtype=torch.int32,
            )
        )
        table[3].copy_(table[3].roll(128))
        graph.replay()
        check(selection)

    def test_decode_and_verify(self):
        self.exercise([1, 1], [129, 255])
        self.exercise([4, 4], [8192, 4093], heads=2)

    def test_ragged_prefill_and_first_chunk(self):
        self.exercise([33, 7], [33, 7], prefill=True)
        self.exercise([17, 5], [4093, 8192], heads=2, prefill=True)

    def test_long_context(self):
        self.exercise([4, 4], [1048576, 996579], max_context=1048576)

    def test_decode_cp_takes_precedence_over_aiter(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cache = torch.empty(256, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
        backend = self.make_backend(cache)
        backend.kv_pool.get_kv_buffer = lambda _: (cache, cache)
        backend._is_sparse_kv_cached_by_fusion = Mock(return_value=True)
        backend.is_hip = True
        backend.is_npu = backend.fp8_attn_gemm = backend._use_msa_decode = False
        backend.index_cache_enabled = True
        backend.block_size_q = 1
        backend._max_seqlen_k = 256
        backend.req_to_token = torch.arange(256, device="cuda").unsqueeze(0)
        q = torch.zeros(1, 1, 128, device="cuda", dtype=torch.bfloat16)
        topk = torch.zeros(1, 1, 16, device="cuda", dtype=torch.int32)
        backend._decode_topk_buf = {1: topk}
        backend.aiter_indexer = Mock()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([256], device="cuda", dtype=torch.int32),
            req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
        )
        layer = SimpleNamespace(
            layer_id=0,
            q_scale_float=1.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
            idx_q_scale_float=1.0,
            idx_k_scale_float=1.0,
            idx_v_scale_float=1.0,
        )
        with patch(
            "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
            "minimax_sparse_decode",
            return_value=(None, q),
        ) as decode:
            for cp in (None, object()):
                backend.indexer_cp = cp
                for reuse in (False, True):
                    with self.subTest(cp_enabled=cp is not None, reuse=reuse):
                        backend.aiter_indexer.reset_mock()
                        backend._topk_is_source = {0: not reuse}
                        backend.init_forward_metadata_in_graph(batch)
                        backend.forward_decode(
                            q, q, q, layer, batch, idx_q=q, idx_k=q, idx_v=None
                        )
                        self.assertEqual(
                            backend.aiter_indexer.prepare.call_count, int(cp is None)
                        )
                        self.assertEqual(
                            backend.aiter_indexer.forward.call_count,
                            int(cp is None and not reuse),
                        )
                        expected = (
                            topk
                            if reuse
                            else (
                                backend.aiter_indexer.forward.return_value
                                if cp is None
                                else None
                            )
                        )
                        self.assertIs(
                            decode.call_args.kwargs["cached_topk_idx"], expected
                        )
                        self.assertIs(decode.call_args.kwargs["indexer_cp"], cp)
                        self.assertIs(
                            decode.call_args.kwargs["topk_out"], None if reuse else topk
                        )

    def test_cp_verify_skips_aiter_metadata(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cache = torch.empty(256, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
        backend = self.make_backend(cache)
        backend.indexer_cp = object()
        backend.is_hip = True
        backend.is_npu = False
        backend.aiter_indexer = Mock()
        backend._init_rocm_linear_verify_metadata = Mock()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, extend_seq_lens=None
        )
        backend.init_forward_metadata_in_graph(batch)
        backend.aiter_indexer.prepare.assert_not_called()
        backend.aiter_indexer.prepare_prefill.assert_not_called()
        backend._init_rocm_linear_verify_metadata.assert_called_once_with(batch)

    def test_prefill_topk_reuse_is_scoped_to_batch(self):
        cache = torch.empty(256, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
        backend = self.make_backend(cache)
        backend.kv_pool.get_kv_buffer = lambda _: (cache, cache)
        backend._is_sparse_kv_cached_by_fusion = Mock(return_value=True)
        backend.is_npu = backend.fp8_attn_gemm = backend.use_msa = False
        backend.index_cache_enabled = True
        backend.disable_value_layer_ids = {0, 1}
        backend._topk_group_of_layer = {0: 0, 1: 0}
        backend._topk_is_source = {0: True, 1: False}
        backend._topk_cache = {}
        backend._topk_cache_owner = None
        backend._loc_mapping = None
        backend.block_size_q = 1
        backend._max_seqlen_q = backend._max_seqlen_k = 3
        backend.req_to_token = torch.arange(256, device="cuda").unsqueeze(0)
        q = torch.zeros(3, 16, 128, device="cuda", dtype=torch.bfloat16)
        cu_q = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor([3], device="cuda", dtype=torch.int32)
        prefix_lens = torch.zeros_like(seq_lens)
        first_topk = torch.zeros(1, 3, 16, device="cuda", dtype=torch.int32)
        next_topk = torch.ones_like(first_topk)
        backend.aiter_indexer.forward = Mock(side_effect=[first_topk, next_topk])
        layer = SimpleNamespace(
            layer_id=0,
            q_scale_float=1.0,
            k_scale_float=1.0,
            v_scale_float=1.0,
            idx_q_scale_float=1.0,
            idx_k_scale_float=1.0,
            idx_v_scale_float=1.0,
        )

        def consume(*args, cached_topk_idx, return_topk_idx, **kwargs):
            result = (None, torch.zeros_like(args[0]))
            return (*result, cached_topk_idx) if return_topk_idx else result

        def run(batch, layer_id):
            layer.layer_id = layer_id
            backend._prefill_seqblock_meta = (
                batch,
                cu_q,
                seq_lens,
                prefix_lens,
                cu_q,
                3,
                3,
            )
            backend.forward_extend(
                q, q, q, layer, batch, idx_q=q[:, :1], idx_k=q[:, :1], idx_v=None
            )

        batch = SimpleNamespace(
            extend_seq_lens_cpu=[3],
            seq_lens_cpu=torch.tensor([3]),
            req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
        )
        with patch(
            "sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse."
            "minimax_sparse_prefill",
            side_effect=consume,
        ) as attention:
            run(batch, 0)
            run(batch, 1)
            self.assertEqual(backend.aiter_indexer.forward.call_count, 1)
            self.assertIs(attention.call_args.kwargs["cached_topk_idx"], first_topk)

            # A different batch must not reuse the previous batch's selection,
            # even when its first call is a layer that normally reuses top-k.
            run(SimpleNamespace(**vars(batch)), 1)
            self.assertEqual(backend.aiter_indexer.forward.call_count, 2)
            self.assertIs(attention.call_args.kwargs["cached_topk_idx"], next_topk)

    def test_disabling_indexer_also_disables_prefill(self):
        cache = torch.empty(256, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
        with patch.dict(os.environ, {"SGLANG_M3_USE_AITER_INDEXER": "0"}):
            backend = self.make_backend(cache)
        self.assertIsNone(backend.aiter_indexer)
        self.assertFalse(backend.aiter_prefill_indexer)

    def test_bf16_index_cache_keeps_existing_indexer(self):
        cache = torch.empty(256, 1, 128, dtype=torch.bfloat16, device="cuda")
        backend = self.make_backend(cache)
        self.assertIsNone(backend.aiter_indexer)
        self.assertFalse(backend.aiter_prefill_indexer)

    def test_non_rocm_keeps_existing_indexer(self):
        cache = torch.empty(256, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
        with patch(
            "sglang.srt.layers.attention.minimax_sparse_backend.is_hip",
            return_value=False,
        ):
            backend = self.make_backend(cache)
        self.assertIsNone(backend.aiter_indexer)
        self.assertFalse(backend.aiter_prefill_indexer)


if __name__ == "__main__":
    unittest.main()
