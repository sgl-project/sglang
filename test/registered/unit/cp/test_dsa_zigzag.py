"""DSA CP query segments must follow the strategy's actual local row order."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.layers.cp import base, utils, zigzag
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSAZigzag(CustomTestCase):
    def test_segments_preserve_causal_attention(self):
        torch.manual_seed(1)
        for cp_size in (2, 4):
            for prefix in ([0, 0], [5, 2]):
                extend = [9, 13]
                lengths = [n + p for n, p in zip(extend, prefix)]
                q = torch.randn(sum(extend), 4)
                keys = [torch.randn(n, 4) for n in lengths]
                values = [torch.randn(n, 3) for n in lengths]
                expected = []
                start = 0
                for req, (n, p) in enumerate(zip(extend, prefix)):
                    for pos in range(n):
                        scores = q[start + pos] @ keys[req][: p + pos + 1].T
                        expected.append(scores.softmax(-1) @ values[req][: p + pos + 1])
                    start += n
                expected = torch.stack(expected)
                full_visible = torch.cat(
                    [torch.arange(p + 1, p + n + 1) for n, p in zip(extend, prefix)]
                ).int()
                for rank in range(cp_size):
                    with self.subTest(cp_size=cp_size, prefix=prefix, rank=rank):
                        parallel = SimpleNamespace(attn_cp_rank=rank)
                        with (
                            patch.object(base, "get_parallel", return_value=parallel),
                            patch.object(
                                zigzag,
                                "get_device",
                                return_value=SimpleNamespace(device="cpu"),
                            ),
                        ):
                            strategy = zigzag.ZigzagCPStrategy(cp_size)
                            batch = SimpleNamespace(
                                input_ids=torch.arange(sum(extend)),
                                batch_size=2,
                                extend_seq_lens_cpu=extend,
                                seq_lens_cpu=torch.tensor(lengths),
                                forward_mode=SimpleNamespace(
                                    is_extend_without_speculative=lambda: True
                                ),
                                attn_cp_metadata=strategy.build_metadata(
                                    sum(extend), lengths, extend
                                ),
                            )
                            indices = strategy.local_q_indices(sum(extend), batch)
                            q_lens, q_lens_t, reqs, reqs_t = strategy.shard_per_request(
                                extend, torch.tensor(extend, dtype=torch.int32)
                            )
                            self.assertEqual(reqs, [0, 1, 0, 1])
                            self.assertEqual(sum(q_lens), len(indices))
                            self.assertEqual(q_lens_t.tolist(), q_lens)
                            self.assertEqual(reqs_t.tolist(), reqs)
                            visible = full_visible[indices]
                            key_lengths = torch.tensor(lengths, dtype=torch.int32)[
                                reqs_t
                            ]
                            backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
                                dsa_backend.DeepseekSparseAttnBackend
                            )
                            backend.device = "cpu"
                            (ks, ke), token_to_batch = backend._cal_indexer_k_start_end(
                                batch, reqs, q_lens_t, key_lengths, visible
                            )
                            self.assertEqual(
                                token_to_batch.tolist(),
                                [i for i, n in enumerate(q_lens) for _ in range(n)],
                            )
                            # RAGGED keys follow the same segment order as the indexer.
                            k = torch.cat([keys[i] for i in reqs])
                            v = torch.cat([values[i] for i in reqs])
                            actual = []
                            for row, lo, hi in zip(q[indices], ks, ke):
                                scores = row @ k[lo:hi].T
                                actual.append(scores.softmax(-1) @ v[lo:hi])
                            torch.testing.assert_close(
                                torch.stack(actual), expected[indices]
                            )
                            if cp_size == 2 and rank == 0:
                                self.assertEqual(
                                    indices.tolist(),
                                    [0, 1, 2, 9, 10, 11, 12, 7, 8, 19, 20, 21],
                                )
                                self.assertEqual(q_lens, [3, 4, 2, 3])

    def test_backend_metadata_uses_local_segments_and_global_cache(self):
        for prefix in ([0, 0], [5, 2]):
            for rank in range(2):
                for transform in (
                    dsa_backend.TopkTransformMethod.RAGGED,
                    dsa_backend.TopkTransformMethod.PAGED,
                ):
                    with self.subTest(prefix=prefix, rank=rank, transform=transform):
                        parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_size=2)
                        lengths = [9 + prefix[0], 13 + prefix[1]]
                        strategy = zigzag.ZigzagCPStrategy(2)
                        with (
                            patch.object(base, "get_parallel", return_value=parallel),
                            patch.object(
                                zigzag,
                                "get_device",
                                return_value=SimpleNamespace(device="cpu"),
                            ),
                        ):
                            meta = strategy.build_metadata(22, lengths, [9, 13])
                        meta.per_rank_logical_token = [12, 10]
                        meta.per_rank_actual_token = [12, 12]
                        batch = SimpleNamespace(
                            batch_size=2,
                            input_ids=torch.arange(22),
                            seq_lens=torch.tensor(lengths, dtype=torch.int32),
                            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32),
                            seq_lens_sum=sum(lengths),
                            req_pool_indices=torch.arange(2),
                            extend_seq_lens_cpu=[9, 13],
                            extend_seq_lens=torch.tensor([9, 13], dtype=torch.int32),
                            extend_prefix_lens_cpu=prefix,
                            extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32),
                            global_num_tokens_cpu=None,
                            attn_cp_metadata=meta,
                            forward_mode=SimpleNamespace(
                                is_target_verify=lambda: False,
                                is_decode_or_idle=lambda: False,
                                is_draft_extend_v2=lambda: False,
                                is_draft_extend=lambda: False,
                                is_extend=lambda: True,
                                is_extend_without_speculative=lambda: True,
                                is_context_parallel_extend=lambda: True,
                            ),
                        )
                        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
                            dsa_backend.DeepseekSparseAttnBackend
                        )
                        backend.device = "cpu"
                        page_table = torch.arange(30, dtype=torch.int32).reshape(2, 15)
                        backend.req_to_token_pool = SimpleNamespace(
                            req_to_token=page_table
                        )
                        backend.token_to_kv_pool = SimpleNamespace(
                            dtype=torch.bfloat16, size=100, page_size=1
                        )
                        backend.dsa_prefill_impl = "flashmla_sparse"
                        backend.use_mha = False
                        backend.dsa_index_kpool = 1
                        backend.dsa_index_topk = 2048
                        backend.physical_page_size = 1
                        backend.set_dsa_prefill_impl = lambda _: None
                        backend.get_topk_transform_method = lambda _: transform
                        backend.dsa_topk_backend = SimpleNamespace(
                            should_use_topk_v2=lambda: False
                        )
                        backend._init_kpool_metadata = (
                            lambda metadata, *args, **kwargs: metadata
                        )
                        backend._arange_buf = torch.arange(100, dtype=torch.int32)
                        with (
                            patch.object(base, "get_parallel", return_value=parallel),
                            patch.object(
                                dsa_backend, "get_cp_strategy", return_value=strategy
                            ),
                            patch.object(
                                dsa_backend, "dsa_use_prefill_cp", return_value=True
                            ),
                            patch.object(
                                dsa_backend, "is_cp_active", return_value=True
                            ),
                            patch.object(
                                dsa_utils, "get_parallel", return_value=parallel
                            ),
                            patch.object(
                                dsa_utils, "dsa_use_prefill_cp", return_value=True
                            ),
                            patch.object(utils, "is_cp_active", return_value=True),
                        ):
                            backend.init_forward_metadata(batch)
                            local_indices = strategy.local_q_indices(22, batch)
                        result = backend.forward_metadata
                        counts = [3, 4, 2, 3] if rank == 0 else [2, 3, 2, 3]
                        self.assertEqual(result.dsa_extend_seq_lens_list, counts)
                        self.assertEqual(
                            result.indexer_seq_lens_cpu.tolist(), lengths * 2
                        )
                        self.assertEqual(
                            result.cache_seqlens_int32.tolist(), lengths * 2
                        )
                        self.assertEqual(
                            result.page_table_1.tolist(),
                            page_table[[0, 1, 0, 1], : max(lengths)].tolist(),
                        )
                        visible = torch.cat(
                            [
                                torch.arange(p + 1, p + n + 1)
                                for n, p in zip([9, 13], prefix)
                            ]
                        ).int()[local_indices]
                        torch.testing.assert_close(result.dsa_seqlens_expanded, visible)
                        self.assertEqual(
                            result.dsa_cache_seqlens_int32.tolist(),
                            visible.tolist() + [0] * (12 - len(visible)),
                        )
                        if transform == dsa_backend.TopkTransformMethod.RAGGED:
                            self.assertEqual(
                                result.page_table_1_flattened.tolist(),
                                page_table[0, : lengths[0]].tolist()
                                + page_table[1, : lengths[1]].tolist()
                                + page_table[0, : lengths[0]].tolist()
                                + page_table[1, : lengths[1]].tolist(),
                            )
                            torch.testing.assert_close(
                                result.topk_indices_offset,
                                result.indexer_k_start_end[0],
                            )
                        else:
                            self.assertIsNone(result.page_table_1_flattened)
                        self.assertEqual(batch.extend_seq_lens_cpu, [9, 13])
                        self.assertEqual(batch.seq_lens_cpu.tolist(), lengths)

    def test_ragged_forward_uses_segment_keys_and_ignores_padding(self):
        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        backend.dsa_prefill_impl = "flashmla_sparse"
        backend.use_mha = False
        backend.use_fused_topk = False
        backend.hisparse_coordinator = None
        backend._resolve_kpool_tail_backend = lambda indices, impl: impl
        backend._check_kpool_tail_backend = lambda *args: None
        backend.get_topk_transform_method = lambda mode: (
            dsa_backend.TopkTransformMethod.RAGGED
        )
        backend._forward_flashmla_sparse = lambda **kwargs: kwargs
        cache = torch.arange(15).reshape(5, 1, 3).float()
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: cache)
        backend.forward_metadata = SimpleNamespace(
            topk_indices_offset=torch.tensor([0, 0, 5, 5]),
            page_table_1_flattened=torch.tensor([0, 1, 2, 3, 4] * 2),
            dsa_cache_seqlens_int32=torch.tensor([2, 3, 4, 5, 0, 0]),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False, is_draft_extend_v2=lambda: False
            ),
            extend_prefix_lens_cpu=[0],
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=1,
            head_dim=3,
            v_head_dim=2,
            scaling=1.0,
        )
        with (
            patch.object(
                dsa_backend,
                "dequantize_k_cache_paged",
                side_effect=AssertionError(
                    "BF16 cache must not use FP8 dequantization"
                ),
            ),
            patch.object(
                dsa_backend,
                "concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
        ):
            result = backend.forward_extend(
                torch.zeros(6, 1, 3),
                cache,
                cache,
                layer,
                batch,
                save_kv_cache=False,
                k_rope=cache[..., :0],
                topk_indices=torch.tensor(
                    [[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.int32
                ),
            )
        self.assertEqual(
            result["page_table_1"].tolist(),
            [[0, 1], [1, 2], [7, 8], [8, 9], [-1, -1], [-1, -1]],
        )
        torch.testing.assert_close(result["kv_cache"], torch.cat([cache, cache]))

    def test_gathers_remove_padding_and_restore_global_order(self):
        for rank in range(2):
            parallel = SimpleNamespace(attn_cp_rank=rank)
            with (
                patch.object(base, "get_parallel", return_value=parallel),
                patch.object(zigzag, "get_parallel", return_value=parallel),
                patch.object(
                    zigzag, "get_device", return_value=SimpleNamespace(device="cpu")
                ),
            ):
                strategy = zigzag.ZigzagCPStrategy(2)
                meta = strategy.build_metadata(22, [9, 13], [9, 13])
                meta.per_rank_logical_token = [12, 10]
                meta.per_rank_actual_token = [12, 12]
                batch = SimpleNamespace(attn_cp_metadata=meta)
                payload = torch.arange(22 * 6).reshape(22, 6).to(torch.uint8)
                shards = []
                for shard_rank in range(2):
                    with patch.object(
                        base,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_cp_rank=shard_rank),
                    ):
                        batch.attn_cp_metadata = strategy.build_metadata(
                            22, [9, 13], [9, 13]
                        )
                        batch.attn_cp_metadata.per_rank_logical_token = [12, 10]
                        batch.attn_cp_metadata.per_rank_actual_token = [12, 12]
                        shards.append(strategy.shard_hidden_states(payload, batch))
                batch.attn_cp_metadata = meta

                def gather(output, local):
                    torch.testing.assert_close(local, shards[rank])
                    output.copy_(torch.cat(shards))

                parallel.attn_cp_group = SimpleNamespace(all_gather_into_tensor=gather)
                actual = strategy.materialize_full_indexer_k_cache(shards[rank], batch)
                torch.testing.assert_close(actual, payload)
                fp8 = shards[rank].view(torch.float8_e4m3fn)
                k, rope = strategy.all_gather_dsa_trtllm_fp8_kv(
                    batch, fp8[:, :4], fp8[:, 4:]
                )
                torch.testing.assert_close(k.view(torch.uint8), payload[:, :4])
                torch.testing.assert_close(rope.view(torch.uint8), payload[:, 4:])

    def test_interleave_indexer_ranges_keep_selected_request_offsets(self):
        # CP3 rank 2 takes packed Q rows 2 and 5: requests 0 and 2,
        # while the one-token middle request contributes no query.
        batch = SimpleNamespace(
            batch_size=3,
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        backend.device = "cpu"
        (ks, ke), reqs = backend._cal_indexer_k_start_end(
            batch,
            [0, 2],
            torch.tensor([1, 1]),
            torch.tensor([8, 7], dtype=torch.int32),
            torch.tensor([8, 6], dtype=torch.int32),
        )
        self.assertEqual(ks.tolist(), [0, 8])
        self.assertEqual(ke.tolist(), [8, 14])
        self.assertEqual(reqs.tolist(), [0, 1])

    def test_dense_mla_writer_contract_is_unchanged(self):
        from unittest.mock import Mock

        strategy = zigzag.ZigzagCPStrategy(2)
        k = torch.arange(12).reshape(3, 1, 4).float()
        rope = k[..., :2]
        full = torch.cat([torch.cat((k, rope), dim=-1)] * 2)
        pool = SimpleNamespace(set_mla_kv_buffer=Mock())
        batch = SimpleNamespace(
            out_cache_loc=torch.arange(6), out_cache_loc_is_physical=False
        )
        with (
            patch.object(strategy, "gather_kv_cache", return_value=full),
            patch.object(zigzag, "get_token_to_kv_pool", return_value=pool),
        ):
            result = strategy.materialize_full_mla_kv(batch, object(), k, rope)
        self.assertIsNone(result)
        self.assertEqual(pool.set_mla_kv_buffer.call_count, 1)
        args = pool.set_mla_kv_buffer.call_args.args
        torch.testing.assert_close(args[2], full[..., :4])
        torch.testing.assert_close(args[3], full[..., 4:])

    def test_dsa_mla_gather_returns_pair_without_dense_cache_write(self):
        k = torch.arange(12).reshape(3, 1, 4).float()
        rope = k[..., :2] + 100
        full = torch.cat([torch.cat([k, rope], -1)] * 2)
        strategy = SimpleNamespace(gather_kv_cache=lambda x, batch: full)
        with (
            patch.object(dsa_backend, "get_cp_strategy", return_value=strategy),
            patch.object(dsa_backend, "dsa_use_prefill_cp", return_value=True),
            patch.object(dsa_backend, "is_cp_active", return_value=True),
        ):
            actual_k, actual_rope = dsa_backend.prepare_kv_for_attention(
                None, object(), k, rope, defer_materialization=False
            )
        torch.testing.assert_close(actual_k, full[..., :4])
        torch.testing.assert_close(actual_rope, full[..., 4:])

    def test_moe_dispatch_masks_physical_padding(self):
        for rank, logical in enumerate((12, 10)):
            strategy = zigzag.ZigzagCPStrategy(2)
            batch = SimpleNamespace(
                input_ids=torch.arange(22),
                num_token_non_padded=torch.tensor(22),
                attn_cp_metadata=SimpleNamespace(
                    per_rank_logical_token=[12, 10],
                    per_rank_actual_token=[12, 12],
                    moe_local_token_count=None,
                ),
            )
            with patch.object(
                base, "get_parallel", return_value=SimpleNamespace(attn_cp_rank=rank)
            ):
                actual = strategy.moe_num_token_non_padded(batch)
                self.assertEqual(actual.item(), logical)
                self.assertIs(strategy.moe_num_token_non_padded(batch), actual)

    def test_rank_major_input_ids_follow_zigzag_hidden_rows(self):
        parallel = SimpleNamespace(attn_cp_rank=0, attn_cp_size=2)
        with (
            patch.object(base, "get_parallel", return_value=parallel),
            patch.object(
                zigzag, "get_device", return_value=SimpleNamespace(device="cpu")
            ),
        ):
            strategy = zigzag.ZigzagCPStrategy(2)
            batch = SimpleNamespace(
                attn_cp_metadata=strategy.build_metadata(22, [9, 13], [9, 13])
            )
        # Two physical shards of 12 rows; rank 1 has only ten logical rows.
        batch.attn_cp_metadata.per_rank_logical_token = [12, 10]
        batch.attn_cp_metadata.per_rank_actual_token = [12, 12]
        ids = torch.arange(22)
        with (
            patch.object(utils, "is_cp_active", return_value=True),
            patch.object(utils, "get_parallel", return_value=parallel),
            patch.object(utils, "get_cp_strategy", return_value=strategy),
            patch.object(
                utils,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
        ):
            actual = utils.cp_interleave_input_ids(ids, batch)
        expected = [
            0,
            1,
            2,
            9,
            10,
            11,
            12,
            7,
            8,
            19,
            20,
            21,
            3,
            4,
            13,
            14,
            15,
            5,
            6,
            16,
            17,
            18,
            0,
            0,
        ]
        self.assertEqual(actual.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
