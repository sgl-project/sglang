import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.radix_attention as radix_attention_module
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.model_executor.runner_utils.prefill_graph import prefill_graph_scope
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# GPU lane: the sparse RadixAttention op is registered for the CUDA dispatch key only.
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestMiniMaxSparseNvfp4IdleOutput(CustomTestCase):
    """An idle DP rank padded to non-zero tokens must still return a BF16 output.

    With the FP8 Q epilogue on, q arrives as FP8; an idle output sized off q
    would hand an FP8 tensor to o_proj on that rank.
    """

    def test_idle_output_follows_v_dtype(self):
        from sglang.srt.layers.attention.minimax_sparse_nvfp4_backend import (
            MiniMaxSparseNvfp4AttnBackend,
        )

        backend = MiniMaxSparseNvfp4AttnBackend.__new__(MiniMaxSparseNvfp4AttnBackend)
        num_tokens, heads, head_dim = 8, 4, 128
        q = torch.zeros(num_tokens, heads, head_dim, dtype=torch.float8_e4m3fn)
        k = torch.zeros(num_tokens, 1, head_dim, dtype=torch.bfloat16)
        v = torch.zeros(num_tokens, 1, head_dim, dtype=torch.bfloat16)
        layer = SimpleNamespace(tp_q_head_num=heads, v_head_dim=head_dim)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True)
        )

        _, out = backend.forward(q, k, v, layer, forward_batch)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (num_tokens, heads * head_dim))


class TestMiniMaxSparseNvfp4DeviceLengths(CustomTestCase):
    def test_prefill_with_prefix_does_not_need_cpu_sequence_lengths(self):
        from sglang.srt.layers.attention import minimax_sparse_nvfp4_backend as module

        backend = object.__new__(module.MiniMaxSparseNvfp4AttnBackend)
        backend.req_to_token = torch.arange(2048).reshape(2, 1024)
        backend._msa_v1_qmul4_available = True
        backend._msa_v1_num_index_heads = 4
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([269, 517], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            extend_seq_lens=torch.tensor([13, 5], dtype=torch.int32),
            extend_seq_lens_cpu=[13, 5],
            extend_prefix_lens_cpu=[256, 512],
            req_pool_indices=torch.tensor([1, 0]),
        )
        with patch.object(module, "MSAV1PrefillPlan", side_effect=SimpleNamespace):
            backend.init_forward_metadata_out_graph(batch)
        plan = backend.forward_metadata.msa_v1_prefill_plan
        self.assertEqual((plan.max_seqlen_q, plan.max_seqlen_k), (13, 517))
        self.assertEqual(plan.total_q, 18)
        torch.testing.assert_close(
            plan.cu_seqlens_q, torch.tensor([0, 13, 18], dtype=torch.int32)
        )
        torch.testing.assert_close(plan.seq_lens, batch.seq_lens)
        torch.testing.assert_close(
            plan.page_table,
            torch.tensor([[8, 9, 10, 11, 12], [0, 1, 2, 3, 4]], dtype=torch.int32),
        )


class TestBreakableSparseRadixAttention(CustomTestCase):
    def test_breakable_sparse_runs_on_real_rows_and_restores_batch(self):
        layer = RadixAttention(
            num_heads=2, head_dim=3, scaling=1.0, num_kv_heads=2, layer_id=0
        )
        query = torch.zeros((4, 2, 3), device="cuda")
        idx_q = torch.zeros((4, 1, 3), device="cuda")
        idx_k = torch.zeros((4, 1, 3), device="cuda")
        out_cache_loc = torch.arange(4, dtype=torch.int64, device="cuda")
        positions = torch.arange(4, dtype=torch.int64, device="cuda")
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_num_token_non_padded_cpu=2,
            out_cache_loc=out_cache_loc,
            positions=positions,
            _attn_output=None,
            mha_return_lse=False,
        )
        calls = []

        class KOnlySparseBackend:
            # A K-only layer has no index value, so the backend returns no index output.
            def forward(self, query, key, value, layer, batch, save_kv_cache, **kwargs):
                calls.append(
                    (
                        query.shape[0],
                        batch.out_cache_loc.shape[0],
                        kwargs["idx_q"].shape[0],
                    )
                )
                return None, torch.full_like(query, 3)

        with (
            forward_context(ForwardContext(KOnlySparseBackend())),
            set_forward_batch(forward_batch),
            prefill_graph_scope(full_graph=False, raw_num_tokens=2),
            patch.object(
                radix_attention_module, "is_in_breakable_cuda_graph", return_value=True
            ),
        ):
            _, output = layer(
                query, query, query, forward_batch, idx_q=idx_q, idx_k=idx_k
            )

        self.assertEqual(calls, [(2, 2, 2)])
        self.assertEqual(output.shape, (4, 6))
        self.assertTrue(torch.all(output[:2] == 3))
        self.assertTrue(torch.all(output[2:] == 0))
        self.assertIs(forward_batch.out_cache_loc, out_cache_loc)
        self.assertIs(forward_batch.positions, positions)


if __name__ == "__main__":
    unittest.main()
