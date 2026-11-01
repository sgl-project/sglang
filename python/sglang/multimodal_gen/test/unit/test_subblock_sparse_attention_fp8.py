# SPDX-License-Identifier: Apache-2.0
"""Contract checks for the SM100 sparse Sage FP8 backend."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    subblock_sparse_attn_fp8 as fp8,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class TestSubBlockSageContract(unittest.TestCase):
    def make_impl(self, config=None):
        with (
            patch(
                "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
                return_value=SimpleNamespace(attention_backend_config=config or {}),
            ),
            patch.object(
                fp8.SubBlockSparseAttentionFP8Impl,
                "_build_dense_impl",
                return_value=Mock(),
            ),
        ):
            return fp8.SubBlockSparseAttentionFP8Impl(14, 128, prefix="blocks.0.attn")

    def test_router_contract_independent_of_general_sage_architecture_defaults(self):
        for config in ({}, {"compute_mode": "sage_fp8"}):
            impl = self.make_impl(config)
            self.assertEqual(impl.schedule.n_k, 4)
            self.assertEqual(impl.schedule.n_q, 4)
            self.assertEqual(impl.schedule.skip_first_steps, 10)
            self.assertEqual(impl.schedule.skip_first_layers, 0)
            self.assertEqual(impl.schedule.sparsity, 0.75)
            self.assertTrue(
                fp8.SubBlockSparseAttentionFP8Backend.supports_packed_varlen()
            )
            self.assertTrue(AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN_FP8.is_sparse)
            self.assertEqual(
                str(fp8.SubBlockSparseAttentionFP8Backend.get_enum()),
                "subblock_sparse_attn_fp8",
            )

    def test_first_ten_steps_stay_dense(self):
        impl = self.make_impl()
        q = torch.empty(1, 4096, 14, 128, dtype=torch.bfloat16)
        with patch.object(
            fp8, "get_forward_context", return_value=SimpleNamespace(current_timestep=9)
        ):
            impl.forward(q, q, q)
        impl.dense_impl.forward.assert_called_once()
        with (
            patch.object(
                fp8,
                "get_forward_context",
                return_value=SimpleNamespace(current_timestep=10),
            ),
            patch.object(impl, "_sparse_attention", return_value=q) as sparse,
        ):
            self.assertIs(impl.forward(q, q, q), q)
        sparse.assert_called_once()

    def test_unsupported_batch_and_gqa_use_dense(self):
        impl = self.make_impl()
        for batch, q_heads, kv_heads in [(2, 14, 14), (1, 14, 7)]:
            with self.subTest(batch=batch, kv_heads=kv_heads):
                q = torch.empty(batch, 4096, q_heads, 128, dtype=torch.bfloat16)
                k = torch.empty(batch, 4096, kv_heads, 128, dtype=torch.bfloat16)
                impl.dense_impl.reset_mock()
                with (
                    patch.object(
                        fp8,
                        "get_forward_context",
                        return_value=SimpleNamespace(current_timestep=10),
                    ),
                    patch.object(impl, "_sparse_attention") as sparse,
                ):
                    impl.forward(q, k, k)
                impl.dense_impl.forward.assert_called_once()
                sparse.assert_not_called()

    def test_single_document_returns_attention_storage_directly(self):
        impl = self.make_impl()
        q = torch.empty(4097, 14, 128, dtype=torch.bfloat16)
        output = torch.empty_like(q).unsqueeze(0)
        with (
            patch.object(impl, "_sparse_ready", return_value=True),
            patch.object(impl, "_sparse_attention", return_value=output),
        ):
            result = impl.forward_varlen(
                q,
                q,
                q,
                cu_seqlens=torch.tensor([0, 4097]),
                max_seqlen=4097,
                cu_seqlens_host=(0, 4097),
            )
        self.assertEqual(result.data_ptr(), output.data_ptr())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0),
        "requires SM100 and FlashInfer sparse Sage FP8 dependencies",
    )
    def test_real_strided_quantization_and_sparse_tail(self):
        impl = self.make_impl({"min_seq_len": 1, "skip_first_steps": 0})
        torch.manual_seed(42)
        packed = (
            torch.randn(1, 652, 3, 14, 128, device="cuda", dtype=torch.bfloat16) * 0.5
        )
        q, k, v = (packed[:, :, i] for i in range(3))
        quantizer = fp8.flashinfer.trtllm_sage_attention_quantize
        with (
            patch.object(
                fp8,
                "get_forward_context",
                return_value=SimpleNamespace(current_timestep=10),
            ),
            patch.object(
                fp8.flashinfer, "trtllm_sage_attention_quantize", wraps=quantizer
            ) as quantize,
        ):
            result = impl.forward(q, k, v)
        self.assertEqual(quantize.call_count, 1)
        call = quantize.call_args
        self.assertTrue(call.kwargs["allow_strided_input"])
        self.assertFalse(call.kwargs["smooth_k"])
        self.assertEqual(call.kwargs["q_block_size"], 1)
        self.assertEqual(call.kwargs["k_block_size"], 16)
        self.assertEqual(call.kwargs["qk_quant_dtype"], torch.float8_e4m3fn)
        self.assertEqual(call.args[0].data_ptr(), q.data_ptr())
        self.assertEqual(call.args[0].stride(), (3 * 14 * 128, 128, 1))
        bounds = torch.tensor([0, 652], device="cuda", dtype=torch.int32)
        q8, k8, v8, qs, ks, vs = quantizer(
            *(x.squeeze(0).contiguous() for x in (q, k, v)),
            q_block_size=1,
            k_block_size=16,
            qk_quant_dtype=torch.float8_e4m3fn,
            cum_seq_lens_q=bounds,
            cum_seq_lens_kv=bounds,
            smooth_k=False,
        )
        plan = impl.router.route(
            q, k, sparsity=impl.schedule.sparsity, softmax_scale=impl.softmax_scale
        )
        expected = fp8.load_sage_attention()(
            q8.unsqueeze(0),
            k8.unsqueeze(0),
            v8.unsqueeze(0),
            plan.index,
            plan.topk,
            q_scale=qs.unsqueeze(0),
            k_scale=ks.unsqueeze(0),
            v_scale=vs,
            softmax_scale=impl.softmax_scale,
            kv_splits="auto",
        )[0]
        self.assertEqual(result.shape, q.shape)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_short_documents_remain_dense_and_uncovered_tail_is_zero(self):
        impl = self.make_impl()
        q = torch.empty(4100, 14, 128, dtype=torch.bfloat16)
        with (
            patch.object(impl, "_sparse_ready", return_value=True),
            patch.object(
                impl,
                "_sparse_attention",
                side_effect=lambda q, k, v, *, out: out.fill_(1),
            ),
            patch.object(
                impl,
                "_dense_segment",
                side_effect=lambda q, k, v: torch.full_like(q, 2),
            ),
        ):
            result = impl.forward_varlen(
                q,
                q,
                q,
                cu_seqlens=torch.tensor([0, 4096, 4098]),
                max_seqlen=4096,
                cu_seqlens_host=(0, 4096, 4098),
            )
        self.assertTrue(torch.all(result[:4096] == 1))
        self.assertTrue(torch.all(result[4096:4098] == 2))
        self.assertTrue(torch.all(result[4098:] == 0))


if __name__ == "__main__":
    unittest.main()
