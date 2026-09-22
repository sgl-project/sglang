"""
Unit tests for the MXFP8 (quant_mode 3) lightning-indexer path:
- dsa_npu_indexer activation quantization (npu_dynamic_mx_quant + E8M0
  (d/64, 2) descale layout normalization);
- the torch.ops.cann_ops_transformer.quant_lightning_indexer /
  quant_lightning_indexer_metadata kernel contract (TND query, PA_BBND paged
  key, E8M0 descales, int32 seqlen tensors).

All tests require an Ascend NPU and the CANN ops-transformer kernels.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")

HAVE_NPU = is_npu() and torch.npu.is_available()

if HAVE_NPU:
    # Importing dsa_npu_indexer on NPU also registers the CANN
    # ops-transformer kernels under torch.ops.cann_ops_transformer.
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
        _check_quant_lightning_indexer_constraints,
        _quantize_npu_indexer_activation,
        create_npu_hadamard_128,
    )

# quant_lightning_indexer (v2) contract constants (quant_mode 3 / MXFP8).
QUANT_MODE_MXFP8 = 3
HEAD_DIM = 128
N_HEADS_Q = 64
N_HEADS_K = 1
TOPK = 64
MASK_MODE_RIGHT_DOWN_CAUSAL = 3
CMP_RATIO = 1

PAGE_SIZE = 64
SIZE = 256
KV_LORA_RANK = 128
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + KV_LORA_RANK // 128 * 4 + QK_ROPE_HEAD_DIM * 2


def _make_pool() -> NPUMLATokenToKVPool:
    return NPUMLATokenToKVPool(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=1,
        device="npu",
        enable_memory_saver=False,
        index_head_dim=HEAD_DIM,
        kv_cache_dim=KV_CACHE_DIM,
    )


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestCheckQuantLightningIndexerConstraints(unittest.TestCase):
    def test_page_size_64_accepted(self):
        _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=64))

    def test_page_size_16_accepted(self):
        _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=16))

    def test_page_size_1_rejected(self):
        with self.assertRaises(AssertionError):
            _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=1))

    def test_page_size_15_rejected(self):
        with self.assertRaises(AssertionError):
            _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=15))

    def test_page_size_320_accepted(self):
        # 320 == 20 * 16 lies in [16, 1024] and is a multiple of 16; the v2
        # spec imposes no power-of-2 requirement, so it is accepted.
        _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=320))

    def test_page_size_1040_rejected(self):
        # 1040 == 65 * 16 is a multiple of 16 but exceeds the 1024 bound.
        with self.assertRaises(AssertionError):
            _check_quant_lightning_indexer_constraints(SimpleNamespace(page_size=1040))


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestQuantizeNPUIndexerActivation(unittest.TestCase):
    def setUp(self):
        self.hadamard = create_npu_hadamard_128(HEAD_DIM, "npu")

    def test_2d_shapes_and_dtypes(self):
        x = torch.randn(7, HEAD_DIM, dtype=torch.bfloat16, device="npu")
        quantized, scale = _quantize_npu_indexer_activation(
            x, self.hadamard, torch.float8_e4m3fn
        )
        self.assertEqual(quantized.shape, (7, HEAD_DIM))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (7, HEAD_DIM // 64, 2))
        self.assertEqual(scale.dtype, torch.float8_e8m0fnu)

    def test_3d_shapes(self):
        x = torch.randn(3, 4, HEAD_DIM, dtype=torch.bfloat16, device="npu")
        quantized, scale = _quantize_npu_indexer_activation(
            x, self.hadamard, torch.float8_e4m3fn
        )
        self.assertEqual(quantized.shape, (3, 4, HEAD_DIM))
        self.assertEqual(scale.shape, (3, 4, 2, 2))
        self.assertEqual(scale.dtype, torch.float8_e8m0fnu)

    def test_empty_input(self):
        x = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device="npu")
        quantized, scale = _quantize_npu_indexer_activation(
            x, self.hadamard, torch.float8_e4m3fn
        )
        self.assertEqual(quantized.shape, (0, HEAD_DIM))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (0, 2, 2))
        self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
        self.assertTrue(torch.all(scale.view(torch.uint8) == 0))

    def test_dequant_consistency(self):
        # Dequantizing with the per-32-block E8M0 scales (block b ==
        # scale[..., b // 2, b % 2]) must reproduce the hadamard-rotated
        # input within e4m3 precision. The e8m0/e4m3 -> float32 casts are
        # unsupported on the NPU, so the dequant math runs on CPU.
        x = torch.randn(64, HEAD_DIM, dtype=torch.bfloat16, device="npu")
        quantized, scale = _quantize_npu_indexer_activation(
            x, self.hadamard, torch.float8_e4m3fn
        )
        rotated = x.cpu().float() @ self.hadamard.cpu().float()
        scales = scale.cpu().to(torch.float32)  # (64, 2, 2)
        quantized_cpu = quantized.cpu().to(torch.float32)
        dequantized = torch.empty_like(rotated)
        for b in range(HEAD_DIM // 32):
            block = slice(b * 32, (b + 1) * 32)
            dequantized[:, block] = (
                quantized_cpu[:, block] * scales[:, b // 2, b % 2].unsqueeze(-1)
            )
        torch.testing.assert_close(dequantized, rotated, rtol=0.10, atol=0.05)


@unittest.skipUnless(HAVE_NPU, "Ascend NPU device required")
class TestQuantLightningIndexerOpSmoke(unittest.TestCase):
    """End-to-end smoke of the production call shape: TND fp8 query, PA_BBND
    fp8 paged key from the pool, E8M0 descales, fp32 weights, right-down
    causal mask."""

    def setUp(self):
        self.pool = _make_pool()
        self.hadamard = self.pool.indexer_hadamard_128
        torch.manual_seed(0)

        # 3 full pages of index-k (192 tokens).
        self.num_k_tokens = 3 * PAGE_SIZE
        loc = torch.arange(self.num_k_tokens, dtype=torch.int64, device="npu")
        k_bf16 = torch.randn(
            self.num_k_tokens, N_HEADS_K, HEAD_DIM, dtype=torch.bfloat16, device="npu"
        )
        k_fp8, k_scale = _quantize_npu_indexer_activation(
            k_bf16, self.hadamard, torch.float8_e4m3fn
        )
        self.pool.set_index_k_buffer(0, loc, k_fp8)
        self.pool.set_index_k_scale_buffer(0, loc, k_scale)

        # Single request with 8 query tokens, 64 indexer heads.
        self.num_q_tokens = 8
        q_bf16 = torch.randn(
            self.num_q_tokens, N_HEADS_Q, HEAD_DIM,
            dtype=torch.bfloat16,
            device="npu",
        )
        self.query, self.query_scale = _quantize_npu_indexer_activation(
            q_bf16, self.hadamard, torch.float8_e4m3fn
        )
        self.weights = torch.randn(
            self.num_q_tokens, N_HEADS_Q, dtype=torch.float32, device="npu"
        )
        self.cu_seqlens_q = torch.tensor(
            [0, self.num_q_tokens], dtype=torch.int32, device="npu"
        )
        self.seqused_k = torch.tensor(
            [self.num_k_tokens], dtype=torch.int32, device="npu"
        )
        self.block_table = torch.tensor(
            [[0, 1, 2]], dtype=torch.int32, device="npu"
        )
        torch.npu.synchronize()

    def _metadata(self, cu_seqlens_q, seqused_k):
        return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            N_HEADS_Q,
            N_HEADS_K,
            HEAD_DIM,
            TOPK,
            QUANT_MODE_MXFP8,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            batch_size=int(seqused_k.numel()),
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=MASK_MODE_RIGHT_DOWN_CAUSAL,
            cmp_ratio=CMP_RATIO,
        )

    def test_metadata_rejects_int64_cu_seqlens_q(self):
        # Regression guard: aclnnQuantLightningIndexerV2Metadata rejects
        # int64 cu_seqlens_q (EZ0020); int32 is required.
        with self.assertRaises(RuntimeError) as ctx:
            self._metadata(
                self.cu_seqlens_q.to(torch.int64), self.seqused_k
            )
        self.assertIn("cu_seqlens_q", str(ctx.exception))

    def test_full_kernel(self):
        metadata = self._metadata(self.cu_seqlens_q, self.seqused_k)

        topk_indices, _ = torch.ops.cann_ops_transformer.quant_lightning_indexer(
            self.query,
            self.pool.get_index_k_buffer(0),
            self.weights,
            self.query_scale,
            self.pool.get_index_k_scale_buffer(0),
            TOPK,
            QUANT_MODE_MXFP8,
            cu_seqlens_q=self.cu_seqlens_q,
            seqused_k=self.seqused_k,
            block_table=self.block_table,
            metadata=metadata,
            max_seqlen_q=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=MASK_MODE_RIGHT_DOWN_CAUSAL,
            cmp_ratio=CMP_RATIO,
        )
        torch.npu.synchronize()

        # TND output: (q_t, k_n, topk) of int32 slot indices.
        self.assertEqual(topk_indices.shape, (self.num_q_tokens, N_HEADS_K, TOPK))
        self.assertEqual(topk_indices.dtype, torch.int32)
        # All indices either point inside the request's 192 k tokens or are
        # the invalid marker -1.
        self.assertGreaterEqual(int(topk_indices.min().item()), -1)
        self.assertLess(int(topk_indices.max().item()), self.num_k_tokens)


if __name__ == "__main__":
    unittest.main()
