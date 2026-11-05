"""CPU-only tests for the DSA NPU indexer head padding helper and call site.

The vendor quant indexer tiling accepts only num_heads_q == 64, so the call
site widens a 32-head indexer and pins the vendor ABI it then calls with.
torch.ops.custom is stubbed; no NPU operator runs here.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _PADDED_HEAD_QUERY_SCALE,
    VENDOR_QUANT_INDEXER_NUM_HEADS,
    DSANPUIndexerMixin,
    _pad_indexer_heads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TOKENS = 3
HEAD_DIM = 128
INDEX_TOPK = 7
NUM_SEQS = 2
NUM_PAGES = 4
PAGE_SIZE = 16


def _inputs(num_heads: int):
    query = torch.randint(-127, 127, (TOKENS, num_heads, HEAD_DIM), dtype=torch.int8)
    weights = torch.randn(TOKENS, num_heads, dtype=torch.float16)
    query_scale = torch.rand(TOKENS, num_heads, dtype=torch.float16) + 0.5
    return query, weights, query_scale


class TestPadIndexerHeads(CustomTestCase):
    def test_real_heads_pass_through_unchanged(self):
        query, weights, query_scale = _inputs(32)
        out_q, out_w, out_s = _pad_indexer_heads(
            query=query,
            weights=weights,
            query_scale=query_scale,
            num_heads=32,
            padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
        )
        self.assertTrue(torch.equal(out_q[:, :32], query))
        self.assertTrue(torch.equal(out_w[:, :32], weights))
        self.assertTrue(torch.equal(out_s[:, :32], query_scale))

    def test_padded_heads_are_suppressed_with_a_finite_scale(self):
        query, weights, query_scale = _inputs(32)
        out_q, out_w, out_s = _pad_indexer_heads(
            query=query,
            weights=weights,
            query_scale=query_scale,
            num_heads=32,
            padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
        )
        # Zero query and zero weight are what the 910B probe measured;
        # the padded scale is then free, so keep it finite and non-zero.
        self.assertFalse(out_q[:, 32:].any())
        self.assertFalse(out_w[:, 32:].any())
        self.assertTrue(torch.all(out_s[:, 32:] == _PADDED_HEAD_QUERY_SCALE))
        self.assertNotEqual(_PADDED_HEAD_QUERY_SCALE, 0.0)
        self.assertTrue(torch.isfinite(out_s).all())

    def test_widened_shapes_and_dtypes_match_the_vendor_contract(self):
        query, weights, query_scale = _inputs(32)
        out_q, out_w, out_s = _pad_indexer_heads(
            query=query,
            weights=weights,
            query_scale=query_scale,
            num_heads=32,
            padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
        )
        self.assertEqual(tuple(out_q.shape), (TOKENS, 64, HEAD_DIM))
        self.assertEqual(tuple(out_w.shape), (TOKENS, 64))
        self.assertEqual(tuple(out_s.shape), (TOKENS, 64))
        self.assertEqual(out_q.dtype, torch.int8)
        self.assertEqual(out_w.dtype, torch.float16)
        self.assertEqual(out_s.dtype, torch.float16)

    def test_a_full_width_indexer_is_not_copied(self):
        query, weights, query_scale = _inputs(64)
        out_q, out_w, out_s = _pad_indexer_heads(
            query=query,
            weights=weights,
            query_scale=query_scale,
            num_heads=64,
            padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
        )
        self.assertIs(out_q, query)
        self.assertIs(out_w, weights)
        self.assertIs(out_s, query_scale)

    def test_too_many_heads_is_rejected_instead_of_truncated(self):
        query, weights, query_scale = _inputs(128)
        with self.assertRaises(ValueError):
            _pad_indexer_heads(
                query=query,
                weights=weights,
                query_scale=query_scale,
                num_heads=128,
                padded_num_heads=VENDOR_QUANT_INDEXER_NUM_HEADS,
            )


def _drive_vendor_indexer():
    indexer = SimpleNamespace(n_heads=32, head_dim=HEAD_DIM, index_topk=INDEX_TOPK)
    handle = torch.zeros(16, dtype=torch.int32)
    with patch("torch.ops.custom", MagicMock(), create=True) as custom_ops:
        custom_ops.npu_quant_lightning_indexer.return_value = (
            torch.zeros(TOKENS, INDEX_TOPK, dtype=torch.int32),
            torch.zeros(TOKENS, INDEX_TOPK, dtype=torch.float32),
        )
        result = DSANPUIndexerMixin._npu_vendor_quant_indexer(
            indexer,
            query=torch.zeros(TOKENS, 32, HEAD_DIM, dtype=torch.int8),
            # The pool hands int8 an fp16 scale today;
            # feed fp32 so the vendor ABI's cast stays observable.
            query_scale=torch.ones(TOKENS, 32, dtype=torch.float32),
            key=torch.zeros(NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM, dtype=torch.int8),
            key_scale=torch.ones(NUM_PAGES, PAGE_SIZE, 1, dtype=torch.float32),
            weights=torch.zeros(TOKENS, 32, dtype=torch.bfloat16),
            block_table=torch.zeros(NUM_SEQS, NUM_PAGES, dtype=torch.int32),
            metadata=handle,
            cu_q_lens=torch.tensor([1, TOKENS], dtype=torch.int32),
            kv_lens=torch.tensor([10, 20], dtype=torch.int32),
        )
        return custom_ops.npu_quant_lightning_indexer.call_args.kwargs, handle, result


class TestVendorQuantIndexerCall(CustomTestCase):
    """Pins the vendor operator ABI the 910B probe established: cmp_ratio=1
    for uncompressed DSA keys, FP16 scales / weights, and a 64-head query.
    """

    def test_op_receives_the_vendor_abi(self):
        op_kwargs, handle, result = _drive_vendor_indexer()
        self.assertEqual(op_kwargs["cmp_ratio"], 1)
        self.assertEqual(op_kwargs["sparse_mode"], 3)
        self.assertEqual(op_kwargs["sparse_count"], INDEX_TOPK)
        self.assertEqual(op_kwargs["layout_query"], "TND")
        self.assertEqual(op_kwargs["layout_key"], "PA_BSND")
        self.assertEqual(op_kwargs["query_quant_mode"], 0)
        self.assertEqual(op_kwargs["key_quant_mode"], 0)
        self.assertIs(op_kwargs["metadata"], handle)
        self.assertEqual(op_kwargs["query"].dtype, torch.int8)
        self.assertEqual(op_kwargs["key"].dtype, torch.int8)
        for name in ("weights", "query_dequant_scale", "key_dequant_scale"):
            self.assertEqual(op_kwargs[name].dtype, torch.float16, name)
        # The per-layer pool scale is [pages, page_size, 1]; no reshape here.
        self.assertEqual(
            tuple(op_kwargs["key_dequant_scale"].shape), (NUM_PAGES, PAGE_SIZE, 1)
        )
        self.assertEqual(
            tuple(op_kwargs["query"].shape),
            (TOKENS, VENDOR_QUANT_INDEXER_NUM_HEADS, HEAD_DIM),
        )
        self.assertEqual(tuple(result.shape), (TOKENS, INDEX_TOPK))


if __name__ == "__main__":
    unittest.main()
