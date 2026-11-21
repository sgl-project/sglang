"""Index storage, pruning routing and causal visible-prefix construction."""

import unittest
from unittest.mock import Mock

import torch

from sglang.srt.hardware_backend.npu.dsv4 import dsv41_int8_indexer as indexer
from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import decode_int8, encode_int8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestDSV41Int8Indexer(unittest.TestCase):
    def test_physical_index_sites_and_ranked_inputs(self):
        gen = torch.Generator().manual_seed(178)
        keys = torch.randn(6, 1, 128, generator=gen).bfloat16()
        storage = torch.zeros((4, 4, 1, 128), dtype=torch.int8)
        metadata = torch.zeros((4, 4, 1, 98), dtype=torch.uint8)
        loc = torch.tensor([4, 5, 10, 11, -1, 0])
        indexer.write_index_rows(storage, metadata, loc, keys)
        expected, scales = encode_int8(keys[:4], 128)
        torch.testing.assert_close(
            storage.view(-1, 1, 128)[loc[:4]], expected, atol=0, rtol=0
        )
        torch.testing.assert_close(
            metadata.view(-1, 1, 98)[loc[:4]], scales, atol=0, rtol=0
        )
        self.assertEqual(torch.count_nonzero(storage[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(metadata[0]).item(), 0)
        query = keys[:1]
        kw = dict(
            block_table=torch.tensor([[2, 1, -1]], dtype=torch.int32),
            seqused_k=torch.tensor([8]),
            cmp_ratio=2,
        )
        weights = torch.ones((1, 1))
        restored_q, bank, w, qs, ks = indexer.prepare_indexer_operands(
            query,
            storage,
            weights,
            metadata,
            torch.tensor([16]),
            torch.tensor([16]),
            kw,
            capture=False,
            capacity=16384,
            filtered=True,
        )
        self.assertIs(restored_q, query)
        self.assertIsNone(qs)
        self.assertIsNone(ks)
        torch.testing.assert_close(w, weights, rtol=0, atol=0)
        ref = decode_int8(storage[2], metadata[2], 128)
        torch.testing.assert_close(bank[kw["block_table"][0, 0]], ref, rtol=0, atol=0)

    def test_empty_query(self):
        q = torch.empty((0, 32, 128), dtype=torch.bfloat16)
        packed, scales = indexer.quantize_rows(q)
        ids, candidates = indexer.index_topk(
            packed, packed, None, scales, None, 512, candidate_topk_blocks=2048
        )
        self.assertEqual((ids.shape, candidates.shape), ((0, 512), (0, 1, 2048)))

    def test_ratio_two_causal_tails_are_per_query(self):
        n = 5
        q = torch.ones((n, 32, 128), dtype=torch.int8)
        k = torch.ones((4, 64, 1, 128), dtype=torch.int8)
        w = torch.ones(n, 32, dtype=torch.float16)
        table = torch.tensor([[3], [2]], dtype=torch.int32)
        kw = dict(
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            seqused_k=torch.tensor([1, 3], dtype=torch.int32),
            cmp_residual_k=torch.tensor([0, 1], dtype=torch.int32),
            block_table=table,
            cmp_ratio=2,
        )
        metadata = Mock(return_value=torch.empty(0))
        indices = torch.arange(n, dtype=torch.int32)[:, None, None].expand(n, 1, 8)
        calculate = Mock(return_value=(indices, None))
        result = indexer._ordinary_indices(
            (metadata, calculate), q, k, w, w, torch.ones(k.shape[:-1]), 8, kw
        )
        self.assertEqual(
            metadata.call_args.kwargs["seqused_k"].tolist(), [1, 1, 2, 3, 3]
        )
        self.assertEqual(
            calculate.call_args.kwargs["block_table"].view(-1).tolist(), [3, 3, 2, 2, 2]
        )
        self.assertTrue((result[0] == -1).all())
        self.assertEqual(result[1:, 0].tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
