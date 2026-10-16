"""CPU checks for block-pair geometry and the per-layer cache dependency."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_block_pair_backend import (
    FlashInferBlockPairBackend,
    block_slice,
    split_pair_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch(bs, block, prefixes):
    rows = bs * 2 * block
    prefix = torch.tensor(prefixes, dtype=torch.int32)
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_dllm_extend=lambda: True),
        batch_size=bs,
        input_ids=torch.arange(rows),
        positions=torch.arange(rows),
        out_cache_loc=torch.arange(rows) + 100,
        seq_lens=prefix + 2 * block,
        seq_lens_cpu=prefix + 2 * block,
        seq_lens_sum=sum(prefixes) + rows,
        extend_prefix_lens=prefix,
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens=torch.full((bs,), 2 * block),
        extend_seq_lens_cpu=[2 * block] * bs,
        spec_info=None,
        encoder_lens=None,
    )


class TestFlashInferBlockPair(CustomTestCase):
    def test_heterogeneous_prefixes_and_request_order(self):
        original = batch(3, 4, [0, 8, 20])
        first, second = split_pair_batch(original, 4)
        self.assertEqual(
            first.input_ids.tolist(), [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
        )
        self.assertEqual(
            second.input_ids.tolist(), [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]
        )
        self.assertEqual(first.seq_lens.tolist(), [4, 12, 24])
        self.assertEqual(second.extend_prefix_lens_cpu, [4, 12, 24])
        self.assertEqual(first.seq_lens_sum, 40)
        self.assertEqual(second.seq_lens_sum, 52)
        self.assertEqual(first.extend_start_loc.tolist(), [0, 4, 8])
        self.assertEqual(original.input_ids.tolist(), list(range(24)))

    def test_reject_invalid_layout(self):
        for prefixes, lengths in [([1], [8]), ([0], [4])]:
            original = batch(1, 4, prefixes)
            original.extend_seq_lens_cpu = lengths
            with self.assertRaises(ValueError):
                split_pair_batch(original, 4)
        with self.assertRaises(ValueError):
            block_slice(torch.zeros(7, 2), 1, 4, 0)

    def test_refresh_is_visible_to_next_block_and_rows_reassemble(self):
        original = batch(2, 4, [4, 12])
        parts = split_pair_batch(original, 4)
        recorded = []
        cache = {}

        class Child:
            def __init__(self, index):
                self.index = index

            def forward_extend(self, q, k, v, layer, part, save_kv_cache):
                recorded.append((self.index, part.out_cache_loc.tolist()))
                if self.index == 0:
                    cache["refreshed"] = k.clone()
                else:
                    torch.testing.assert_close(
                        cache["refreshed"], block_slice(values, 2, 4, 0)
                    )
                return q

        backend = FlashInferBlockPairBackend.__new__(FlashInferBlockPairBackend)
        backend.block_size, backend.parts = 4, parts
        backend.attn_backend_list = [Child(0), Child(1)]
        values = torch.arange(16 * 6).reshape(16, 2, 3).float()
        output = backend.forward_extend(values, values, values, None, original)
        torch.testing.assert_close(output, values.reshape(16, 6))
        self.assertEqual([item[0] for item in recorded], [0, 1])
        original.out_cache_loc += 1000
        backend.forward_extend(values, values, values, None, original)
        self.assertEqual(
            recorded[2][1], [1100, 1101, 1102, 1103, 1108, 1109, 1110, 1111]
        )

    def test_no_cross_request_or_future_block_leakage(self):
        # Compare the two-call attention dependency with a block-causal reference.
        torch.manual_seed(7)
        for bs in (1, 3):
            for block in (2, 4, 16, 32):
                q, k, v = [torch.randn(bs, 2 * block, 8) for _ in range(3)]
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=(
                        torch.arange(2 * block)[:, None] // block
                        >= torch.arange(2 * block)[None, :] // block
                    ),
                )
                first = torch.nn.functional.scaled_dot_product_attention(
                    q[:, :block], k[:, :block], v[:, :block]
                )
                second = torch.nn.functional.scaled_dot_product_attention(
                    q[:, block:], k, v
                )
                torch.testing.assert_close(torch.cat([first, second], 1), expected)


if __name__ == "__main__":
    unittest.main()
