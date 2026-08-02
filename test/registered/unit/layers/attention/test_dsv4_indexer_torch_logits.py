import unittest

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEAD_DIM = 128
BLOCK_SIZE = 64


def _build_inputs(seq_lens, num_heads=4, seed=0):
    """Inputs in the layout the paged indexer call site produces."""
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch_size * max_pages + 1

    torch.manual_seed(seed)
    q_fp8 = torch.randn(batch_size, 1, num_heads, HEAD_DIM).to(FP8_DTYPE)
    kvcache_fp8 = torch.randn(num_pages, BLOCK_SIZE, 1, HEAD_DIM + 4).to(FP8_DTYPE)
    weight = torch.randn(batch_size, num_heads)
    page_table = torch.zeros(batch_size, max_pages, dtype=torch.int64)
    for i in range(batch_size):
        page_table[i] = torch.arange(1 + i * max_pages, 1 + (i + 1) * max_pages)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return q_fp8, kvcache_fp8, weight, seq_lens_t, page_table, max_seq_len


class TestDsv4IndexerTorchLogits(unittest.TestCase):
    def test_reference_and_dispatched_variants_agree(self):
        """Both torch variants produce the same logits, tail included."""
        for seq_lens in ([70, 100], [64, 128], [1, 191], [200]):
            with self.subTest(seq_lens=seq_lens):
                q, kv, w, sl, pt, msl = _build_inputs(seq_lens)
                ref = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, msl, False)
                out = fp8_paged_mqa_logits_torch_sm120(
                    q, kv, w, sl, pt, None, msl, False
                )
                self.assertEqual(ref.shape, out.shape)
                for i, length in enumerate(seq_lens):
                    self.assertTrue(
                        torch.equal(ref[i, :length], out[i, :length]),
                        f"valid region differs for row {i}",
                    )

    def test_both_variants_fill_the_tail_with_neg_inf(self):
        """Positions at or beyond seq_lens are -inf in both variants."""
        seq_lens = [70, 100]
        q, kv, w, sl, pt, msl = _build_inputs(seq_lens)
        for fn in (fp8_paged_mqa_logits_torch, fp8_paged_mqa_logits_torch_sm120):
            with self.subTest(fn=fn.__name__):
                out = fn(q, kv, w, sl, pt, None, msl, False)
                positions = torch.arange(msl)
                invalid = positions.unsqueeze(0) >= sl.unsqueeze(1)
                self.assertTrue(
                    torch.all(torch.isinf(out[invalid]) & (out[invalid] < 0)),
                    f"{fn.__name__} left non -inf values past seq_lens",
                )

    def test_padded_tail_beyond_page_boundary_is_neg_inf(self):
        """max_seq_len shorter than the padded page span still masks the gap."""
        # 191 pads to 3 pages = 192 positions; row 0 ends at 10.
        seq_lens = [10, 191]
        q, kv, w, sl, pt, msl = _build_inputs(seq_lens)
        ref = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, msl, False)
        self.assertTrue(torch.all(torch.isneginf(ref[0, 10:])))
        self.assertTrue(torch.all(torch.isneginf(ref[1, 191:])))

    def test_only_the_dispatched_variant_accepts_the_paged_seq_lens_shape(self):
        """The paged call site passes seq_lens with a trailing dim of 1."""
        seq_lens = [70, 100]
        q, kv, w, sl, pt, msl = _build_inputs(seq_lens)
        sl_2d = sl.unsqueeze(-1)

        out = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl_2d, pt, None, msl, False)
        self.assertEqual(out.shape, (len(seq_lens), msl))

        with self.assertRaises(AssertionError):
            fp8_paged_mqa_logits_torch(q, kv, w, sl_2d, pt, None, msl, False)


if __name__ == "__main__":
    unittest.main()
