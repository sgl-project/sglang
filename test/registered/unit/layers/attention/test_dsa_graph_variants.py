"""Tests for the DSA decode/verify CUDA-graph variant selection."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.graph_variants import (
    DSA_DENSE,
    DSA_SPARSE,
    DSA_SPARSE_SHARED_PAGES,
    DsaGraphVariants,
    dsa_max_kv_len,
    dsa_share_logits_pages,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(seq_lens):
    return SimpleNamespace(
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens=None,
        batch_size=len(seq_lens),
    )


class TestDsaGraphVariants(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(DsaGraphVariants(2048).capture_labels, (DSA_DENSE, DSA_SPARSE))
        self.assertEqual(
            DsaGraphVariants(2048, verify_rows_per_req=6).capture_labels,
            (DSA_DENSE, DSA_SPARSE, DSA_SPARSE_SHARED_PAGES),
        )

    def test_without_verify_rows_never_shares_pages(self):
        v = DsaGraphVariants(2048)
        self.assertEqual(v.select(_batch([1000])), DSA_DENSE)
        self.assertEqual(v.select(_batch([500000])), DSA_SPARSE)

    def test_verify_selection(self):
        v = DsaGraphVariants(2048, verify_rows_per_req=6)
        cases = [
            ([2048], DSA_DENSE),
            ([2049], DSA_SPARSE),
            ([98303], DSA_SPARSE),
            ([98304], DSA_SPARSE_SHARED_PAGES),
            ([5000, 98304], DSA_SPARSE_SHARED_PAGES),
            ([98303, 100, 100], DSA_SPARSE),
            ([98304, 100, 100], DSA_SPARSE_SHARED_PAGES),
            ([500000] * 4, DSA_SPARSE),  # 24 rows keep the per-row kernel
        ]
        for lens, want in cases:
            with self.subTest(lens=lens):
                self.assertEqual(v.select(_batch(lens)), want)

    def test_lengths_fall_back_to_device_tensor(self):
        """Verify batches can carry an empty host mirror; the device lengths decide."""
        v = DsaGraphVariants(2048, verify_rows_per_req=6)
        batch = SimpleNamespace(
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            seq_lens=torch.tensor([120000]),
            batch_size=1,
        )
        self.assertEqual(v.select(batch), DSA_SPARSE_SHARED_PAGES)
        self.assertEqual(dsa_max_kv_len(batch), 120000)

    def test_share_logits_pages(self):
        self.assertTrue(dsa_share_logits_pages(6, 98304))
        self.assertFalse(dsa_share_logits_pages(6, 98303))
        self.assertTrue(dsa_share_logits_pages(18, 98304))
        self.assertFalse(dsa_share_logits_pages(19, 10**6))


if __name__ == "__main__":
    unittest.main()
