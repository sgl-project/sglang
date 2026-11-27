import unittest
from unittest import mock

import torch

from sglang.srt.true_on_policy.dsa import flashmla_fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSparseFromBf16ByPasses(unittest.TestCase):
    """Prefill runs ``sparse_from_bf16`` in fixed row passes; each pass sees exactly its own rows."""

    def _run(self, rows, topk_length):
        calls = []

        def fake(q, kv, indices, scale, d_v, topk_length=None, out=None):
            calls.append((q[:, 0, 0].clone(), indices[:, 0].clone(), topk_length))
            value = q[..., :1].expand(-1, -1, d_v) + indices[:, :1, None]
            if out is None:
                return value.clone()
            return out.copy_(value)

        q = torch.arange(rows, dtype=torch.float32)[:, None, None].expand(rows, 16, 576)
        indices = torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, 2048) * 10
        with mock.patch.object(flashmla_fp8, "sparse_from_bf16", side_effect=fake):
            out = flashmla_fp8.sparse_from_bf16_by_passes(
                q, None, indices, 1.0, 4, topk_length
            )
        self.assertTrue(
            torch.equal(out, (q[..., :1] + indices[:, :1, None]).expand(-1, -1, 4))
        )
        return calls

    def test_rows_split_into_fixed_passes_in_order(self):
        passes = flashmla_fp8._ROWS_PER_PASS
        rows = 2 * passes + 3
        lengths = torch.arange(rows, dtype=torch.int32)
        calls = self._run(rows, lengths)
        self.assertEqual([len(q) for q, _, _ in calls], [passes, passes, 3])
        self.assertTrue(
            torch.equal(torch.cat([q for q, _, _ in calls]), torch.arange(rows).float())
        )
        self.assertTrue(
            torch.equal(
                torch.cat([ix for _, ix, _ in calls]),
                torch.arange(rows, dtype=torch.int32) * 10,
            )
        )
        self.assertTrue(torch.equal(torch.cat([t for _, _, t in calls]), lengths))

    def test_padded_batch_drops_lengths_for_every_pass(self):
        """Graph padding gives fewer lengths than rows; one call would drop them, so every pass must."""
        rows = flashmla_fp8._ROWS_PER_PASS + 1
        calls = self._run(rows, torch.ones(rows - 1, dtype=torch.int32))
        self.assertEqual([t for _, _, t in calls], [None, None])

    def test_small_batch_is_one_unchanged_call(self):
        lengths = torch.ones(7, dtype=torch.int32)
        calls = self._run(7, lengths)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][2], lengths)


if __name__ == "__main__":
    unittest.main()
