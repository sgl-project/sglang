"""Eager NEXTN draft-extend metadata for the aiter CK prefill fallback.

The CK kernel reads 128 page ids past the last token, and it reads whichever
qo_indptr it is given. Draft extend used to hand it the prompt-length indptr
left in the shared buffer. These checks cover the pad, the length, and the
decision to take unified_attention instead.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


class TestAiterDraftExtendMetadata(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import aiter_backend as ab

        cls.ab = ab

    def test_pad_repeats_a_live_index(self):
        idx = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        padded = self.ab._pad_mha_prefill_kv_indices(idx)
        self.assertEqual(
            padded.shape[0], idx.shape[0] + self.ab._MHA_PREFILL_KV_INDEX_PAD
        )
        self.assertTrue(torch.equal(padded[: idx.shape[0]], idx))
        self.assertTrue(torch.all(padded[idx.shape[0] :] == idx[0]))

    def test_pad_empty_stays_empty(self):
        idx = torch.empty(0, dtype=torch.int32)
        self.assertEqual(self.ab._pad_mha_prefill_kv_indices(idx).numel(), 0)

    def test_max_kv_len_prefers_cpu_mirror(self):
        batch = type("B", (), {})()
        batch.seq_lens_cpu = [4, 11, 2]
        batch.seq_lens = torch.tensor([100, 100, 100])
        self.assertEqual(self.ab._batch_max_kv_len(batch, 3), 11)
        self.assertEqual(self.ab._batch_max_kv_len(batch, 0), 0)

    def test_max_kv_len_falls_back_to_device_lens(self):
        batch = type("B", (), {})()
        batch.seq_lens_cpu = None
        batch.seq_lens = torch.tensor([3, 9, 1])
        self.assertEqual(self.ab._batch_max_kv_len(batch, 2), 9)

    def test_unified_draft_extend_is_default_for_linear_topk(self):
        uses = self.ab._draft_extend_uses_unified
        self.assertTrue(uses(False, 1, True))
        self.assertTrue(uses(False, None, True))
        self.assertFalse(uses(False, 1, False))
        self.assertFalse(uses(True, 1, True))
        self.assertFalse(uses(False, 4, True))


if __name__ == "__main__":
    unittest.main()
