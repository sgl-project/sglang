"""Pad for the aiter CK draft-extend fallback: mha_batch_prefill over-reads 128
page ids past the last token; draft extend must pad that tail or the kernel faults."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm attention backend")
class TestAiterDraftExtendMetadata(CustomTestCase):
    def test_pad_repeats_a_live_index(self):
        from sglang.srt.layers.attention import aiter_backend as ab

        idx = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
        padded = ab._pad_mha_prefill_kv_indices(idx)
        self.assertEqual(padded.shape[0], idx.shape[0] + ab._MHA_PREFILL_KV_INDEX_PAD)
        self.assertTrue(torch.equal(padded[: idx.shape[0]], idx))
        self.assertTrue(torch.all(padded[idx.shape[0] :] == idx[0]))
        self.assertEqual(ab._pad_mha_prefill_kv_indices(idx[:0]).numel(), 0)
