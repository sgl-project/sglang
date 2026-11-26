import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.speculative.spec_sampling_mask import build_ngram_sampling_mask


class TestNgramSamplingMask(CustomTestCase):
    def test_selects_policy_rows_on_the_committed_tree_path(self):
        # The first request skips a sibling. The second did not request masks.
        # The third accepts only its root and requests selected-token logprobs.
        probs = torch.ones(3, 3, 4) / 4
        probs[0, 0] = torch.tensor([0.0, 0.6, 0.4, 0.0])
        probs[0, 2] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        probs[2, 0] = torch.tensor([0.75, 0.0, 0.25, 0.0])
        predict = torch.tensor([2, 0, 3, 0, 0, 0, 0, 0, 0])
        out = build_ngram_sampling_mask(
            SimpleNamespace(
                sampling_mask_batch_indices=torch.tensor([0, 2]),
                sampling_support_logprobs_capture_indices=torch.tensor([0]),
            ),
            probs,
            predict,
            torch.tensor([[0, 2, -1], [3, 4, 5], [6, -1, -1]]),
            torch.tensor([2, 3, 1]),
            max_tokens=3,
        )
        self.assertEqual(out.num_accept_tokens.tolist(), [2, 1])
        self.assertTrue(out.token_ids.is_contiguous())
        self.assertTrue(out.support_logprobs.is_contiguous())
        self.assertEqual(out.lengths[0, :2].tolist(), [2, 1])
        self.assertEqual(out.token_ids[0, 0, :2].tolist(), [1, 2])
        self.assertEqual(out.token_ids[0, 1, 0].item(), 3)
        torch.testing.assert_close(
            out.support_logprobs[0, 0, :2].exp(), torch.tensor([0.6, 0.4])
        )
        torch.testing.assert_close(
            out.selected_logprobs[0, :2].exp(), torch.tensor([0.4, 1.0])
        )
        self.assertAlmostEqual(out.selected_logprobs[1, 0].exp().item(), 0.75)

    def test_reports_overflow_for_tied_support(self):
        out = build_ngram_sampling_mask(
            SimpleNamespace(
                sampling_mask_batch_indices=torch.tensor([0]),
                sampling_support_logprobs_capture_indices=None,
            ),
            torch.full((1, 1, 4), 0.25),
            torch.tensor([3]),
            torch.tensor([[0]]),
            torch.tensor([1]),
            max_tokens=2,
        )
        self.assertEqual(out.statuses.item(), SamplingMaskStatus.OVERFLOW)
        self.assertEqual(out.token_ids.shape[-1], 2)


if __name__ == "__main__":
    unittest.main()
