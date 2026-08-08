"""Unit tests for bounded sampling-mask reconstruction."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="base-c-test-cpu")

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.test_utils import CustomTestCase


def _make_info(top_ps, return_sampling_masks):
    batch_size = len(top_ps)
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.tensor(top_ps, dtype=torch.float32),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=True,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=10,
        return_sampling_masks=return_sampling_masks,
        sampling_mask_max_top_k=TOP_K_ALL,
        device="cpu",
    )


class TestSamplingMaskReconstruction(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.sampling_mask_max_tokens = 3

    def _reconstruct(self, probs, top_ps, return_sampling_masks, sampled_tokens):
        sampling_info = _make_info(top_ps, return_sampling_masks)
        sampling_mask_data = self.sampler._compute_sampling_mask_from_probs(
            probs, sampling_info
        )
        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_sampling_mask_to_output(
            output,
            sampling_info,
            torch.tensor(sampled_tokens),
            sampling_mask_data,
        )
        return sampling_mask_data, output

    def test_only_reconstructs_requested_rows(self):
        probs = torch.tensor(
            [
                [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1] * 10,
                [0.1] * 10,
            ]
        )

        sampling_mask_data, output = self._reconstruct(
            probs,
            top_ps=[0.8, 1.0, 1.0],
            return_sampling_masks=[True, False, False],
            sampled_tokens=[0, 1, 2],
        )

        request_indices, probs_idx = sampling_mask_data[:2]
        self.assertEqual(request_indices, [0])
        self.assertEqual(probs_idx.shape, (1, 4))
        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {0, 1})
        self.assertEqual(output.next_token_sampling_mask_idx[1:], [None, None])

    def test_support_at_cap_is_returned(self):
        _, output = self._reconstruct(
            torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            top_ps=[0.8],
            return_sampling_masks=[True],
            sampled_tokens=[0],
        )

        self.assertEqual(len(output.next_token_sampling_mask_idx[0]), 3)

    def test_support_above_cap_is_not_materialized(self):
        _, output = self._reconstruct(
            torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            top_ps=[0.95],
            return_sampling_masks=[True],
            sampled_tokens=[0],
        )

        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertIsNone(output.next_token_sampling_logprobs[0])

    def test_overflow_is_per_request(self):
        probs = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        _, output = self._reconstruct(
            probs,
            top_ps=[0.95, 0.8],
            return_sampling_masks=[True, True],
            sampled_tokens=[0, 0],
        )

        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertEqual(set(output.next_token_sampling_mask_idx[1]), {0, 1})

    def test_sampled_token_counts_toward_cap(self):
        _, output = self._reconstruct(
            torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            top_ps=[0.8],
            return_sampling_masks=[True],
            sampled_tokens=[3],
        )

        self.assertIsNone(output.next_token_sampling_mask_idx[0])

    def test_float32_top_p_rounding_cannot_bypass_cap(self):
        top_p = torch.tensor(0.99999999, dtype=torch.float32).item()
        self.assertEqual(top_p, 1.0)

        _, output = self._reconstruct(
            torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            top_ps=[top_p],
            return_sampling_masks=[True],
            sampled_tokens=[0],
        )

        self.assertIsNone(output.next_token_sampling_mask_idx[0])


if __name__ == "__main__":
    unittest.main()
