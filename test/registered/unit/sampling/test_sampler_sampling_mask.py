"""Focused tests for faithful sampling-mask capture."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import (
    Sampler,
    _SamplingMaskCapture,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


def _pytorch_support_oracle(probs, top_k, top_p, min_p=0.0):
    sorted_probs, sorted_ids = probs.sort(descending=True)
    mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
    positions = torch.arange(probs.numel(), device=probs.device)
    keep = (positions < top_k) & (mass_before <= top_p)
    keep &= sorted_probs >= sorted_probs[0] * min_p
    return set(sorted_ids[keep].tolist())


class TestSamplingMaskCapture(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)

    def _attach(self, weights, sampled_tokens, token_ids=None, selected_weight=None):
        output = LogitsProcessorOutput(next_token_logits=None)
        sampling_info = SimpleNamespace(return_sampling_masks=[True] * weights.shape[0])
        self.sampler._attach_sampling_mask_to_output(
            output,
            sampling_info,
            torch.tensor(sampled_tokens, device=weights.device),
            _SamplingMaskCapture(weights, token_ids, selected_weight),
        )
        return output

    def test_strict_positive_support_and_exact_selected_logprob(self):
        weights = torch.tensor([[0.4, 0.3, 0.2, 0.0]], device="cuda")
        output = self._attach(weights, sampled_tokens=[1])

        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {0, 1, 2})
        self.assertAlmostEqual(
            output.next_token_sampling_logprobs[0], math.log(0.3 / 0.9), places=6
        )

    def test_token_permutation_maps_support_and_selected_weight(self):
        weights = torch.tensor([[0.7, 0.3, 0.0]], device="cuda")
        token_ids = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda")
        output = self._attach(
            weights,
            sampled_tokens=[1],
            token_ids=token_ids,
            selected_weight=torch.tensor([0.3], device="cuda"),
        )

        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {1, 3})
        self.assertAlmostEqual(
            output.next_token_sampling_logprobs[0], math.log(0.3), places=6
        )

    def test_sampled_token_outside_support_is_invariant_failure(self):
        weights = torch.tensor([[0.6, 0.4, 0.0]], device="cuda")
        with self.assertRaisesRegex(RuntimeError, "outside captured positive"):
            self._attach(weights, sampled_tokens=[2])

    def test_only_requested_rows_are_materialized_and_validated(self):
        output = LogitsProcessorOutput(next_token_logits=None)
        sampling_info = SimpleNamespace(return_sampling_masks=[True, False])
        self.sampler._attach_sampling_mask_to_output(
            output,
            sampling_info,
            torch.tensor([0, 2], device="cuda"),
            _SamplingMaskCapture(
                weights=torch.tensor([[0.6, 0.4, 0.0], [0.0, 0.0, 0.0]], device="cuda"),
                token_ids=None,
                selected_weight=None,
            ),
        )

        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {0, 1})
        self.assertIsNone(output.next_token_sampling_mask_idx[1])
        self.assertIsNone(output.next_token_sampling_logprobs[1])

    def test_greedy_support_is_sampled_token_singleton(self):
        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_greedy_sampling_mask_to_output(
            output,
            SimpleNamespace(return_sampling_masks=[True, False, True]),
            torch.tensor([4, 5, 6], device="cuda"),
        )

        self.assertEqual(output.next_token_sampling_mask_idx, [[4], None, [6]])
        self.assertEqual(output.next_token_sampling_logprobs, [0.0, None, 0.0])

    def test_pytorch_capture_matches_independent_oracle(self):
        for top_k, top_p, min_p in ((4, 0.72, 0.0), (4, 1.0, 0.4)):
            with self.subTest(top_k=top_k, top_p=top_p, min_p=min_p):
                probs = torch.tensor([[0.40, 0.25, 0.15, 0.12, 0.08]], device="cuda")
                sampled, filtered, token_ids, selected_weight = (
                    top_k_top_p_min_p_sampling_from_probs_torch(
                        probs,
                        top_ks=torch.tensor([top_k], device="cuda"),
                        top_ps=torch.tensor([top_p], device="cuda"),
                        min_ps=torch.tensor([min_p], device="cuda"),
                        need_min_p_sampling=min_p > 0,
                        sampling_seed=None,
                        positions=torch.tensor([0], device="cuda"),
                        return_filtered_probs=True,
                    )
                )

                actual_support = set(token_ids[filtered > 0].tolist())
                expected_support = _pytorch_support_oracle(
                    probs[0], top_k, top_p, min_p
                )
                self.assertEqual(actual_support, expected_support)
                self.assertIn(int(sampled[0]), actual_support)
                selected_position = (token_ids[0] == sampled[0]).nonzero()[0]
                self.assertEqual(
                    float(selected_weight[0]), float(filtered[0, selected_position])
                )

    def test_seeded_pytorch_capture_does_not_change_sample(self):
        probs = torch.tensor([[0.40, 0.25, 0.15, 0.12, 0.08]], device="cuda")
        args = dict(
            probs=probs,
            top_ks=torch.tensor([4], device="cuda"),
            top_ps=torch.tensor([0.9], device="cuda"),
            min_ps=torch.tensor([0.0], device="cuda"),
            need_min_p_sampling=False,
            sampling_seed=torch.tensor([1234], device="cuda"),
            positions=torch.tensor([7], device="cuda"),
        )

        without_capture = top_k_top_p_min_p_sampling_from_probs_torch(**args)
        with_capture, filtered, token_ids, selected_weight = (
            top_k_top_p_min_p_sampling_from_probs_torch(
                **args, return_filtered_probs=True
            )
        )

        self.assertTrue(torch.equal(without_capture, with_capture))
        self.assertGreater(int(torch.count_nonzero(filtered)), 0)
        self.assertIn(int(with_capture[0]), token_ids[filtered > 0].tolist())
        self.assertGreater(float(selected_weight[0]), 0)

    def _flashinfer_sampling_info(self, *, top_k, top_p, min_p=0.0):
        return SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=top_k > 0,
            need_top_p_sampling=top_p < 1.0,
            need_min_p_sampling=min_p > 0,
            top_ks=torch.tensor([top_k], device="cuda"),
            top_ps=torch.tensor([top_p], device="cuda"),
            min_ps=torch.tensor([min_p], device="cuda"),
        )

    def _sample_flashinfer(self, probs, sampling_info):
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="flashinfer")
            ),
        ):
            return self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.tensor([0], device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

    def test_flashinfer_joint_capture_matches_independent_oracle(self):
        probs = torch.tensor([[0.40, 0.30, 0.20, 0.10]], device="cuda")
        sampled, capture = self._sample_flashinfer(
            probs, self._flashinfer_sampling_info(top_k=3, top_p=0.75)
        )

        actual_support = set((capture.weights[0] > 0).nonzero().view(-1).tolist())
        self.assertEqual(actual_support, {0, 1, 2})
        self.assertIn(int(sampled[0]), actual_support)
        output = self._attach(
            capture.weights, sampled.tolist(), selected_weight=capture.selected_weight
        )
        expected = math.log(
            float(capture.selected_weight[0]) / float(capture.weights.sum())
        )
        self.assertAlmostEqual(
            output.next_token_sampling_logprobs[0], expected, places=6
        )

    def test_flashinfer_min_p_capture_matches_sequential_oracle(self):
        probs = torch.tensor([[0.50, 0.25, 0.15, 0.10]], device="cuda")
        sampled, capture = self._sample_flashinfer(
            probs, self._flashinfer_sampling_info(top_k=4, top_p=1.0, min_p=0.4)
        )

        actual_support = set((capture.weights[0] > 0).nonzero().view(-1).tolist())
        self.assertEqual(actual_support, {0, 1})
        self.assertIn(int(sampled[0]), actual_support)

    def test_flashinfer_top_k_cutoff_ties_are_captured_faithfully(self):
        probs = torch.tensor([[0.40, 0.20, 0.20, 0.20]], device="cuda")
        sampled, capture = self._sample_flashinfer(
            probs, self._flashinfer_sampling_info(top_k=2, top_p=1.0)
        )

        support = (capture.weights[0] > 0).nonzero().view(-1).tolist()
        self.assertEqual(set(support), {0, 1, 2, 3})
        self.assertGreater(len(support), 2)
        self.assertIn(int(sampled[0]), support)

    def test_capture_off_returns_no_data(self):
        probs = torch.tensor([[0.40, 0.30, 0.20, 0.10]], device="cuda")
        sampling_info = SimpleNamespace(
            sampling_seed=torch.tensor([99], device="cuda"),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.tensor([3], device="cuda"),
            top_ps=torch.tensor([0.9], device="cuda"),
            min_ps=torch.tensor([0.0], device="cuda"),
        )
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="pytorch")
            ),
        ):
            without_mask, no_capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.tensor([3], device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=False,
            )
            with_mask, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.tensor([3], device="cuda"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNone(no_capture)
        self.assertIsNotNone(capture)
        self.assertTrue(torch.equal(without_mask, with_mask))


if __name__ == "__main__":
    import unittest

    unittest.main()
