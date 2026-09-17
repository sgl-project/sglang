"""
python3 -m unittest test_sampling_mask_xpu.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestXpuSamplingMaskCapture(CustomTestCase):
    """Sampling-mask capture on the intel_xpu backend.

    The backend draws its token from the fused SYCL joint kernel but rebuilds the
    captured support from the separate top_k/top_p renorm kernels, which are
    written independently of it. If the two disagree on a cutoff or a tie,
    selected_weight comes back 0 for a token that really was sampled -- a
    silently wrong number rather than an error.

    These cases mirror TestSamplingMaskCapture in
    test/registered/sampling/test_sampling_mask.py, which covers the same
    invariants against the CUDA and ROCm kernels; the kernels behind them are a
    separate implementation on XPU.
    """

    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)

    def test_hard_exclusion_replay_in_mixed_batch(self):
        for backend in ["pytorch", "intel_xpu"]:
            with self.subTest(backend=backend):
                logits = (
                    torch.tensor([[0.3, 0.2, 0.5, 0.15, 0.1]], device="xpu")
                    .log()
                    .repeat(2, 1)
                )
                original = logits.clone()
                info = SamplingBatchInfo(
                    temperatures=torch.ones(2, 1, device="xpu"),
                    top_ps=torch.full((2,), 0.9, device="xpu"),
                    top_ks=torch.full((2,), 3, dtype=torch.int32, device="xpu"),
                    min_ps=torch.zeros(2, device="xpu"),
                    is_all_greedy=False,
                    is_any_greedy=False,
                    need_top_p_sampling=True,
                    need_top_k_sampling=True,
                    need_min_p_sampling=False,
                    vocab_size=5,
                    has_custom_logit_processor=True,
                    custom_params=[{"token_ids": [2]}, None],
                    custom_logit_processor={
                        0: (
                            DisallowedTokensLogitsProcessor(),
                            torch.tensor([True, False], device="xpu"),
                        )
                    },
                    return_sampling_masks=[True, True],
                )
                logits = self.sampler._preprocess_logits(logits, info)
                with patch(
                    "sglang.srt.layers.sampler.get_exec",
                    return_value=SimpleNamespace(
                        kernel=SimpleNamespace(sampling_backend=backend)
                    ),
                ):
                    sampled, capture = self.sampler._sample_from_probs(
                        logits.softmax(-1),
                        info,
                        positions=torch.zeros(2, dtype=torch.int64, device="xpu"),
                        simple_sampling_case=False,
                        return_sampling_mask=True,
                    )
                output = LogitsProcessorOutput(next_token_logits=None)
                self.sampler._attach_sampling_mask_to_output(
                    output, info, sampled, capture
                )
                support = output.next_token_sampling_mask_idx[0]
                self.assertEqual(set(support), {0, 1, 3})
                self.assertIn(int(sampled[0]), support)
                expected = original[0, sampled[0]] - original[0, support].logsumexp(0)
                self.assertAlmostEqual(
                    output.next_token_sampling_logprobs[0], expected.item(), places=5
                )
                self.assertIn(2, output.next_token_sampling_mask_idx[1])

    def test_intel_xpu_joint_cutoff_ties_match_capture(self):
        batch_size = 256
        top_k = 2
        top_p = 0.45
        base_probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="xpu")
        probs = base_probs.repeat(batch_size, 1)

        # Derive the threshold-based joint support independently. Both filters
        # cut at 0.2, so the tied entries must survive even though this yields
        # more support entries than top_k.
        sorted_probs = base_probs[0].sort(descending=True).values
        top_k_cutoff = sorted_probs[top_k - 1]
        mass_before = sorted_probs.cumsum(dim=-1) - sorted_probs
        top_p_cutoff = sorted_probs[mass_before <= top_p][-1]
        expected_support = (base_probs[0] >= top_k_cutoff) & (
            base_probs[0] >= top_p_cutoff
        )
        expected_ids = expected_support.nonzero(as_tuple=True)[0].tolist()
        self.assertEqual(expected_ids, [0, 1, 2])

        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device="xpu"),
            top_ps=torch.full((batch_size,), top_p, device="xpu"),
            min_ps=torch.zeros(batch_size, device="xpu"),
            return_sampling_masks=[True] * batch_size,
        )
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="intel_xpu")
            ),
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="xpu"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), list(range(batch_size)))
        actual_support = capture.weights > 0
        self.assertTrue(
            torch.equal(actual_support, expected_support.expand_as(actual_support))
        )
        self.assertGreater(int(actual_support[0].sum().item()), top_k)
        self.assertTrue(
            bool(actual_support.gather(1, sampled.view(-1, 1)).all().item())
        )

    def test_intel_xpu_capture_only_materializes_requested_rows(self):
        batch_size = 4
        top_k = 2
        top_p = 0.45
        requested_rows = [1, 3]
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="xpu").repeat(
            batch_size, 1
        )
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), top_k, dtype=torch.int32, device="xpu"),
            top_ps=torch.full((batch_size,), top_p, device="xpu"),
            min_ps=torch.zeros(batch_size, device="xpu"),
            return_sampling_masks=[False, True, False, True],
        )
        top_k_renorm = sampler_module.top_k_renorm_prob
        top_p_renorm = sampler_module.top_p_renorm_prob
        with (
            patch(
                "sglang.srt.layers.sampler.get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="intel_xpu")
                ),
            ),
            patch(
                "sglang.srt.layers.sampler.top_k_renorm_prob",
                wraps=top_k_renorm,
            ) as top_k_mock,
            patch(
                "sglang.srt.layers.sampler.top_p_renorm_prob",
                wraps=top_p_renorm,
            ) as top_p_mock,
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="xpu"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(top_k_mock.call_args.args[0].shape), (2, 5))
        self.assertEqual(tuple(top_p_mock.call_args.args[0].shape), (2, 5))

        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_sampling_mask_to_output(
            output, sampling_info, sampled, capture
        )
        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertEqual(set(output.next_token_sampling_mask_idx[1]), {0, 1, 2})
        self.assertIsNone(output.next_token_sampling_mask_idx[2])
        self.assertEqual(set(output.next_token_sampling_mask_idx[3]), {0, 1, 2})
        self.assertIsNone(output.next_token_sampling_logprobs[0])
        self.assertIsNotNone(output.next_token_sampling_logprobs[1])
        self.assertIsNone(output.next_token_sampling_logprobs[2])
        self.assertIsNotNone(output.next_token_sampling_logprobs[3])

    def test_pytorch_capture_compacts_requested_rows(self):
        batch_size = 4
        requested_rows = [1, 3]
        probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]], device="xpu").repeat(
            batch_size, 1
        )
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
            top_ks=torch.full((batch_size,), 2, dtype=torch.int32, device="xpu"),
            top_ps=torch.full((batch_size,), 0.45, device="xpu"),
            min_ps=torch.zeros(batch_size, device="xpu"),
            return_sampling_masks=[False, True, False, True],
        )
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="pytorch")
            ),
        ):
            sampled, capture = self.sampler._sample_from_probs(
                probs,
                sampling_info,
                positions=torch.zeros(batch_size, dtype=torch.int64, device="xpu"),
                simple_sampling_case=False,
                return_sampling_mask=True,
            )

        self.assertIsNotNone(capture)
        self.assertEqual(capture.batch_rows.cpu().tolist(), requested_rows)
        self.assertEqual(tuple(capture.weights.shape), (len(requested_rows), 5))
        self.assertEqual(tuple(capture.token_ids.shape), (len(requested_rows), 5))

        output = LogitsProcessorOutput(next_token_logits=None)
        self.sampler._attach_sampling_mask_to_output(
            output, sampling_info, sampled, capture
        )
        for batch_row in requested_rows:
            self.assertIn(
                int(sampled[batch_row]),
                output.next_token_sampling_mask_idx[batch_row],
            )
            self.assertIsNotNone(output.next_token_sampling_logprobs[batch_row])
        self.assertIsNone(output.next_token_sampling_mask_idx[0])
        self.assertIsNone(output.next_token_sampling_mask_idx[2])

if __name__ == "__main__":
    unittest.main()
