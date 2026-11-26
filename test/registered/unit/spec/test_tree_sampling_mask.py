# ruff: noqa: E402
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

# Import runtime modules after installing the CPU kernel stub.
from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.speculative import spec_sampling_mask
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_sampling_mask import build_tree_sampling_mask


class TestTreeSamplingMask(CustomTestCase):
    def test_selects_policy_rows_on_the_committed_tree_path(self):
        # The first request skips a sibling. The second did not request masks.
        # The third accepts only its root and requests selected-token logprobs.
        probs = torch.ones(3, 3, 4) / 4
        probs[0, 0] = torch.tensor([0.0, 0.6, 0.4, 0.0])
        probs[0, 2] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        probs[2, 0] = torch.tensor([0.75, 0.0, 0.25, 0.0])
        predict = torch.tensor([2, 0, 3, 0, 0, 0, 0, 0, 0])
        out = build_tree_sampling_mask(
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

    def test_greedy_chain_includes_bonus_and_ignores_unfilled_slots(self):
        out = build_tree_sampling_mask(
            SimpleNamespace(
                sampling_mask_batch_indices=torch.tensor([0, 1]),
                sampling_support_logprobs_capture_indices=torch.tensor([1]),
            ),
            None,
            torch.tensor([3, 5, 1, 4, -1, -1]),
            torch.tensor([[0, 1, 2], [3, -1, -1]]),
            torch.tensor([3, 1]),
            max_tokens=2,
        )
        self.assertEqual(out.num_accept_tokens.tolist(), [3, 1])
        self.assertEqual(out.token_ids[0, :, 0].tolist(), [3, 5, 1])
        self.assertEqual(out.token_ids[1, 0, 0].item(), 4)
        self.assertEqual(out.support_logprobs[0, 0, 0].item(), 0.0)
        self.assertEqual(out.selected_logprobs.count_nonzero().item(), 0)

    def test_only_stock_tree_verifiers_admit_sampling_masks(self):
        req = SimpleNamespace(sampling_params=SimpleNamespace(top_k=2, min_p=0.0))
        spec = SimpleNamespace(
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
            speculative_use_rejection_sampling=False,
            enable_multi_layer_eagle=False,
            speculative_eagle_topk=2,
        )
        validate = spec_sampling_mask.validate_spec_sampling_mask_request
        alg = SpeculativeAlgorithm
        with (
            patch.object(spec_sampling_mask, "is_cuda", return_value=True),
            patch.object(spec_sampling_mask, "get_spec", return_value=spec),
            spec_sampling_mask.envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(
                False
            ),
            spec_sampling_mask.envs.SGLANG_SIMULATE_ACC_LEN.override(0),
            spec_sampling_mask.envs.SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING.override(
                False
            ),
        ):
            for name in ("EAGLE", "EAGLE3", "NEXTN", "NGRAM"):
                with self.subTest(name=name):
                    self.assertIsNone(validate(req, alg.from_string(name)))
            for member in (alg.FROZEN_KV_MTP, alg.UNO, alg.STANDALONE):
                self.assertIn("not supported", validate(req, member))
            spec.enable_multi_layer_eagle = True
            self.assertIn("chain", validate(req, alg.EAGLE))
            spec.speculative_eagle_topk = 1
            self.assertIsNone(validate(req, alg.EAGLE))
            spec.speculative_use_rejection_sampling = True
            self.assertIn("target-only", validate(req, alg.EAGLE))
            spec.speculative_use_rejection_sampling = False
            spec.speculative_accept_threshold_acc = 0.5
            for member in (alg.EAGLE, alg.EAGLE3, alg.NGRAM):
                self.assertIn("threshold", validate(req, member))
            spec.speculative_accept_threshold_acc = 1.0
            with spec_sampling_mask.envs.SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING.override(
                True
            ):
                self.assertIn("compact", validate(req, alg.EAGLE3))
            with patch.object(spec_sampling_mask, "is_cuda", return_value=False):
                for member in (alg.EAGLE, alg.EAGLE3, alg.NGRAM):
                    self.assertIn("CUDA", validate(req, member))

    def test_reports_overflow_for_tied_support(self):
        out = build_tree_sampling_mask(
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
