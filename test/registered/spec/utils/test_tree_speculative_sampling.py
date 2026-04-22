import unittest

import torch
import torch.nn.functional as F

from sglang.srt.speculative.spec_utils import (
    top_k_renorm_prob_fallback,
    top_p_renorm_prob_fallback,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")


class TestTreeSpeculativeSampling(CustomTestCase):

    def _run_tree_spec_sampling(self, threshold_single, threshold_acc):
        from sgl_kernel import tree_speculative_sampling_target_only

        device = get_device()

        candidates = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]],
            dtype=torch.int64,
            device=device,
        )
        retrive_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
            dtype=torch.int64,
            device=device,
        )
        retrive_next_token = torch.tensor(
            [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
            dtype=torch.int64,
            device=device,
        )
        retrive_next_sibling = torch.tensor(
            [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
            dtype=torch.int64,
            device=device,
        )

        target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
        target_logits[0, 0, 3] = 10
        target_logits[0, 3, 4] = 10
        target_logits[0, 4, 5] = 10
        target_logits[1, 0, 11] = 10
        target_logits[1, 4, 12] = 10

        for i in range(target_logits.shape[0]):
            for j in range(target_logits.shape[1]):
                if torch.max(target_logits[i, j]) < 10:
                    target_logits[i, j, 18] = 10

        temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
        bs, num_draft_tokens = candidates.shape
        num_spec_step = 4

        predicts = torch.full((12,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full(
            (bs, num_spec_step), -1, dtype=torch.int32, device=device
        )
        accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

        expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
        target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
        draft_probs = torch.zeros_like(target_probs)
        coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
        coins_for_final_sampling = torch.rand(bs, device=device, dtype=torch.float32)

        tree_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
        )

        return predicts, accept_index, accept_token_num

    def test_threshold_one(self):
        predicts, accept_index, accept_token_num = self._run_tree_spec_sampling(1, 1)
        self.assertEqual(
            predicts.tolist(),
            [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        )
        self.assertEqual(
            accept_index.tolist(),
            [[0, 3, 4, 5], [6, 10, 11, -1]],
        )
        self.assertEqual(accept_token_num.tolist(), [3, 2])

    def test_threshold_zero(self):
        predicts, accept_index, accept_token_num = self._run_tree_spec_sampling(0, 0)
        self.assertEqual(
            predicts.tolist(),
            [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        )
        self.assertEqual(
            accept_index.tolist(),
            [[0, 1, 2, -1], [6, 10, 11, -1]],
        )
        self.assertEqual(accept_token_num.tolist(), [2, 2])


class TestRenormProbFallbacks(CustomTestCase):

    def test_top_k_renorm_scalar(self):
        device = get_device()
        probs = torch.tensor(
            [[0.1, 0.3, 0.05, 0.4, 0.15]], dtype=torch.float32, device=device
        )
        result = top_k_renorm_prob_fallback(probs, top_k=2)
        # Only the top-2 values (0.3, 0.4) should remain and be renormalized
        self.assertEqual((result > 0).sum().item(), 2)
        self.assertAlmostEqual(result.sum().item(), 1.0, places=5)

    def test_top_k_renorm_tensor(self):
        device = get_device()
        probs = torch.tensor(
            [[0.1, 0.3, 0.05, 0.4, 0.15], [0.5, 0.2, 0.1, 0.1, 0.1]],
            dtype=torch.float32,
            device=device,
        )
        top_k = torch.tensor([2, 3], device=device)
        result = top_k_renorm_prob_fallback(probs, top_k=top_k)
        self.assertAlmostEqual(result[0].sum().item(), 1.0, places=5)
        self.assertAlmostEqual(result[1].sum().item(), 1.0, places=5)

    def test_top_p_renorm_scalar(self):
        device = get_device()
        probs = torch.tensor(
            [[0.4, 0.3, 0.15, 0.1, 0.05]], dtype=torch.float32, device=device
        )
        result = top_p_renorm_prob_fallback(probs, top_p=0.8)
        # Top-p=0.8: cumsum of sorted [0.4, 0.3, 0.15] = 0.85 > 0.8
        # So at most 3 tokens should survive
        self.assertLessEqual((result > 0).sum().item(), 3)
        self.assertAlmostEqual(result.sum().item(), 1.0, places=5)

    def test_top_p_renorm_tensor(self):
        device = get_device()
        probs = torch.tensor(
            [[0.4, 0.3, 0.15, 0.1, 0.05], [0.5, 0.2, 0.1, 0.1, 0.1]],
            dtype=torch.float32,
            device=device,
        )
        top_p = torch.tensor([0.8, 0.6], device=device)
        result = top_p_renorm_prob_fallback(probs, top_p=top_p)
        self.assertAlmostEqual(result[0].sum().item(), 1.0, places=5)
        self.assertAlmostEqual(result[1].sum().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=3)
