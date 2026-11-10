"""Chain reject sampling accepts a draft whose q exceeds 1.0 by float rounding,
as FlashInfer's softmax returns for peaked rows, and still rejects broken q
(0, NaN, inf, or far above 1)."""

import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

VOCAB = 64
GAMMA = 3
DRAFT = 7


def _num_accept(draft_q: float, target_p: float = 1.0) -> int:
    """Accepted drafts for one request whose three drafts are all token DRAFT."""
    device = "cuda"
    slots = GAMMA + 1
    candidates = torch.full((1, slots), DRAFT, dtype=torch.int64, device=device)
    target_probs = torch.zeros((1, slots, VOCAB), dtype=torch.float32, device=device)
    target_probs[:, :, DRAFT] = target_p
    target_probs[:, :, DRAFT + 1] = max(0.0, 1.0 - target_p)
    draft_probs = torch.zeros((1, GAMMA, VOCAB), dtype=torch.float32, device=device)
    draft_probs[:, :, DRAFT] = draft_q
    accept_token_num = torch.empty((1,), dtype=torch.int32, device=device)
    chain_speculative_sampling_triton(
        predicts=torch.empty((slots,), dtype=torch.int32, device=device),
        accept_index=torch.empty((1, slots), dtype=torch.int32, device=device),
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=torch.arange(slots, dtype=torch.int64, device=device).view(1, -1),
        retrive_next_token=None,
        retrive_next_sibling=None,
        uniform_samples=torch.full((1, slots), 0.5, device=device),
        uniform_samples_for_final_sampling=torch.full((1,), 0.5, device=device),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return int(accept_token_num.item())


class TestRejectSamplingQRounding(CustomTestCase):
    def test_q_rounding_above_one_is_accepted(self):
        for overshoot in (1.2e-7, 7.6e-6, 4e-5):
            with self.subTest(overshoot=overshoot):
                self.assertEqual(_num_accept(1.0 + overshoot, 1.0 + overshoot), GAMMA)
                self.assertEqual(_num_accept(1.0 + overshoot, 0.999), GAMMA)

    def test_broken_q_is_still_rejected(self):
        for q in (0.0, float("nan"), float("inf"), float("-inf"), 2.0):
            with self.subTest(q=q):
                self.assertEqual(_num_accept(q), 0)


if __name__ == "__main__":
    unittest.main()
