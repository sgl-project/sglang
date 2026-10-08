import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.speculative.spec_sampling_mask import SpeculativeSamplingMaskCapture
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkDraftSamplingMask(unittest.TestCase):
    def test_deterministic_draft_preserves_stochastic_target_support(self):
        target_probs = torch.tensor([[0.1, 0.6, 0.3], [0.2, 0.7, 0.1]])
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(2, 1),
            top_ks=torch.tensor([3, 1], dtype=torch.int32),
            top_ps=torch.ones(2),
            is_all_greedy=False,
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            sampling_mask_batch_indices=torch.arange(2),
            sampling_mask_top_ks=[3, 1],
            sampling_support_logprobs_capture_indices=torch.arange(2),
        )
        for temperatures, top_ks in (
            (torch.zeros(2, 1), torch.tensor([3, 1])),
            (torch.ones(2, 1), torch.ones(2, dtype=torch.int32)),
        ):
            with self.subTest(temperatures=temperatures, top_ks=top_ks):
                q = build_draft_probs(
                    torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]),
                    DraftSamplingParams(temperatures, top_ks, torch.ones(2)),
                )
                torch.testing.assert_close(
                    q, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                )
                capture = SpeculativeSamplingMaskCapture.from_logits(
                    sampling_info,
                    next_token_logits=target_probs.log().repeat_interleave(2, 0),
                    draft_input=SimpleNamespace(max_top_k=3, uniform_top_k_value=None),
                    draft_token_num=2,
                    bs=2,
                )
                torch.testing.assert_close(
                    capture.greedy_mask, torch.tensor([False, True])
                )
                output = capture.build_output(
                    out_tokens=torch.tensor([[0, 2], [1, 1]]),
                    commit_lens=torch.tensor([2, 2]),
                )
                torch.testing.assert_close(
                    output.lengths, torch.tensor([[3, 3], [1, 1]], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    output.selected_logprobs,
                    torch.tensor([[0.1, 0.3], [1.0, 1.0]]).log(),
                )
                torch.testing.assert_close(
                    output.token_ids[0],
                    torch.tensor([[1, 2, 0], [1, 2, 0]]),
                )
                torch.testing.assert_close(
                    output.support_logprobs[0],
                    torch.tensor([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1]]).log(),
                )


if __name__ == "__main__":
    unittest.main()
