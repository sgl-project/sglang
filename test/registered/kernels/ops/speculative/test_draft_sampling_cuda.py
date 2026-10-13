"""Draft q and target p must agree on CUDA for the same logits and cutoffs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDraftSamplingCUDA(unittest.TestCase):
    def test_same_logits_and_params_match_dense_target(self):
        torch.manual_seed(7)
        for width in (16, 4096, 129280):
            with self.subTest(width=width):
                logits = torch.randn(3, width, device="cuda")
                params = DraftSamplingParams(
                    torch.tensor([0.7, 1.0, 1.5], device="cuda"),
                    torch.tensor([7, 12, TOP_K_ALL], dtype=torch.int32, device="cuda"),
                    torch.tensor([0.7, 0.95, 1.0], device="cuda"),
                )
                info = SimpleNamespace(
                    temperatures=params.temperatures[:, None],
                    top_ks=params.top_ks,
                    top_ps=params.top_ps,
                    need_top_k_sampling=True,
                    need_top_p_sampling=True,
                )
                target = build_speculative_verify_target_probs(
                    next_token_logits=logits,
                    sampling_info=info,
                    draft_token_num=1,
                    bs=3,
                    use_sparse_topk=False,
                )[:, 0]
                proposal = build_draft_probs(logits, params)
                self.assertTrue(torch.equal(proposal > 0, target > 0))
                torch.testing.assert_close(proposal, target, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
