"""CUDA coverage for proposal cutoffs and graph parameter staging."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_graph_replay_observes_new_cutoffs_and_zero_temperature(self):
        params = DraftSamplingParams.greedy(4, "cuda")
        logits = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                build_draft_probs(logits, params)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_q = build_draft_probs(logits, params)
        info = SimpleNamespace(
            temperatures=torch.tensor([[0.5], [2.0]], device="cuda"),
            top_ks=torch.tensor([2, 5], dtype=torch.int32, device="cuda"),
            top_ps=torch.tensor([0.95, 0.7], device="cuda"),
        )
        for override, top_k, top_p in (
            (None, None, None),
            (0.0, -1, 1.0),
            (1.2, 3, 0.5),
            (None, -1, 1.0),
        ):
            with patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_temperature=override,
                    speculative_draft_top_k=top_k,
                    speculative_draft_top_p=top_p,
                ),
            ):
                params.copy_from(info, 2)
            graph.replay()
            expected = build_draft_probs(logits, params)
            torch.testing.assert_close(captured_q, expected, rtol=0, atol=0)
            torch.testing.assert_close(captured_q.sum(-1), torch.ones(4, device="cuda"))


if __name__ == "__main__":
    unittest.main()
