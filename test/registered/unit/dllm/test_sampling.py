"""Unit tests for DLLM sampling helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="base-b-test-cpu")

import unittest

import torch

from sglang.srt.dllm.algorithm.sampling import sample_block_tokens
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.test_utils import CustomTestCase


def _make_sampling_info(
    *,
    temperature=1.0,
    top_p=1.0,
    top_k=TOP_K_ALL,
    min_p=0.0,
):
    return SamplingBatchInfo(
        temperatures=torch.tensor([temperature], dtype=torch.float).view(1, 1),
        top_ps=torch.tensor([top_p], dtype=torch.float),
        top_ks=torch.tensor([top_k], dtype=torch.int32),
        min_ps=torch.tensor([min_p], dtype=torch.float),
        is_all_greedy=top_k <= 1,
        need_top_p_sampling=top_p != 1.0,
        need_top_k_sampling=top_k != TOP_K_ALL,
        need_min_p_sampling=min_p > 0.0,
        vocab_size=4,
        device="cpu",
    )


class TestDllmSampling(CustomTestCase):

    def test_greedy_preserves_argmax(self):
        logits = torch.tensor(
            [
                [0.0, 2.0, 1.0, -1.0],
                [3.0, 0.0, 1.0, -1.0],
            ]
        )
        sampling_info = _make_sampling_info(
            temperature=0.7,
            top_p=0.5,
            top_k=1,
        )

        token_ids, token_probs = sample_block_tokens(logits, sampling_info, 0)

        expected_probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.equal(token_ids, torch.tensor([1, 0])))
        self.assertTrue(
            torch.allclose(
                token_probs,
                expected_probs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1),
            )
        )

    def test_top_k_limits_sampled_tokens(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.0, 10.0, 9.0, 8.0]]).repeat(128, 1)
        sampling_info = _make_sampling_info(top_k=2)

        token_ids, _ = sample_block_tokens(logits, sampling_info, 0)

        self.assertTrue(set(token_ids.tolist()).issubset({1, 2}))

    def test_top_p_limits_sampled_tokens(self):
        torch.manual_seed(0)
        logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]])).repeat(128, 1)
        sampling_info = _make_sampling_info(top_p=0.69)

        token_ids, _ = sample_block_tokens(logits, sampling_info, 0)

        self.assertTrue(set(token_ids.tolist()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
