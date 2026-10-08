"""Draft draws and graph storage must use the same truncated proposal q."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.draft_sampling import DraftSamplingParams
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import sample_draft_proposal
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestEagleDraftSampling(unittest.TestCase):
    def setUp(self):
        # CPU exercises the supported multinomial fallback; Gumbel's row
        # argmax is a GPU kernel and is covered by device tests.
        self.enterContext(envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False))

    def test_sample_frequencies_match_returned_truncated_q(self):
        torch.manual_seed(100)
        batch_size = 20000
        # top-k keeps the first three tokens; top-p=0.7 then keeps two.
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log().repeat(batch_size, 1)
        params = DraftSamplingParams.create(batch_size, "cpu")
        params.top_ks.fill_(3)
        params.top_ps.fill_(0.7)
        q, sampled_probs, tokens = sample_draft_proposal(logits, params)
        expected = torch.tensor([4 / 7, 3 / 7, 0.0, 0.0])
        torch.testing.assert_close(q, expected.expand_as(q))
        torch.testing.assert_close(sampled_probs, q.gather(1, tokens))
        frequencies = torch.bincount(tokens.flatten(), minlength=4) / batch_size
        torch.testing.assert_close(frequencies, expected, atol=0.015, rtol=0)

    def test_greedy_and_zero_temperature_return_point_mass(self):
        logits = torch.tensor([[0.2, 0.8, 0.6], [0.7, 0.3, 0.5]])
        params = DraftSamplingParams(
            temperatures=torch.tensor([[1.0], [0.0]]),
            top_ks=torch.tensor([1, TOP_K_ALL], dtype=torch.int32),
            top_ps=torch.ones(2),
        )
        q, sampled_probs, tokens = sample_draft_proposal(logits, params)
        torch.testing.assert_close(q, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        self.assertEqual(tokens.flatten().tolist(), [1, 0])
        torch.testing.assert_close(sampled_probs, torch.ones(2, 1))

    def test_single_graph_retains_final_q_for_each_step(self):
        runner = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
        runner.prune_draft_extend_logits = True
        params = DraftSamplingParams.create(2, "cpu")
        params.temperatures[0] = 0
        params.top_ks[1] = 2
        runner.buffers = SimpleNamespace(
            sampling_params=params, draft_probs=torch.full((2, 2, 3), -1.0)
        )
        ret = SimpleNamespace(
            next_token_logits=torch.tensor([[0.0, 2.0, 1.0], [1.0, 0.0, 2.0]])
        )
        expected = torch.tensor(
            [[0.0, 1.0, 0.0], [1 / (1 + torch.e), 0.0, torch.e / (1 + torch.e)]]
        )
        for step in range(2):
            runner.step = step
            runner._sample_draft_proposal(ret, 2)
            stored_q = runner.buffers.draft_probs[:, step]
            torch.testing.assert_close(stored_q, expected)
            torch.testing.assert_close(ret.topk_p, stored_q.gather(1, ret.topk_index))
            self.assertTrue(bool((ret.topk_p > 0).all()))
        self.assertEqual(runner.buffers.draft_probs.shape, (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
