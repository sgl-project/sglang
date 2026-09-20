import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MarkovHead:
    def __init__(self, rank: int):
        self.rank = rank
        self.prev_tokens = None

    def get_prev_embeddings(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        self.prev_tokens = prev_tokens
        return torch.zeros((*prev_tokens.shape, self.rank), dtype=torch.float32)


class TestDeepseekV4DSparkConfidence(CustomTestCase):
    def test_uses_live_proposal_gamma_after_runtime_override(self):
        checkpoint_gamma = 5
        runtime_gamma = 6
        bs = 2
        hidden_size = 8
        markov_rank = 4
        markov_head = _MarkovHead(markov_rank)
        model = SimpleNamespace(
            gamma=checkpoint_gamma,
            confidence_head=DSparkConfidenceHead(
                hidden_size=hidden_size,
                markov_rank=markov_rank,
                with_markov=True,
                bias=False,
            ),
            markov_head=markov_head,
        )
        anchor_tokens = torch.tensor([10, 20])
        sampled_tokens = torch.arange(bs * runtime_gamma).reshape(bs, runtime_gamma)
        x_post_hc = torch.randn(bs * runtime_gamma, hidden_size)

        confidence = DeepseekV4ForCausalLMDSpark.compute_confidence(
            model,
            anchor_tokens=anchor_tokens,
            sampled_tokens=sampled_tokens,
            x_post_hc=x_post_hc,
        )

        self.assertEqual(tuple(confidence.shape), (bs, runtime_gamma))
        expected_prev = torch.cat(
            [anchor_tokens[:, None], sampled_tokens[:, :-1]], dim=1
        )
        torch.testing.assert_close(markov_head.prev_tokens, expected_prev)

    def test_rejects_hidden_rows_from_a_different_proposal_width(self):
        model = SimpleNamespace(
            gamma=5,
            confidence_head=DSparkConfidenceHead(
                hidden_size=8,
                markov_rank=0,
                with_markov=False,
                bias=False,
            ),
            markov_head=None,
        )

        with self.assertRaisesRegex(ValueError, r"expected bs \* gamma = 2 \* 6 = 12"):
            DeepseekV4ForCausalLMDSpark.compute_confidence(
                model,
                anchor_tokens=torch.tensor([10, 20]),
                sampled_tokens=torch.zeros((2, 6), dtype=torch.int64),
                x_post_hc=torch.randn(10, 8),
            )


if __name__ == "__main__":
    unittest.main()
