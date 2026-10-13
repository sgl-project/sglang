"""CPU tests for the PPLX readout's allocation and numerical contract."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.compact_decision_head import CompactDecisionPooler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCompactDecisionPooler(CustomTestCase):
    def test_pools_before_scoring_a_variable_length_batch(self):
        torch.manual_seed(7)
        pooler = CompactDecisionPooler(11, 255)
        hidden = torch.randn(10, 11)
        batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([3, 5, 2]),
            return_pooled_hidden_states=True,
        )
        shapes = []
        pooler.readout.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape))
        )
        result = pooler(hidden, batch)
        last = hidden[[2, 7, 9]]
        torch.testing.assert_close(
            result.embeddings, nn.functional.linear(last, pooler.readout.weight)
        )
        torch.testing.assert_close(result.pooled_hidden_states, last)
        self.assertEqual(shapes, [(3, 11)])
        self.assertEqual(tuple(result.embeddings.shape), (3, 255))

    def test_production_head_size_has_no_vocabulary_padding(self):
        with torch.device("meta"):
            pooler = CompactDecisionPooler(5120, 255)
        self.assertEqual(list(dict(pooler.named_parameters())), ["readout.weight"])
        self.assertEqual(pooler.readout.weight.numel(), 255 * 5120)
        self.assertIsNone(pooler.readout.bias)

    def test_candidate_probabilities_equal_the_scattered_head_reference(self):
        torch.manual_seed(19)
        pooler = CompactDecisionPooler(9, 255).double()
        hidden = torch.randn(4, 9, dtype=torch.float64)
        token_ids = torch.randperm(1024)[:255]
        expanded = torch.zeros(1024, 9, dtype=torch.float64)
        expanded[token_ids] = pooler.readout.weight
        full_logp = torch.log_softmax(hidden @ expanded.T, dim=-1)
        compact = pooler.readout(hidden)
        for count in (2, 3, 26, 255):
            for temperature in (0.6, 1.0087417621345625, 2.0):
                expected = torch.softmax(
                    full_logp[:, token_ids[:count]] / temperature, dim=-1
                )
                actual = torch.softmax(compact[:, :count] / temperature, dim=-1)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
