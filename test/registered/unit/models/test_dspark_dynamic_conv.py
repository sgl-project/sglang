import unittest

import torch
from torch import nn

from sglang.srt.models.dspark import (
    ContextAwareCausalResidualHead,
    DFlashGroupedConv,
    DSparkDraftMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkDynamicConv(CustomTestCase):
    def test_identity_initialization_matches_plain_dflash(self):
        hidden = torch.randn(6, 4)
        for mode in ("legacy", "survival-gated"):
            conv = DFlashGroupedConv(
                hidden_size=4,
                block_size=3,
                taps=2,
                group_size=2,
                mode=mode,
            )
            prepared, output_kernel = conv.prepare(hidden)
            finished = conv.finish(hidden, output_kernel)
            torch.testing.assert_close(prepared, hidden)
            torch.testing.assert_close(finished, hidden)

    def test_legacy_lag_does_not_cross_proposal_blocks(self):
        conv = DFlashGroupedConv(
            hidden_size=1,
            block_size=3,
            taps=2,
            group_size=1,
            mode="legacy",
        )
        with torch.no_grad():
            conv.base_kernel[0, 1].fill_(1.0)
        hidden = torch.tensor([[1.0], [2.0], [3.0], [10.0], [20.0], [30.0]])
        prepared, _ = conv.prepare(hidden)
        expected = torch.tensor([[1.0], [3.0], [5.0], [10.0], [30.0], [50.0]])
        torch.testing.assert_close(prepared, expected)

    def test_rejects_runtime_length_that_cannot_form_checkpoint_blocks(self):
        conv = DFlashGroupedConv(
            hidden_size=2,
            block_size=3,
            taps=2,
            group_size=1,
        )
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            conv.prepare(torch.randn(5, 2))

    def test_shorter_runtime_horizon_uses_runtime_block_boundaries(self):
        conv = DFlashGroupedConv(
            hidden_size=1,
            block_size=7,
            taps=2,
            group_size=1,
            mode="legacy",
        )
        with torch.no_grad():
            conv.base_kernel[0, 1].fill_(1.0)
        hidden = torch.tensor(
            [
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [10.0],
                [20.0],
                [30.0],
                [40.0],
                [50.0],
            ]
        )
        prepared, _ = conv.prepare(hidden, block_size=5)
        expected = torch.tensor(
            [
                [1.0],
                [3.0],
                [5.0],
                [7.0],
                [9.0],
                [10.0],
                [30.0],
                [50.0],
                [70.0],
                [90.0],
            ]
        )
        torch.testing.assert_close(prepared, expected)

    def test_dynamic_conv_weight_contract_accepts_matching_checkpoint(self):
        name = "layers.0.attention_conv.kernel_projection.weight"
        parameter = nn.Parameter(torch.zeros(2, 2))
        DSparkDraftMixin._validate_dynamic_conv_weights(
            weights=[(name, torch.ones_like(parameter))],
            params_dict={name: parameter},
        )

    def test_dynamic_conv_weight_contract_rejects_missing_weights(self):
        name = "layers.0.attention_conv.kernel_projection.weight"
        with self.assertRaisesRegex(ValueError, "checkpoint is missing"):
            DSparkDraftMixin._validate_dynamic_conv_weights(
                weights=[],
                params_dict={name: nn.Parameter(torch.zeros(2, 2))},
            )

    def test_dynamic_conv_weight_contract_rejects_unconstructed_modules(self):
        name = "layers.0.attention_conv.kernel_projection.weight"
        with self.assertRaisesRegex(ValueError, "did not construct matching modules"):
            DSparkDraftMixin._validate_dynamic_conv_weights(
                weights=[(name, torch.ones(2, 2))],
                params_dict={},
            )

    def test_carh_supports_shorter_runtime_horizon(self):
        head = ContextAwareCausalResidualHead(
            vocab_size=11,
            markov_rank=4,
            hidden_size=6,
            block_size=7,
        )
        base_logits = torch.randn(2, 5, 11)
        hidden_states = torch.randn(2, 5, 6)
        tokens, logits = head.sample_block(
            base_logits,
            first_prev_tokens=torch.tensor([1, 2]),
            hidden_states=hidden_states,
            sampler=lambda step_logits, _: step_logits.argmax(dim=-1),
        )
        self.assertEqual(tokens.shape, (2, 5))
        self.assertEqual(logits.shape, (2, 5, 11))


if __name__ == "__main__":
    unittest.main()
