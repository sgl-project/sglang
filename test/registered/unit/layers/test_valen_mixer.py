# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.srt.layers.valen_layout import ValenLayout, ValenQuestionLayout
from sglang.srt.layers.valen_mixer import ValenMixerHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def head():
    torch.manual_seed(42)
    return ValenMixerHead(
        8,
        head_width=6,
        head_layers=2,
        head_token_hidden_dim=5,
        head_channel_hidden_dim=12,
    ).eval()


def layout():
    return ValenLayout(
        14, (0, 3), (ValenQuestionLayout("choice", (4, 6), ((7, 9), (10, 11)), 12),)
    )


class TestValenMixer(CustomTestCase):
    def test_packed_batch_has_no_request_leakage(self):
        model = head()
        torch.manual_seed(17)
        (first, second) = (torch.randn(14, 8), torch.randn(14, 8) + 100)
        expected = [
            model.forward_batch(row, [14], [layout()])[0] for row in (first, second)
        ]
        actual = model.forward_batch(
            torch.cat((first, second)), [14, 14], [layout(), layout()]
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_roles_pool_spans_and_read_decision_token(self):
        model = head()
        hidden = torch.arange(112).reshape(14, 8).float() / 13
        features = torch.stack(
            [
                torch.stack(
                    (
                        hidden[:3].mean(0),
                        hidden[4:6].mean(0),
                        hidden[7:9].mean(0),
                        hidden[12],
                    )
                ),
                torch.stack(
                    (
                        hidden[:3].mean(0),
                        hidden[4:6].mean(0),
                        hidden[10:11].mean(0),
                        hidden[12],
                    )
                ),
            ]
        )
        torch.testing.assert_close(
            model.forward_batch(hidden, [14], [layout()])[0],
            model.score_features(features),
        )
        changed = hidden.clone()
        changed[[3, 6, 9, 11, 13]] += 10000
        torch.testing.assert_close(
            model.forward_batch(changed, [14], [layout()])[0],
            model.forward_batch(hidden, [14], [layout()])[0],
            rtol=0,
            atol=0,
        )

    def test_fp32_head_under_bf16_default_and_autocast(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = head()
        finally:
            torch.set_default_dtype(previous)
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        hidden = torch.randn(14, 8, dtype=torch.bfloat16)
        expected = model.forward_batch(hidden, [14], [layout()])[0]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = model.forward_batch(hidden, [14], [layout()])[0]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_packed_hidden_length_must_be_exact(self):
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            head().forward_batch(torch.randn(15, 8), [14], [layout()])

    def test_mixer_scores_candidate_rows_independently(self):
        model = head()
        features = torch.randn(7, 4, 8)
        expected = model.score_features(features)
        order = torch.tensor([6, 3, 0, 2, 5, 1, 4])
        torch.testing.assert_close(
            model.score_features(features[order]), expected[order]
        )

    def test_checkpoint_names_and_shapes(self):
        state = head().state_dict()
        assert state["roles"].shape == (1, 4, 6)
        assert state["projection.weight"].shape == (6, 8)
        assert state["blocks.0.token_mlp.0.weight"].shape == (5, 4)
        assert state["blocks.1.channel_mlp.0.weight"].shape == (12, 6)
        assert state["output.1.weight"].shape == (1, 6)


if __name__ == "__main__":
    unittest.main()
