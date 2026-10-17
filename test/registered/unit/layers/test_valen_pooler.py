# SPDX-License-Identifier: Apache-2.0
"""Valen layout rejection must preserve unrelated requests in a prefill batch."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.valen_layout import (
    ValenLayout,
    ValenQuestionLayout,
    pack_valen_layout,
)
from sglang.srt.layers.valen_mixer import ValenMixerHead
from sglang.srt.layers.valen_pooler import ValenMixerPooler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestValenPooler(CustomTestCase):
    def test_invalid_and_cached_requests_do_not_shift_valid_neighbors(self):
        layout = ValenLayout(
            8, (0, 2), (ValenQuestionLayout("choice", (2, 3), ((3, 4), (4, 5)), 6),)
        )
        raw = pack_valen_layout(layout)
        torch.manual_seed(42)
        head = ValenMixerHead(
            8,
            head_width=8,
            head_layers=1,
            head_token_hidden_dim=4,
            head_channel_hidden_dim=12,
        )
        pooler = ValenMixerPooler(head)
        hidden = torch.randn(31, 8)
        batch = SimpleNamespace(
            batch_size=4,
            extend_seq_lens_cpu=[8, 7, 8, 8],
            extend_prefix_lens_cpu=[0, 0, 3, 0],
            valen_layouts=[raw, raw, raw, raw],
        )
        result = pooler(hidden, batch).embeddings
        self.assertEqual([len(x) for x in result], [2, 0, 0, 2])
        torch.testing.assert_close(
            result[0], head.forward_batch(hidden[:8], [8], [layout])[0]
        )
        torch.testing.assert_close(
            result[3], head.forward_batch(hidden[23:], [8], [layout])[0]
        )

    def test_health_check_does_not_require_a_layout(self):
        head = ValenMixerHead(
            8,
            head_width=8,
            head_layers=1,
            head_token_hidden_dim=4,
            head_channel_hidden_dim=12,
        )
        batch = SimpleNamespace(
            batch_size=1, extend_seq_lens_cpu=[2], valen_layouts=None
        )
        result = ValenMixerPooler(head)(torch.randn(2, 8), batch)
        self.assertEqual(len(result.embeddings), 1)
        self.assertEqual(result.embeddings[0].numel(), 0)


if __name__ == "__main__":
    unittest.main()
