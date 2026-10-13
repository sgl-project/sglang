"""Stacked GPU embedding overrides under TP=2: every rank copies them out of the
tokenizer's CUDA IPC pool. The single-GPU cases are in test_score_engine.py."""

import os
import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")

_MODEL = os.environ.get("TEST_MODEL_NAME", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
_PLACEHOLDER = 50


class TestScoreStackedEmbedsTP2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(model_path=_MODEL, tp_size=2)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_stacked_embed_overrides_match_by_value(self):
        hidden = self.engine.tokenizer_manager.model_config.hidden_size
        gen = torch.Generator().manual_seed(0)
        query = [_PLACEHOLDER, 1000, _PLACEHOLDER, 1001]
        items = [[2000 + i, _PLACEHOLDER, 3000 + i] for i in range(3)]
        query_vecs = 0.02 * torch.randn(2, hidden, generator=gen)
        item_vecs = [0.02 * torch.randn(1, hidden, generator=gen) for _ in items]
        kwargs = dict(
            query=query,
            items=items,
            label_token_ids=[33, 32],
            apply_softmax=True,
            embed_override_token_id=_PLACEHOLDER,
        )

        by_value = self.engine.score(
            query_embed_overrides=list(query_vecs),
            item_embed_overrides=[list(v) for v in item_vecs],
            **kwargs,
        ).scores
        stacked = [torch.cat([query_vecs, v]).cuda() for v in item_vecs]
        actual = self.engine.score(
            stacked_query_item_embed_overrides=stacked, **kwargs
        ).scores

        torch.testing.assert_close(torch.tensor(actual), torch.tensor(by_value))
        transport = self.engine.tokenizer_manager.positional_embeds_transport
        self.assertIsNotNone(transport._pool)


if __name__ == "__main__":
    unittest.main()
