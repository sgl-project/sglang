"""Unit tests for srt/models/bailing_moe_linear.py."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import bailing_moe_linear
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _EmbeddingOnlyModel(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        super().__init__()
        self.word_embeddings = (
            nn.Embedding(config.vocab_size, config.hidden_size)
            if get_parallel().pp_group.is_first_rank
            else bailing_moe_linear.PPMissingLayer()
        )
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers


class TestBailingMoELinearTiedLMHead(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")

        self.config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_experts=2,
            tie_word_embeddings=True,
        )
        self.enterContext(
            patch.object(
                bailing_moe_linear, "BailingMoELinearModel", _EmbeddingOnlyModel
            )
        )

    def test_tied_config_shares_the_embedding_with_the_lm_head(self):
        with get_parallel().override(
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True)
        ):
            model = bailing_moe_linear.BailingMoeV2_5ForCausalLM(config=self.config)

        self.assertIs(model.lm_head, model.model.word_embeddings)
        embed, head = model.get_embed_and_head()
        self.assertIs(embed, head)

    def test_last_pipeline_stage_loads_tied_head_from_embedding(self):
        weight = torch.randn(
            self.config.vocab_size, self.config.hidden_size, dtype=torch.float32
        )
        with (
            set_default_torch_dtype(torch.bfloat16),
            get_parallel().override(
                pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=True)
            ),
        ):
            model = bailing_moe_linear.BailingMoeV2_5ForCausalLM(config=self.config)
            loaded = model.load_weights([("model.word_embeddings.weight", weight)])

        self.assertIsInstance(model.lm_head, bailing_moe_linear.ParallelLMHead)
        self.assertIn("lm_head.weight", loaded)
        torch.testing.assert_close(
            model.lm_head.weight[: self.config.vocab_size], weight.bfloat16()
        )


if __name__ == "__main__":
    unittest.main()
