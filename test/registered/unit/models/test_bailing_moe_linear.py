"""Unit tests for srt/models/bailing_moe_linear.py."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

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
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers


class TestBailingMoELinearTiedLMHead(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")

    def test_tied_config_shares_the_embedding_with_the_lm_head(self):
        """A config with tie_word_embeddings must build, with lm_head being the
        inner model's embedding rather than a second vocab matrix."""
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            tie_word_embeddings=True,
        )
        with (
            patch.object(
                bailing_moe_linear, "BailingMoELinearModel", _EmbeddingOnlyModel
            ),
            get_parallel().override(pp_group=SimpleNamespace(is_last_rank=True)),
        ):
            model = bailing_moe_linear.BailingMoELinearForCausalLM(config=config)

        self.assertIs(model.lm_head, model.model.word_embeddings)
        embed, head = model.get_embed_and_head()
        self.assertIs(embed, head)


if __name__ == "__main__":
    unittest.main()
