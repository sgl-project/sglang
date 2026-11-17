"""Unit tests for srt/models/dots_ocr.py."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.dots_ocr import DotsOCRConfig
from sglang.srt.models.dots_ocr import DotsOCRForCausalLM
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A multiple of 64, so the embedding table carries no padding rows.
_VOCAB_SIZE = 64
_HIDDEN_SIZE = 16


def _build_model(tie_word_embeddings: bool) -> DotsOCRForCausalLM:
    config = DotsOCRConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
        vision_config=dict(
            embed_dim=16,
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
        ),
    )
    pp_group = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
    )
    with get_parallel().override(pp_group=pp_group):
        return DotsOCRForCausalLM(config)


class TestDotsOCRLanguageHead(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")

    def test_embed_and_head_are_the_weights_the_checkpoint_fills(self):
        """The embedding and LM head the wrapper hands to a draft model must be
        the ones load_weights fills."""
        model = _build_model(tie_word_embeddings=False)
        embed_weight = torch.arange(_VOCAB_SIZE * _HIDDEN_SIZE).reshape(
            _VOCAB_SIZE, _HIDDEN_SIZE
        )
        head_weight = -embed_weight
        model.load_weights(
            [
                ("model.embed_tokens.weight", embed_weight),
                ("lm_head.weight", head_weight),
            ]
        )

        embed, head = model.get_embed_and_head()

        torch.testing.assert_close(embed.float(), embed_weight.float())
        torch.testing.assert_close(head.float(), head_weight.float())

    def test_tied_config_builds_with_one_shared_vocab_matrix(self):
        """A checkpoint with tied word embeddings must build, with the LM head
        sharing the input embedding."""
        model = _build_model(tie_word_embeddings=True)

        embed, head = model.get_embed_and_head()

        self.assertIs(embed, head)
        self.assertIs(model.lm_head.weight, embed)


if __name__ == "__main__":
    unittest.main()
