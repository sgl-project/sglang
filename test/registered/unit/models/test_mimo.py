"""Unit tests for srt/models/mimo.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import Qwen2Config

from sglang.srt.models.mimo import MiMoForCausalLM
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiMoForCausalLM(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )

    def test_get_input_embeddings_returns_the_embedding_module(self):
        """The runner and the speculative workers call get_input_embeddings()
        without arguments and apply the returned module to token ids."""
        model = MiMoForCausalLM(
            Qwen2Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=64,
                max_position_embeddings=64,
            )
        )
        self.assertIs(model.get_input_embeddings(), model.model.embed_tokens)


if __name__ == "__main__":
    unittest.main()
