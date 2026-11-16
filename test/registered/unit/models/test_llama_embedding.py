"""Unit tests for srt/models/llama_embedding.py"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from sglang.srt.models.llama_embedding import LlamaEmbeddingModel
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestLlamaEmbeddingLoadWeights(CustomTestCase):
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

    def test_skipped_tensors_do_not_end_the_load(self):
        """Checkpoint tensors the model has no parameter for are skipped one by
        one; the weights that follow them in the checkpoint are still loaded."""
        model = LlamaEmbeddingModel(
            LlamaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=64,
                max_position_embeddings=64,
            )
        )
        norm = model.model.norm.weight
        skipped = torch.zeros(1)
        for name in (
            "layers.0.self_attn.rotary_emb.inv_freq",
            "layers.0.self_attn.rotary_emb.cos_cached",
            "model.vision_tower.embeddings.weight",
            "layers.0.self_attn.o_proj.bias",
        ):
            with self.subTest(skipped=name):
                expected = torch.randn_like(norm)
                model.load_weights([(name, skipped), ("norm.weight", expected)])
                self.assertTrue(torch.equal(norm, expected))


if __name__ == "__main__":
    unittest.main()
