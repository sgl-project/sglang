"""Unit tests for the shared Transformers-fallback loader path in SGLang."""

import unittest

from sglang.srt.models.transformers import TransformersBase
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestTransformersFallbackWeightMapper(CustomTestCase):
    def test_gpt_neox_prefix_rewrites(self):
        # GPT-NeoX checkpoints (Pythia, gpt-neox-20b) wrap the backbone under
        # `gpt_neox.` and expose the LM head as `embed_out.`. `AutoModel`
        # returns the backbone as `self.model` and `CausalMixin` installs a
        # `ParallelLMHead` at `self.lm_head`, so the fallback mapper must
        # rewrite both prefixes. Without these rules, `load_weights` raises
        # unexpected-key errors on every `gpt_neox.*` / `embed_out.*` entry.
        mapped = TransformersBase.hf_to_sglang_mapper.apply_list(
            [
                "gpt_neox.embed_in.weight",
                "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
                "gpt_neox.final_layer_norm.weight",
                "embed_out.weight",
            ]
        )

        self.assertEqual(
            mapped,
            [
                "model.embed_in.weight",
                "model.layers.0.mlp.dense_h_to_4h.weight",
                "model.final_layer_norm.weight",
                "lm_head.weight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
