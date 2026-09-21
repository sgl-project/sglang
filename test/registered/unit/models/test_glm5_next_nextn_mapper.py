"""Unit tests for the GLM-5.3-Flash NextN draft weight-name mapper.

The draft's quantization lookups are keyed on names produced by
``Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper``, so what that
mapper emits decides whether the checkpoint's per-expert schemes and its
`exclude` list can be found at all.

``WeightsMapper._map_name`` rewrites a name at most once -- longest matching rule
first, then it stops -- so rules do not chain. These tests pin the two namings a
draft lookup can arrive in (the checkpoint's own ``model.language_model.layers.N``
and the already-normalized ``model.layers.N``) to the same runtime name, and pin
that the draft layer's siblings land beside the transformer block rather than
inside it.

Pure CPU logic (no server / engine launch).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.test.test_utils import CustomTestCase

# GLM-5.3-Flash: 45 target layers, so the draft layer is index 45.
NUM_HIDDEN_LAYERS = 45
DRAFT = NUM_HIDDEN_LAYERS


class _TextConfig:
    num_hidden_layers = NUM_HIDDEN_LAYERS


class _Config:
    text_config = _TextConfig


class TestGlm5NextNMapper(CustomTestCase):
    def setUp(self):
        self.mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(
            _Config
        )

    def assert_maps(self, cases):
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(self.mapper._map_name(source), expected)

    def test_transformer_block_goes_under_decoder(self):
        # Both namings have to reach the runtime module path, which is what
        # `_find_matched_config` and `should_ignore_layer` are queried with.
        self.assert_maps(
            {
                f"model.language_model.layers.{DRAFT}.mlp.experts.7.up_proj": "model.decoder.mlp.experts.7.up_proj",
                f"model.layers.{DRAFT}.mlp.experts.7.up_proj": "model.decoder.mlp.experts.7.up_proj",
                f"model.language_model.layers.{DRAFT}.self_attn.kv_b_proj": "model.decoder.self_attn.kv_b_proj",
                f"model.layers.{DRAFT}.self_attn.kv_b_proj": "model.decoder.self_attn.kv_b_proj",
                f"model.language_model.layers.{DRAFT}.mlp.gate": "model.decoder.mlp.gate",
                f"model.language_model.layers.{DRAFT}": "model.decoder",
                f"model.layers.{DRAFT}": "model.decoder",
            }
        )

    def test_sibling_projections_stay_beside_the_block(self):
        # eh_proj/enorm/hnorm are children of `model`, not of the decoder layer.
        # `model.layers.45.eh_proj` also occurs inside
        # `model.language_model.layers.45.eh_proj` (because `language_model.`
        # ends in `model.`), so the prefixed rule must win and not mangle it.
        self.assert_maps(
            {
                f"model.language_model.layers.{DRAFT}.eh_proj.weight": "model.eh_proj.weight",
                f"model.layers.{DRAFT}.eh_proj.weight": "model.eh_proj.weight",
                f"model.language_model.layers.{DRAFT}.enorm.weight": "model.enorm.weight",
                f"model.layers.{DRAFT}.enorm.weight": "model.enorm.weight",
                f"model.language_model.layers.{DRAFT}.hnorm.weight": "model.hnorm.weight",
                f"model.layers.{DRAFT}.hnorm.weight": "model.hnorm.weight",
            }
        )

    def test_target_layers_are_normalized_but_not_sent_to_decoder(self):
        # The checkpoint's `exclude` list names every target layer too; those
        # still need the `model.language_model.` prefix stripped, and must not
        # be routed into the draft's decoder.
        self.assert_maps(
            {
                "model.language_model.layers.0.self_attn.q_proj": "model.layers.0.self_attn.q_proj",
                f"model.language_model.layers.{DRAFT - 1}.mlp.experts.3.up_proj": f"model.layers.{DRAFT - 1}.mlp.experts.3.up_proj",
                "model.visual.blocks.0.attn.proj": "visual.blocks.0.attn.proj",
                "lm_head.weight": "lm_head.weight",
            }
        )

    def test_no_mapped_name_keeps_the_checkpoint_prefix(self):
        # A surviving `language_model` in the output means a lookup can never
        # match a runtime module name.
        for source in (
            f"model.language_model.layers.{DRAFT}.eh_proj",
            f"model.language_model.layers.{DRAFT}.enorm",
            f"model.language_model.layers.{DRAFT}.hnorm",
            f"model.language_model.layers.{DRAFT}.mlp.experts.0.gate_proj",
            "model.language_model.layers.0.self_attn.o_proj",
        ):
            with self.subTest(source=source):
                self.assertNotIn("language_model", self.mapper._map_name(source))

    def test_draft_index_comes_from_the_text_config_when_present(self):
        # Multimodal configs nest the layer count under text_config; a text-only
        # config carries it directly, and both must pick the same draft index.
        for config in (_Config, _TextConfig):
            with self.subTest(config=config.__name__):
                mapper = Glm5NextForConditionalGenerationNextN.get_hf_to_sglang_mapper(
                    config
                )
                self.assertEqual(
                    mapper._map_name(f"model.layers.{DRAFT}.mlp.gate"),
                    "model.decoder.mlp.gate",
                )


if __name__ == "__main__":
    unittest.main()
