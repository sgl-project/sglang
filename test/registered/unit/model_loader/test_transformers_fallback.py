"""Unit tests for the shared Transformers-fallback loader path in SGLang."""

import unittest

import torch

from sglang.srt.models.transformers import TransformersBase
from sglang.srt.models.utils import AutoWeightsLoader
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


class TestAutoWeightsLoaderIgnoreUnexpectedPatterns(CustomTestCase):
    def _make_loader(self) -> tuple[torch.nn.Module, AutoWeightsLoader]:
        module = torch.nn.Module()
        module.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        )
        # GLM-4-MoE style: one pattern names an MTP-only layer, the other a real layer.
        loader = AutoWeightsLoader(
            module, ignore_unexpected_patterns=[r"layers\.2.*", r"layers\.1.*"]
        )
        return module, loader

    def test_skips_only_keys_without_a_parameter(self):
        # HF `_keys_to_ignore_on_load_unexpected` regexes must drop checkpoint keys
        # with no matching parameter, but still load a real parameter they match.
        module, loader = self._make_loader()

        loaded = loader.load_weights(
            [
                ("layers.0.weight", torch.ones(2, 2)),
                ("layers.1.weight", torch.full((2, 2), 2.0)),
                ("layers.2.weight", torch.zeros(2, 2)),
            ]
        )

        self.assertEqual(loaded, {"layers.0.weight", "layers.1.weight"})
        self.assertTrue(torch.equal(module.layers[1].weight, torch.full((2, 2), 2.0)))

    def test_unmatched_unexpected_key_still_raises(self):
        _, loader = self._make_loader()

        with self.assertRaisesRegex(ValueError, "No module or parameter named"):
            loader.load_weights([("mtp.weight", torch.zeros(2, 2))])

    def test_patterns_match_complete_keys(self):
        # A full-key pattern must match the unexpected key itself, not the
        # module prefix (`layers.2`) where the loader finds no submodule.
        module, _ = self._make_loader()
        loader = AutoWeightsLoader(
            module,
            ignore_unexpected_patterns=[r"layers\.1\.weight$", r"layers\.2\.weight$"],
        )

        loaded = loader.load_weights(
            [
                ("layers.1.weight", torch.full((2, 2), 2.0)),
                ("layers.2.weight", torch.zeros(2, 2)),
            ]
        )

        self.assertEqual(loaded, {"layers.1.weight"})
        self.assertTrue(torch.equal(module.layers[1].weight, torch.full((2, 2), 2.0)))

    def test_patterns_match_checkpoint_names_before_mapping(self):
        # The fallback mapper turns `mtp.weight` into `model.mtp.weight`; an
        # anchored HF pattern still has to match the original checkpoint key.
        module = torch.nn.Module()
        module.model = torch.nn.Module()
        module.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
        loader = AutoWeightsLoader(
            module, ignore_unexpected_patterns=[r"^mtp.*", r"^layers\.0\."]
        )

        loaded = loader.load_weights(
            [
                ("layers.0.weight", torch.ones(2, 2)),
                ("mtp.weight", torch.zeros(2, 2)),
            ],
            mapper=TransformersBase.hf_to_sglang_mapper,
        )

        self.assertEqual(loaded, {"model.layers.0.weight"})
        self.assertTrue(torch.equal(module.model.layers[0].weight, torch.ones(2, 2)))


if __name__ == "__main__":
    unittest.main()
