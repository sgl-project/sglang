"""Unit tests for the shared Transformers-fallback loader path in SGLang."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.transformers import TransformersBase
from sglang.srt.models.utils import AutoWeightsLoader
from sglang.srt.runtime_context import get_parallel
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


class TestTransformersFallbackSkipSubstrs(CustomTestCase):
    def test_init_registers_attention_bias_skip(self):
        # Older GPT-NeoX checkpoints (pythia-1.4b, gpt-neox-20b) ship a
        # persistent `attention.bias` causal-mask buffer. Newer transformers
        # builds register it as `persistent=False`, so it is absent from the
        # constructed module tree and `AutoWeightsLoader` raises on the
        # unexpected key unless the fallback tells it to skip. The pre-existing
        # `.attn.bias` covers GPT-2, not NeoX, so `.attention.bias` must be its
        # own entry.
        stub = TransformersBase.__new__(TransformersBase)

        class _Stop(RuntimeError):
            pass

        with (
            get_parallel().override(pp_group=SimpleNamespace()),
            patch(
                "sglang.srt.models.transformers.get_hf_text_config",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang.srt.models.transformers._resolve_attention_backend_model_cls",
                side_effect=_Stop,
            ),
            self.assertRaises(_Stop),
        ):
            TransformersBase.__init__(stub, config=SimpleNamespace())

        self.assertIn(".attention.bias", stub.skip_substrs)


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


if __name__ == "__main__":
    unittest.main()
