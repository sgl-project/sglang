"""Kimi-K3's load_weights indexes parameters one decoder layer at a time, and
the index answers as dict(model.named_parameters()) does."""

import unittest

from torch import nn

from sglang.srt.models.kimi_k3 import _LayerwiseParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.experts = nn.Linear(4, 8, bias=False)


class _Model(nn.Module):
    """K3's parameter layout: model.layers.{i} beside embeddings, norm, head."""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        self.model.layers = nn.ModuleList(_Layer() for _ in range(num_layers))
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 10, bias=False)


def _index(model: nn.Module) -> _LayerwiseParams:
    return _LayerwiseParams(model, _LayerwiseParams.shared_names(model))


class TestKimiK3LayerwiseParams(CustomTestCase):
    def test_answers_as_the_whole_model_index(self):
        model = _Model()
        expected = dict(model.named_parameters())
        index = _index(model)
        absent = [
            "model.layers.1.mlp.experts.bias",
            "model.layers.7.self_attn.weight",
            "model.layers.01.self_attn.weight",
            "model.layers.2",
            "model.norm.scale",
            "visual.layers.0.weight",
        ]
        for name in list(expected) + absent:
            self.assertEqual(name in index, name in expected, name)
        for name, param in expected.items():
            self.assertIs(index[name], param, name)

    def test_reads_only_the_layers_it_is_asked_about(self):
        index = _index(_Model())
        self.assertIn("model.layers.2.self_attn.weight", index)
        # A layer-only call never indexes the whole model.
        self.assertIsNone(index._all)
        self.assertEqual(set(index._layers), {2})

    def test_answers_a_shared_parameter_only_by_its_first_name(self):
        # Whole-model duplicate removal keeps a shared parameter under its first
        # name only, so loading its second name must not reach (and overwrite) it.
        model = _Model()
        model.model.layers[1].self_attn.weight = model.model.layers[0].self_attn.weight
        expected = dict(model.named_parameters())
        index = _index(model)
        self.assertNotIn("model.layers.1.self_attn.weight", expected)
        self.assertNotIn("model.layers.1.self_attn.weight", index)
        for name, param in expected.items():
            self.assertIs(index[name], param, name)


if __name__ == "__main__":
    unittest.main()
