import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.llama_classification import LlamaForClassification
from sglang.srt.models.llama_reward import (
    LlamaForSequenceClassification,
    LlamaForSequenceClassificationWithNormal_Weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLlamaRewardWeightLoading(CustomTestCase):
    model_classes = (
        LlamaForSequenceClassification,
        LlamaForSequenceClassificationWithNormal_Weights,
        LlamaForClassification,
    )

    def make_model(self, model_class, parameters):
        model = model_class.__new__(model_class)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.model = SimpleNamespace()
        model.named_parameters = lambda: iter(parameters.items())
        return model

    def test_backbone_and_task_heads(self):
        for model_class, head_names in (
            (LlamaForSequenceClassification, ["score.weight"]),
            (
                LlamaForSequenceClassificationWithNormal_Weights,
                ["score.weight", "weights.fc.0.weight", "weights.fc.0.bias"],
            ),
            (LlamaForClassification, ["classification_head.weight"]),
        ):
            with self.subTest(model=model_class.__name__):
                names = ["model.embed_tokens.weight", *head_names]
                parameters = {name: nn.Parameter(torch.zeros(2)) for name in names}
                weights = [
                    (name, torch.full((2,), i + 1.0)) for i, name in enumerate(names)
                ]
                model = self.make_model(model_class, parameters)
                with torch.no_grad():
                    model.load_weights(iter(weights))
                for name, weight in weights:
                    torch.testing.assert_close(parameters[name], weight)

    def test_packed_backbone_weights(self):
        def load_shard(param, weight, shard_id):
            index = {"q": 0, "k": 1, "v": 2}.get(shard_id, shard_id)
            param[index].copy_(weight)

        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                qkv = nn.Parameter(torch.zeros(3, 2))
                gate_up = nn.Parameter(torch.zeros(2, 2))
                qkv.weight_loader = gate_up.weight_loader = load_shard
                model = self.make_model(
                    model_class,
                    {
                        "model.layers.0.self_attn.qkv_proj.weight": qkv,
                        "model.layers.0.mlp.gate_up_proj.weight": gate_up,
                    },
                )
                weights = [
                    (
                        f"model.layers.0.self_attn.{name}_proj.weight",
                        torch.full((2,), value),
                    )
                    for name, value in (("q", 1.0), ("k", 2.0), ("v", 3.0))
                ] + [
                    (f"model.layers.0.mlp.{name}_proj.weight", torch.full((2,), value))
                    for name, value in (("gate", 4.0), ("up", 5.0))
                ]
                with torch.no_grad():
                    model.load_weights(iter(weights))
                torch.testing.assert_close(
                    qkv, torch.tensor([[1.0] * 2, [2.0] * 2, [3.0] * 2])
                )
                torch.testing.assert_close(
                    gate_up, torch.tensor([[4.0] * 2, [5.0] * 2])
                )

    def test_scale_name_remapping(self):
        for model_class in self.model_classes:
            with self.subTest(model=model_class.__name__):
                input_scale = nn.Parameter(torch.zeros(1))
                weight_scale = nn.Parameter(torch.zeros(1))
                prefix = "model.layers.0.mlp.down_proj."
                model = self.make_model(
                    model_class,
                    {
                        prefix + "input_scale": input_scale,
                        prefix + "weight_scale": weight_scale,
                    },
                )
                with torch.no_grad():
                    model.load_weights(
                        iter(
                            [
                                (prefix + "activation_scale", torch.tensor([0.25])),
                                (prefix + "weight_scale_inv", torch.tensor([0.5])),
                            ]
                        )
                    )
                torch.testing.assert_close(input_scale, torch.tensor([0.25]))
                torch.testing.assert_close(weight_scale, torch.tensor([0.5]))


if __name__ == "__main__":
    unittest.main()
