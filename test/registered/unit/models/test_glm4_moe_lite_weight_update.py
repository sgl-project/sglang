"""Bucketed QKV-A updates for GLM-4.7-Flash."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Q = "model.layers.0.self_attn.q_a_proj.weight"
KV = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"


class TestGlm4MoeLiteWeightUpdate(unittest.TestCase):
    def setUp(self):
        model = Glm4MoeLiteForCausalLM.__new__(Glm4MoeLiteForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(q_lora_rank=2, n_routed_experts=0)
        model.num_fused_shared_experts = 0
        model.quant_config = None
        model.model = nn.Module()
        model.model.start_layer, model.model.end_layer = 0, 1
        model.model.layers = nn.ModuleList([nn.Module()])
        attn = model.model.layers[0].self_attn = nn.Module()
        self.proj = attn.fused_qkv_a_proj_with_mqa = nn.Linear(2, 3, bias=False)
        self.model = model
        self.weights = [(Q, torch.arange(4.0).reshape(2, 2)), (KV, torch.ones(1, 2))]
        self.updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=model.config,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=None,
            recapture_cuda_graph=None,
            get_model_runner=None,
        )

    def check_split_update(self, order):
        self.model.load_weights(self.weights)
        expected = self.proj.weight.detach().clone()
        torch.testing.assert_close(expected, torch.cat([w for _, w in self.weights]))
        self.proj.weight.data.zero_()
        self.updater.begin_weight_update()
        name, tensor = order[0]
        bucket = tensor.clone()
        self.model.load_weights([(name, bucket)])
        bucket.fill_(-9)
        self.model.load_weights([order[1]])
        self.assertIsNone(self.updater.end_weight_update(run_post_load=False))
        torch.testing.assert_close(self.proj.weight, expected)

    def test_split_q_first_after_bucket_reuse(self):
        self.check_split_update(self.weights)

    def test_split_kv_first_after_bucket_reuse(self):
        self.check_split_update(self.weights[::-1])

    def test_incomplete_updates_do_not_mix_sessions(self):
        self.proj.weight.data.zero_()
        self.model.load_weights(self.weights[:1])
        self.updater.begin_weight_update()
        self.model.load_weights(self.weights[1:])
        message = self.updater.end_weight_update(run_post_load=False)
        self.assertIsNotNone(message)
        self.assertIn(KV, message)
        self.model.load_weights(self.weights[:1])
        self.assertIn(Q, self.updater.end_weight_update(run_post_load=False))
        torch.testing.assert_close(self.proj.weight, torch.zeros_like(self.proj.weight))

    def test_split_nextn_loads_the_remapped_decoder(self):
        self.model.config.num_hidden_layers = 2
        self.model.config.num_nextn_predict_layers = 1
        self.model.model.decoder = self.model.model.layers.pop(0)
        self.proj.weight.data.zero_()
        for name, weight in self.weights:
            name = name.replace("layers.0", "layers.2")
            self.model.load_weights([(name, weight)], is_nextn=True)
        expected = torch.cat([w for _, w in self.weights])
        torch.testing.assert_close(self.proj.weight, expected)

    def test_absent_fused_parameter_is_not_pending(self):
        absent = Q.replace("layers.0", "layers.1")
        self.model.load_weights([(absent, self.weights[0][1])])
        self.assertIsNone(self.updater.end_weight_update(run_post_load=False))


if __name__ == "__main__":
    unittest.main()
