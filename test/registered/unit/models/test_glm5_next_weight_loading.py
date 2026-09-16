import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _FakeExpertParam(_FakeParam):
    def weight_loader(self, param, loaded_weight, weight_name, *, shard_id, expert_id):
        self.loaded = loaded_weight
        self.weight_name = weight_name
        self.shard_id = shard_id
        self.expert_id = expert_id


class TestGlm5NextWeightLoading(unittest.TestCase):
    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_quark_block_fp8_weight_scale_loads_scale_inv(self, post_load):
        scale_param = _FakeParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=0,
                num_hidden_layers=45,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            named_parameters=lambda: iter(
                [("model.layers.0.mlp.down_proj.weight_scale_inv", scale_param)]
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.0.mlp.down_proj.weight_scale",
                    loaded_scale,
                )
            ],
        )

        self.assertIs(scale_param.loaded, loaded_scale)
        post_load.assert_called_once()

    @patch("sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights")
    def test_nextn_fused_moe_block_fp8_scale_loads_scale_inv(self, post_load):
        scale_param = _FakeExpertParam()
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=1,
                num_hidden_layers=45,
                num_nextn_predict_layers=1,
            ),
            num_fused_shared_experts=0,
            quant_config=None,
            named_parameters=lambda: iter(
                [("model.decoder.mlp.experts.w2_weight_scale_inv", scale_param)]
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        Glm5NextForConditionalGeneration.load_weights(
            model,
            [
                (
                    "model.language_model.layers.45.mlp.experts.0."
                    "down_proj.weight_scale",
                    loaded_scale,
                )
            ],
            is_nextn=True,
        )

        self.assertIs(scale_param.loaded, loaded_scale)
        self.assertEqual(
            scale_param.weight_name,
            "model.decoder.mlp.experts.w2_weight_scale_inv",
        )
        self.assertEqual(scale_param.shard_id, "w2")
        self.assertEqual(scale_param.expert_id, 0)
        post_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
