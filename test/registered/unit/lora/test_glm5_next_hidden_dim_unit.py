"""Regression tests for GLM-5.3-Flash LoRA projection dimensions."""

import unittest

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.lora.utils import get_default_hidden_dim, get_hidden_dim
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# GLM-5.3-Flash has 34 KDA and 11 DSA layers. Its released config sets
# head_dim=0, so the generic qkv_proj calculation returns a zero output width.
_DSA_LAYERS = range(3, 45, 4)
_KDA_LAYERS = tuple(i for i in range(45) if i not in _DSA_LAYERS)


class TestGlm5NextLoRAHiddenDims(CustomTestCase):
    def setUp(self):
        self.config = Glm5NextTextConfig(
            hidden_size=4096,
            num_hidden_layers=45,
            num_attention_heads=64,
            num_key_value_heads=64,
            head_dim=0,
            v_head_dim=256,
            linear_num_heads=64,
            linear_head_dim=128,
            layer_types=[
                "deepseek_sparse_attention" if i in _DSA_LAYERS else "linear_attention"
                for i in range(45)
            ],
        )
        # The dimension resolver only needs the text config. Skip weight loading.
        self.model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        self.model.config = self.config

    def _dims(self, module_name, layer_idx):
        return get_hidden_dim(module_name, self.config, self.model, layer_idx)

    def test_kda_o_proj_uses_linear_attention_width(self):
        for layer_idx in _KDA_LAYERS:
            with self.subTest(layer_idx=layer_idx):
                self.assertEqual(self._dims("o_proj", layer_idx), (8192, 4096))

    def test_kda_qkv_proj_uses_packed_width_with_zero_head_dim(self):
        for layer_idx in _KDA_LAYERS:
            with self.subTest(layer_idx=layer_idx):
                self.assertEqual(self._dims("qkv_proj", layer_idx), (4096, 24576))

    def test_dsa_o_proj_keeps_dsa_width(self):
        for layer_idx in _DSA_LAYERS:
            with self.subTest(layer_idx=layer_idx):
                self.assertEqual(self._dims("o_proj", layer_idx), (16384, 4096))

    def test_other_projections_keep_default_dimensions(self):
        for layer_idx, module_name in (
            (0, "gate_up_proj"),
            (3, "fused_qkv_a_proj_with_mqa"),
            (4, "down_proj"),
        ):
            with self.subTest(layer_idx=layer_idx, module_name=module_name):
                self.assertEqual(
                    self._dims(module_name, layer_idx),
                    get_default_hidden_dim(module_name, self.config, layer_idx),
                )


if __name__ == "__main__":
    unittest.main()
