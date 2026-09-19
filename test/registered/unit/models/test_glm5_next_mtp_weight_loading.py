"""Unit tests for GLM-5-Next MTP checkpoint weight loading and layer filtering."""

import unittest
from types import SimpleNamespace

import torch

try:
    from sglang.test.test_utils import CustomTestCase
except ImportError:
    CustomTestCase = unittest.TestCase

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    register_cpu_ci(est_time=10, suite="base-a-test-cpu")
except ImportError:
    pass


class _FakeParam:
    def __init__(self):
        self.loaded = []

    def weight_loader(self, param, loaded_weight, *args, **kwargs):
        self.loaded.append((param, loaded_weight, args, kwargs))


class TestGlm5NextMTPWeightLoading(CustomTestCase):
    def test_mtp_layer_skipped_with_various_prefixes(self):
        """Verify that layers >= num_hidden_layers are skipped prefix-agnostically when is_nextn=False."""
        import re

        fake_param = _FakeParam()
        params_dict = {
            "model.layers.44.input_layernorm.weight": fake_param,
            "model.layers.45.input_layernorm.weight": fake_param,
        }

        num_hidden_layers = 45
        num_nextn_predict_layers = 1
        is_nextn = False

        weights = [
            ("model.language_model.layers.45.input_layernorm.weight", torch.zeros(1)),
            ("language_model.layers.45.input_layernorm.weight", torch.zeros(1)),
            ("model.layers.45.input_layernorm.weight", torch.zeros(1)),
            ("model.language_model.layers.44.input_layernorm.weight", torch.ones(1)),
        ]

        loaded_weights = []

        for name, loaded_weight in weights:
            if "language_model." in name:
                name = name.replace("language_model.", "")

            # Exact guard logic patched in Glm5NextForConditionalGeneration.load_weights
            if not is_nextn:
                if num_nextn_predict_layers > 0:
                    match = re.search(r"layers\.(\d+)", name)
                    if match and int(match.group(1)) >= num_hidden_layers:
                        continue

            if name in params_dict:
                param = params_dict[name]
                param.weight_loader(param, loaded_weight)
                loaded_weights.append(name)

        # Ensure layer 45 is skipped across all prefix variants and only layer 44 loads
        self.assertEqual(len(fake_param.loaded), 1)
        self.assertEqual(loaded_weights, ["model.layers.44.input_layernorm.weight"])
        self.assertEqual(fake_param.loaded[0][1].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
