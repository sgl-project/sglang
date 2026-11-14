"""--quantization-param-path KV scales are stored the way checkpoint KV scales are."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    """Assigns float scales the way the models' ``load_kv_cache_scales`` do."""

    def __init__(self, scales):
        super().__init__()
        self.attn = torch.nn.ModuleList(
            RadixAttention(
                num_heads=2, head_dim=8, scaling=1.0, num_kv_heads=1, layer_id=i
            )
            for i in range(len(scales))
        )
        self.scales = scales

    def load_kv_cache_scales(self, _path):
        for attn, scale in zip(self.attn, self.scales):
            if scale is not None:
                attn.k_scale = attn.v_scale = scale


class TestLoadKvCacheScales(unittest.TestCase):
    def test_float_scales_become_device_tensors_with_host_copies(self):
        model = _Model([0.25, None])
        with patch.object(
            load_model_utils,
            "get_model",
            return_value=SimpleNamespace(quantization_param_path="scales.json"),
        ):
            load_model_utils.load_kv_cache_scales(
                model=model, kv_cache_dtype="fp8_e4m3", device="cpu"
            )

        loaded, untouched = model.attn
        for name in ("k_scale", "v_scale"):
            scale = getattr(loaded, name)
            self.assertIsInstance(scale, torch.Tensor)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertEqual(scale.shape, torch.Size([]))
            self.assertEqual(scale.device, torch.device("cpu"))
            self.assertEqual(scale.item(), 0.25)
            self.assertEqual(getattr(loaded, f"{name}_float"), 0.25)
            self.assertIsNone(getattr(untouched, name))
            self.assertIsNone(getattr(untouched, f"{name}_float"))


if __name__ == "__main__":
    unittest.main()
