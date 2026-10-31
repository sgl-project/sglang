import unittest

import torch
from torch import nn

from sglang.srt.model_loader.weight_utils import set_kv_cache_layer_scales
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKvCacheScaleAssignment(CustomTestCase):
    def test_preserves_registered_parameters_and_updates_host_mirrors(self):
        layer = nn.Module()
        layer.k_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        layer.v_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        layer.k_scale_float = 1.0
        layer.v_scale_float = 1.0
        k_scale = layer.k_scale
        v_scale = layer.v_scale

        set_kv_cache_layer_scales(layer, 0.25)

        self.assertIs(layer.k_scale, k_scale)
        self.assertIs(layer.v_scale, v_scale)
        self.assertEqual(layer.k_scale.item(), 0.25)
        self.assertEqual(layer.v_scale.item(), 0.25)
        self.assertEqual(layer.k_scale_float, 0.25)
        self.assertEqual(layer.v_scale_float, 0.25)

    def test_creates_expandable_tensors_when_scales_are_absent(self):
        layer = nn.Module()
        layer.k_scale = None
        layer.v_scale = None
        layer.k_scale_float = None
        layer.v_scale_float = None

        set_kv_cache_layer_scales(layer, 0.25, device=torch.device("cpu"))

        self.assertEqual(layer.k_scale.expand((2, 3)).shape, (2, 3))
        self.assertEqual(layer.v_scale.expand((2, 3)).shape, (2, 3))
        self.assertEqual(layer.k_scale.dtype, torch.float32)
        self.assertEqual(layer.v_scale.dtype, torch.float32)
        self.assertEqual(layer.k_scale_float, 0.25)
        self.assertEqual(layer.v_scale_float, 0.25)


if __name__ == "__main__":
    unittest.main()
