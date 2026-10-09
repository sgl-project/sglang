"""
Unit tests for the NemotronHMLP activation.

Shared experts and MLP layers must follow ``config.mlp_hidden_act``,
like the routed experts do.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.models.nemotron_h import NemotronHMLP
from sglang.test.test_utils import CustomTestCase


def _build_mlp(act: str) -> NemotronHMLP:
    config = SimpleNamespace(hidden_size=8, mlp_hidden_act=act)
    return NemotronHMLP(config, intermediate_size=16, tp_rank=0, tp_size=1)


class TestNemotronHMLPActivation(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 16)

    def test_relu2_is_the_default_activation(self):
        act_fn = _build_mlp("relu2").act_fn

        torch.testing.assert_close(
            act_fn.forward_native(self.x), F.relu(self.x) ** 2, rtol=0, atol=0
        )

    def test_silu_config_uses_silu(self):
        act_fn = _build_mlp("silu").act_fn

        torch.testing.assert_close(act_fn(self.x), F.silu(self.x), rtol=0, atol=0)

    def test_activation_name_is_case_insensitive(self):
        self.assertIsInstance(_build_mlp("SiLU").act_fn, torch.nn.SiLU)

    def test_unsupported_activation_raises(self):
        with self.assertRaisesRegex(ValueError, "mlp_hidden_act 'gelu'"):
            _build_mlp("gelu")


if __name__ == "__main__":
    unittest.main()
