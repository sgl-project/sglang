import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-mlu")


class TestMLUFusedOps(CustomTestCase):
    def setUp(self):
        self.assertTrue(current_platform.is_mlu())
        torch.mlu.set_device(0)
        self.device = torch.device("mlu", 0)

    def _assert_uses_mlu_forward(self, op):
        self.assertIs(op._forward_method.__self__, op)
        self.assertIs(op._forward_method.__func__, op.forward_mlu.__func__)

    def test_silu_and_mul_matches_native(self):
        from sglang.srt.layers.activation import SiluAndMul

        with patch(
            "sglang.srt.layers.activation.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            op = SiluAndMul()
        self._assert_uses_mlu_forward(op)
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        d = x.shape[-1] // 2
        expected = (torch.nn.functional.silu(x.cpu()[..., :d]) * x.cpu()[..., d:]).to(
            self.device
        )
        actual = op(x)
        self.assertEqual(actual.device.type, "mlu")
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_quick_gelu_matches_native(self):
        from sglang.srt.layers.activation import QuickGELU

        op = QuickGELU()
        self._assert_uses_mlu_forward(op)
        x = torch.randn(4, 8, dtype=torch.bfloat16, device=self.device)
        expected = op.forward_native(x.cpu()).to(self.device)
        actual = op(x)
        self.assertEqual(actual.device.type, "mlu")
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_rms_norm_matches_native(self):
        from sglang.srt.layers.layernorm import RMSNorm

        op = RMSNorm(16, eps=1e-6, weight_dtype=torch.bfloat16).to(self.device)
        self._assert_uses_mlu_forward(op)
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        cpu_op = RMSNorm(16, eps=1e-6, weight_dtype=torch.bfloat16)
        cpu_op.load_state_dict({k: v.cpu() for k, v in op.state_dict().items()})
        expected = cpu_op.forward_native(x.cpu()).to(self.device)
        actual = op(x)
        self.assertEqual(actual.device.type, "mlu")
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_layer_norm_matches_native(self):
        from sglang.srt.layers.layernorm import LayerNorm

        op = LayerNorm(16, eps=1e-6, dtype=torch.bfloat16).to(self.device)
        self._assert_uses_mlu_forward(op)
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        cpu_op = LayerNorm(16, eps=1e-6, dtype=torch.bfloat16)
        cpu_op.load_state_dict({k: v.cpu() for k, v in op.state_dict().items()})
        expected = cpu_op.forward_native(x.cpu()).to(self.device)
        actual = op(x)
        self.assertEqual(actual.device.type, "mlu")
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
