import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-mlu")


def _mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401

        return bool(torch.mlu.is_available())
    except Exception:
        return False


@unittest.skipUnless(_mlu_available(), "MLU device is not available")
class TestMLUFusedOps(CustomTestCase):
    def setUp(self):
        torch.mlu.set_device(0)
        self.device = torch.device("mlu", 0)

    def test_silu_and_mul_matches_native(self):
        from sglang.srt.layers.activation import SiluAndMul

        with patch(
            "sglang.srt.layers.activation.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            op = SiluAndMul()
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        d = x.shape[-1] // 2
        expected = (torch.nn.functional.silu(x.cpu()[..., :d]) * x.cpu()[..., d:]).to(
            self.device
        )
        actual = op.forward_mlu(x)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_quick_gelu_matches_native(self):
        from sglang.srt.layers.activation import QuickGELU

        op = QuickGELU()
        x = torch.randn(4, 8, dtype=torch.bfloat16, device=self.device)
        expected = op.forward_native(x.cpu()).to(self.device)
        actual = op.forward_mlu(x)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_rms_norm_matches_native(self):
        from sglang.srt.layers.layernorm import RMSNorm

        op = RMSNorm(16, eps=1e-6, weight_dtype=torch.bfloat16).to(self.device)
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        cpu_op = RMSNorm(16, eps=1e-6, weight_dtype=torch.bfloat16)
        cpu_op.load_state_dict({k: v.cpu() for k, v in op.state_dict().items()})
        expected = cpu_op.forward_native(x.cpu()).to(self.device)
        actual = op.forward_mlu(x)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_layer_norm_matches_native(self):
        from sglang.srt.layers.layernorm import LayerNorm

        op = LayerNorm(16, eps=1e-6, dtype=torch.bfloat16).to(self.device)
        x = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)
        cpu_op = LayerNorm(16, eps=1e-6, dtype=torch.bfloat16)
        cpu_op.load_state_dict({k: v.cpu() for k, v in op.state_dict().items()})
        expected = cpu_op.forward_native(x.cpu()).to(self.device)
        actual = op.forward_mlu(x)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
