import unittest

from sglang.srt.layers.utils import MultiPlatformOp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _ForwardTrackerOp(MultiPlatformOp):
    def dispatch_forward(self):
        return self.forward_platform

    def forward_platform(self, *args, **kwargs):
        return "platform"

    def forward_native(self, *args, **kwargs):
        return "native"


class DummyLayer(_ForwardTrackerOp):
    pass


class TopK(_ForwardTrackerOp):
    pass


class DummyFusedMoE(_ForwardTrackerOp):
    pass


class TestMultiPlatformTorchCompileOverride(CustomTestCase):
    def test_default_heuristic_switches_regular_layers(self):
        layer = DummyLayer()

        self.assertEqual(layer.forward(), "platform")

        layer.enter_torch_compile(num_tokens=4)

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_default_topk_heuristic_keeps_multi_token_path(self):
        layer = TopK()

        self.assertEqual(layer.forward(), "platform")

        layer.enter_torch_compile(num_tokens=4)

        self.assertEqual(layer.forward(), "platform")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_default_topk_heuristic_switches_single_token_path(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=1)

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_default_fused_moe_heuristic_switches_single_token_path(self):
        layer = DummyFusedMoE()

        layer.enter_torch_compile(num_tokens=1)

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_override_layers_switch_fused_moe_for_multi_token(self):
        layer = DummyFusedMoE()

        layer.enter_torch_compile(num_tokens=4, override_layers=["DummyFusedMoE"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

    def test_override_layers_switch_topk_for_multi_token(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=4, override_layers=["TopK"])

        self.assertEqual(layer.forward(), "native")
        self.assertTrue(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")

    def test_override_layers_require_exact_class_name(self):
        layer = TopK()

        layer.enter_torch_compile(num_tokens=4, override_layers=["Top"])

        self.assertEqual(layer.forward(), "platform")
        self.assertFalse(layer.is_torch_compile)

    def test_override_layers_skip_non_listed_layers(self):
        layer = DummyLayer()

        layer.enter_torch_compile(num_tokens=4, override_layers=["TopK"])

        self.assertEqual(layer.forward(), "platform")
        self.assertFalse(layer.is_torch_compile)

        layer.leave_torch_compile()

        self.assertEqual(layer.forward(), "platform")


if __name__ == "__main__":
    unittest.main(verbosity=3)
