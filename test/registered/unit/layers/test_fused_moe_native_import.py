import importlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestFusedMoeNativeImport(CustomTestCase):
    def test_fused_moe_native_imports(self):
        module = importlib.import_module("sglang.srt.layers.moe.fused_moe_native")

        self.assertTrue(hasattr(module, "fused_moe_forward_native"))
        self.assertTrue(hasattr(module, "moe_forward_native"))


if __name__ == "__main__":
    unittest.main()
