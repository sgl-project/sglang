import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.kernels.plugin_kernels.npu import spec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestNPURegistration(unittest.TestCase):
    def test_lazy_resolution_and_conflicts(self):
        module = ModuleType("example_npu_provider")
        module.target = spec.MegakernelSpec("target", lambda runner, cfg: None)
        with patch.dict(spec._registry, clear=True):
            key = "GlmMoeDsaForCausalLM"
            for _ in range(2):
                spec.register(architecture=key, path=module.__name__ + ":target")
            self.assertNotIn(module.__name__, sys.modules)
            self.assertIsNone(spec.find_spec("UnknownArchitecture"))
            with patch.dict(sys.modules, {module.__name__: module}):
                self.assertIs(spec.find_spec(key), module.target)
            with self.assertRaises(ValueError):
                spec.register(architecture=key, path="conflicting:spec")


if __name__ == "__main__":
    unittest.main()
