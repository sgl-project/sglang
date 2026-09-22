import unittest

from sglang.srt.model_executor.model_runner import (
    _can_use_synchronous_hisparse_shared,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseSharedGate(unittest.TestCase):
    def test_requires_every_runtime_capability(self):
        enabled = {
            "is_hip_backend": True,
            "use_aiter": True,
            "gfx95": True,
            "is_hisparse_dsa_pool": True,
        }
        self.assertTrue(_can_use_synchronous_hisparse_shared(**enabled))
        for name in enabled:
            with self.subTest(disabled=name):
                candidate = {**enabled, name: False}
                self.assertFalse(_can_use_synchronous_hisparse_shared(**candidate))


if __name__ == "__main__":
    unittest.main()
