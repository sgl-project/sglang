import unittest
from unittest import mock

from sglang.srt.layers.rotary_embedding import base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRdna4RopeFallback(CustomTestCase):
    def test_gfx1201_forces_native_rope(self):
        with mock.patch.object(base, "_is_gfx1201", True):
            self.assertTrue(base._should_force_native_rope())


if __name__ == "__main__":
    unittest.main()
