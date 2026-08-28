import unittest
from unittest import mock

from sglang.srt.layers.rotary_embedding import base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRdna4RopeFallback(CustomTestCase):
    def test_gfx1201_forces_native_rope(self):
        with mock.patch.object(base, "is_gfx1201_supported", return_value=True):
            self.assertTrue(base._should_force_native_rope())

    def test_other_architectures_keep_the_fused_path(self):
        with (
            mock.patch.object(base, "is_gfx1201_supported", return_value=False),
            mock.patch.object(base, "publish_role", return_value=None),
        ):
            self.assertFalse(base._should_force_native_rope())


if __name__ == "__main__":
    unittest.main()
