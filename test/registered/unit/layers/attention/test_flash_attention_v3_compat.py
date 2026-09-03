import unittest

from sglang.kernels.ops.attention.flash_attention_v3 import _call_fa3_kernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashAttentionV3Compatibility(unittest.TestCase):
    def test_omits_false_only_qv_for_older_kernel(self):
        calls = []

        def old_kernel(value, **kwargs):
            calls.append(kwargs.copy())
            if "only_qv" in kwargs:
                raise TypeError("unexpected keyword argument 'only_qv'")
            return value

        self.assertEqual(
            _call_fa3_kernel(old_kernel, 7, only_qv=False),
            7,
        )
        self.assertEqual(calls, [{"only_qv": False}, {}])

    def test_does_not_drop_enabled_only_qv(self):
        def old_kernel(**kwargs):
            raise TypeError("unexpected keyword argument 'only_qv'")

        with self.assertRaisesRegex(TypeError, "only_qv"):
            _call_fa3_kernel(old_kernel, only_qv=True)


if __name__ == "__main__":
    unittest.main()
