"""maybe_copy_inplace must skip the copy when both dst and src are None."""

import unittest

from sglang.srt.layers.attention.dsv4.metadata import maybe_copy_inplace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTensor:
    def __init__(self):
        self.copy_calls = []

    def copy_(self, src):
        self.copy_calls.append(src)


class TestMaybeCopyInplace(unittest.TestCase):
    def test_none_none_skips_copy(self):
        # The "maybe" in the name: dst=None means there is nothing to copy
        # into; the call must be a no-op, not an AttributeError.
        maybe_copy_inplace(None, src=None)

    def test_tensor_tensor_copies(self):
        dst, src = _FakeTensor(), _FakeTensor()
        maybe_copy_inplace(dst, src=src)
        self.assertEqual(dst.copy_calls, [src])

    def test_type_mismatch_asserts(self):
        with self.assertRaises(AssertionError):
            maybe_copy_inplace(_FakeTensor(), src=object())


if __name__ == "__main__":
    unittest.main()
