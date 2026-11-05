"""Unit tests for sanitizing caller-supplied profiler filename components (#44189)."""

import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.utils.profile_utils import sanitize_profile_filename_component

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSanitizeProfileFilenameComponent(CustomTestCase):
    def _assert_stays_in(self, prefix, profile_id):
        out_dir = os.path.realpath("/tmp/sglang-profile-root/expected")
        filename = (
            sanitize_profile_filename_component(prefix)
            + "-"
            + sanitize_profile_filename_component(profile_id)
            + "-TP-0.trace.json.gz"
        )
        path = os.path.realpath(os.path.join(out_dir, filename))
        self.assertEqual(os.path.dirname(path), out_dir)

    def test_relative_prefix_cannot_escape(self):
        self._assert_stays_in("../escaped", "123")
        self._assert_stays_in("..\\escaped", "123")

    def test_absolute_prefix_cannot_replace_output_dir(self):
        self._assert_stays_in("/tmp/elsewhere/x", "123")

    def test_profile_id_cannot_escape(self):
        self._assert_stays_in("p", "../../../tmp/x")

    def test_plain_values_unchanged(self):
        self.assertEqual(sanitize_profile_filename_component("run-1.a_b"), "run-1.a_b")
        self.assertEqual(
            sanitize_profile_filename_component("1730361600.5"), "1730361600.5"
        )
        self.assertIsNone(sanitize_profile_filename_component(None))
        self.assertEqual(sanitize_profile_filename_component(""), "")

    def test_separators_replaced(self):
        self.assertEqual(sanitize_profile_filename_component("a/b\\c\0d"), "a_b_c_d")


if __name__ == "__main__":
    unittest.main()
