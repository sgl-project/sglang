"""The HIP weight loader must stage large pageable copies, not pin them in place.

The default raised here is what keeps the loader out of the MMU-notifier eviction
loop, so what is worth pinning is that it is a default and not an override: an
operator who has tuned the threshold must keep their value.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.platform_hook import handle_amd_specifics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VAR = "GPU_PINNED_MIN_XFER_SIZE"


class _Platform:
    def __init__(self, is_hip: bool):
        self.is_hip = is_hip


class TestRocmPageableH2DStaging(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(VAR, None)

    def tearDown(self):
        os.environ.pop(VAR, None)
        if self._saved is not None:
            os.environ[VAR] = self._saved

    def _run(self, is_hip: bool):
        with (
            patch(
                "sglang.srt.arg_groups.platform_hook.get_platform",
                return_value=_Platform(is_hip),
            ),
            patch("sglang.srt.arg_groups.platform_hook.declare_resolution"),
        ):
            handle_amd_specifics(object())

    def test_set_on_hip(self):
        self._run(is_hip=True)
        self.assertEqual(os.environ.get(VAR), str(4 * 1024 * 1024))

    def test_absent_off_hip(self):
        self._run(is_hip=False)
        self.assertIsNone(os.environ.get(VAR))

    def test_an_operators_value_survives(self):
        os.environ[VAR] = "12345"
        self._run(is_hip=True)
        self.assertEqual(os.environ[VAR], "12345")


if __name__ == "__main__":
    unittest.main()
