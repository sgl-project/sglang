"""The HIP weight loader must stage large pageable copies, not pin them in place.

The default raised here is what keeps the loader out of the MMU-notifier eviction
loop, so what is worth pinning is that it is a default and not an override: an
operator who has tuned the threshold must keep their value. The same hook defaults
aiter's AITER_FORCE_A8W4 on gfx1250, again without overriding an operator's value.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.platform_hook import handle_amd_specifics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

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


class TestGfx1250AiterA8W4Default(CustomTestCase):
    VAR = "AITER_FORCE_A8W4"

    def setUp(self):
        saved = {v: os.environ.pop(v, None) for v in (self.VAR, VAR)}
        self.addCleanup(self._restore, saved)

    @staticmethod
    def _restore(saved):
        for var, value in saved.items():
            os.environ.pop(var, None)
            if value is not None:
                os.environ[var] = value

    def _run(self, gcn_arch: str):
        arch = SimpleNamespace(hip=SimpleNamespace(gcn_arch=gcn_arch))
        with (
            patch(
                "sglang.srt.arg_groups.platform_hook.get_platform",
                return_value=_Platform(True),
            ),
            patch("sglang.srt.arg_groups.platform_hook.declare_resolution"),
            patch(
                "sglang.srt.arg_groups.platform_hook.current_arch", return_value=arch
            ),
        ):
            handle_amd_specifics(object())

    def test_set_on_gfx1250(self):
        self._run("gfx1250")
        self.assertEqual(os.environ.get(self.VAR), "1")

    def test_absent_on_gfx950(self):
        self._run("gfx950")
        self.assertIsNone(os.environ.get(self.VAR))

    def test_an_operators_value_survives(self):
        os.environ[self.VAR] = "0"
        self._run("gfx1250")
        self.assertEqual(os.environ[self.VAR], "0")


if __name__ == "__main__":
    unittest.main()
