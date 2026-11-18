"""Tests for disable_transparent_hugepages (SGLANG_DISABLE_THP)."""

import ctypes
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

from sglang.srt.utils import process
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PR_GET_THP_DISABLE = 42


class TestDisableTransparentHugepages(CustomTestCase):
    def _call(self, *, platform="linux", prctl_result=0, errno=0):
        libc = Mock()
        libc.prctl.return_value = prctl_result
        with (
            patch.object(process.sys, "platform", platform),
            patch.object(process.ctypes, "CDLL", return_value=libc) as cdll,
            patch.object(process.ctypes, "get_errno", return_value=errno),
            patch.object(process, "logger") as logger,
        ):
            ok = process.disable_transparent_hugepages()
        return ok, cdll, libc, logger

    def test_sets_thp_disable_with_unsigned_long_arguments(self):
        ok, _, libc, logger = self._call()
        self.assertTrue(ok)
        (option, *args), _ = libc.prctl.call_args
        self.assertEqual(option, 41)
        self.assertTrue(all(isinstance(a, ctypes.c_ulong) for a in args))
        self.assertEqual([a.value for a in args], [1, 0, 0, 0])
        logger.warning.assert_not_called()

    def test_a_failed_prctl_warns(self):
        ok, _, _, logger = self._call(prctl_result=-1, errno=22)
        self.assertFalse(ok)
        logger.warning.assert_called_once()

    def test_other_platforms_skip_prctl(self):
        ok, cdll, _, logger = self._call(platform="darwin")
        self.assertFalse(ok)
        cdll.assert_not_called()
        logger.warning.assert_called_once()


@unittest.skipUnless(sys.platform == "linux", "PR_SET_THP_DISABLE is Linux-only")
class TestDisableTransparentHugepagesOnLinux(CustomTestCase):
    def test_this_process_and_its_children_have_thp_disabled(self):
        # In a child interpreter, so this test process keeps its THP setting.
        script = textwrap.dedent(
            f"""
            import ctypes, subprocess, sys
            from sglang.srt.utils.process import disable_transparent_hugepages

            def thp_disabled():
                return ctypes.CDLL(None).prctl({_PR_GET_THP_DISABLE}, 0, 0, 0, 0)

            before = thp_disabled()
            assert disable_transparent_hugepages()
            child = subprocess.run(
                [sys.executable, "-c",
                 "import ctypes; print(ctypes.CDLL(None).prctl({_PR_GET_THP_DISABLE}, 0, 0, 0, 0))"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            print(before, thp_disabled(), child)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        before, after, child = result.stdout.split()[-3:]
        if before == "1":
            self.skipTest("THP is already disabled for this process tree")
        self.assertEqual((after, child), ("1", "1"))


if __name__ == "__main__":
    unittest.main()
