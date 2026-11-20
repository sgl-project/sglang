"""Tests that py-spy can dump a process whose watchdog opted it in under CI."""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.utils.watchdog import WatchdogRaw
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
# The NVIDIA runner containers are where watchdog py-spy dumps are refused, so
# check there too.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _yama_ptrace_scope():
    try:
        return int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text())
    except (OSError, ValueError):
        return None


@unittest.skipUnless(sys.platform == "linux", "Yama and py-spy dumps are Linux-only")
class TestAllowPyspyAttach(CustomTestCase):
    def test_pyspy_can_dump_this_process(self):
        if shutil.which("py-spy") is None:
            self.skipTest("py-spy is not installed")
        scope = _yama_ptrace_scope()
        if scope is not None and scope >= 2:
            self.skipTest(f"Yama ptrace_scope={scope} needs CAP_SYS_PTRACE")

        # Opt in through the watchdog so the test also covers its CI wiring.
        with envs.SGLANG_IS_IN_CI.override(True):
            WatchdogRaw(
                debug_name="test",
                get_counter=lambda: 0,
                is_active=lambda: False,
                watchdog_timeout=3600,
            )
        # py-spy runs as a child of the process it dumps, like the watchdog's.
        result = subprocess.run(
            ["py-spy", "dump", "--pid", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"py-spy dump failed (ptrace_scope={scope}): {result.stderr}",
        )
        self.assertIn("test_pyspy_can_dump_this_process", result.stdout)


if __name__ == "__main__":
    unittest.main()
