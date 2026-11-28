"""Unit tests for run_unittest_files in sglang.test.ci.ci_utils."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils.process import kill_process_tree
from sglang.test.ci import ci_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestRunUnittestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _file(self, name, body):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(textwrap.dedent(body))
        return SimpleNamespace(name=path, estimated_time=1)

    def test_output_is_relayed(self):
        test_file = self._file(
            "test_print.py",
            """
            import sys
            print("relayed-stdout")
            print("relayed-stderr", file=sys.stderr)
            """,
        )
        script = (
            "from types import SimpleNamespace\n"
            "from sglang.test.ci.ci_utils import run_unittest_files\n"
            f"f = SimpleNamespace(name={test_file.name!r}, estimated_time=1)\n"
            "raise SystemExit(run_unittest_files([f], timeout_per_file=60))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("relayed-stdout", result.stdout)
        self.assertIn("relayed-stderr", result.stderr)

    def test_unkillable_timeout_stops_the_suite(self):
        marker = os.path.join(self.tmp.name, "second_ran")
        slow = self._file("test_slow.py", "import time\ntime.sleep(120)\n")
        second = self._file("test_second.py", f"open({marker!r}, 'w').close()\n")

        def kill_then_report_unreaped(pid, *args, **kwargs):
            kill_process_tree(pid, *args, **kwargs)
            raise RuntimeError("1 process(es) not reaped within 60s after SIGKILL")

        with patch.object(ci_utils, "kill_process_tree", kill_then_report_unreaped):
            ret = ci_utils.run_unittest_files(
                [slow, second], timeout_per_file=2, continue_on_error=True
            )

        self.assertEqual(ret, -1)
        self.assertFalse(os.path.exists(marker))


if __name__ == "__main__":
    unittest.main()
