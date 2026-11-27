"""Unit tests for the delta-apply pool size in srt/weight_sync/local_checkpoint.py."""

import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.weight_sync import local_checkpoint
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestAvailableCpus(CustomTestCase):
    def _get_cpus(self, cpu_max):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "cpu.max")
            if cpu_max is not None:
                with open(path, "w") as f:
                    f.write(cpu_max)
            with (
                mock.patch.object(local_checkpoint, "_CPU_MAX_PATH", path),
                mock.patch.object(
                    local_checkpoint.os,
                    "sched_getaffinity",
                    lambda pid: set(range(6)),
                    create=True,
                ),
            ):
                return local_checkpoint._get_available_cpus()

    def test_quota(self):
        self.assertEqual(self._get_cpus("850000 100000\n"), 8)
        self.assertEqual(self._get_cpus("50000 100000\n"), 1)

    def test_no_quota(self):
        self.assertEqual(self._get_cpus("max 100000\n"), 6)
        self.assertEqual(self._get_cpus(None), 6)


if __name__ == "__main__":
    unittest.main()
