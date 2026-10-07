"""Unit tests for get_amdgpu_memory_capacity — no GPU needed."""

import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MIB = 1024 * 1024


def _rocminfo(stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(
        args="rocminfo", returncode=returncode, stdout=stdout, stderr=""
    )


def _hip_devices(*total_mib: int):
    props = [SimpleNamespace(total_memory=t * _MIB) for t in total_mib]
    return (
        patch.object(common.torch.cuda, "is_available", return_value=bool(props)),
        patch.object(common.torch.cuda, "device_count", return_value=len(props)),
        patch.object(
            common.torch.cuda, "get_device_properties", side_effect=lambda i: props[i]
        ),
    )


class TestAmdgpuMemoryCapacity(CustomTestCase):
    def _capacity(self, rocminfo, *total_mib):
        avail, count, props = _hip_devices(*total_mib)
        with (
            patch.object(common.subprocess, "run", return_value=rocminfo),
            avail,
            count,
            props,
        ):
            return common.get_amdgpu_memory_capacity()

    def test_reads_smallest_rocminfo_pool(self):
        out = "301989888(0x12000000) KB\n301465600(0x11f80000) KB\n"
        self.assertEqual(self._capacity(_rocminfo(out)), 301465600 / 1024)

    def test_skips_lines_without_a_size(self):
        out = "\n301989888(0x12000000) KB\nN/A\n"
        self.assertEqual(self._capacity(_rocminfo(out)), 301989888 / 1024)

    def test_empty_rocminfo_falls_back_to_hip(self):
        self.assertEqual(self._capacity(_rocminfo(""), 294896, 294800), 294800)

    def test_failed_rocminfo_falls_back_to_hip(self):
        self.assertEqual(self._capacity(_rocminfo("", returncode=1), 294896), 294896)

    def test_no_rocminfo_and_no_device_raises(self):
        with self.assertRaises(RuntimeError):
            self._capacity(_rocminfo(""))


if __name__ == "__main__":
    unittest.main()
