"""memfd creation for the shared engram host table. CPU only, Linux only."""

import os
import sys
import unittest

from sglang.srt.layers.engram import _memfd_create
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@unittest.skipUnless(sys.platform == "linux", "memfd is Linux only")
class TestEngramMemfd(CustomTestCase):
    def test_libc_path_works_without_the_os_helper(self):
        """Interpreters built without HAVE_MEMFD_CREATE still get a memfd.

        The libc path is the one that fixes the startup crash, so it is exercised
        even where os.memfd_create happens to exist.
        """
        fd = _memfd_create("sglang-engram-test", use_os=False)
        try:
            os.ftruncate(fd, 8192)
            self.assertEqual(os.fstat(fd).st_size, 8192)
        finally:
            os.close(fd)

    @unittest.skipUnless(
        hasattr(os, "memfd_create"), "this interpreter has no os.memfd_create"
    )
    def test_os_path_returns_a_usable_descriptor(self):
        fd = _memfd_create("sglang-engram-test", use_os=True)
        try:
            os.ftruncate(fd, 8192)
            self.assertEqual(os.fstat(fd).st_size, 8192)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
