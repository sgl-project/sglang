"""CPU-only coverage for exact-size paged-expert backing without Python's memfd binding."""

import ctypes
import errno
import fcntl
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.paged_experts import store
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HAS_LIBC_MEMFD = sys.platform == "linux" and hasattr(ctypes.CDLL(None), "memfd_create")


class TestMemfd(CustomTestCase):
    def test_prefers_native_binding(self):
        native = Mock(return_value=123)
        with (
            patch.dict(os.__dict__, memfd_create=native),
            patch.object(store, "_libc") as libc,
        ):
            self.assertEqual(store._memfd_create("experts"), 123)
            native.assert_called_once_with("experts", getattr(os, "MFD_CLOEXEC", 1))
            libc.assert_not_called()

    def test_native_error_is_propagated(self):
        failure = OSError(errno.EMFILE, "descriptor limit")
        with (
            patch.dict(os.__dict__, memfd_create=Mock(side_effect=failure)),
            patch.object(store, "_libc") as libc,
        ):
            with self.assertRaises(OSError) as raised:
                store._memfd_create("experts")
            self.assertIs(raised.exception, failure)
            libc.assert_not_called()

    @unittest.skipUnless(HAS_LIBC_MEMFD, "requires libc.memfd_create")
    def test_missing_binding_and_constant_use_a_cloexec_memfd(self):
        before = len(os.listdir("/proc/self/fd"))
        with patch.dict(os.__dict__):
            os.__dict__.pop("memfd_create", None)
            os.__dict__.pop("MFD_CLOEXEC", None)
            fd = store._memfd_create("experts")
            try:
                os.ftruncate(fd, 4096)
                os.pwrite(fd, b"expert", 0)
                self.assertEqual(os.pread(fd, 6, 0), b"expert")
                self.assertEqual(os.fstat(fd).st_size, 4096)
                self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)
            finally:
                os.close(fd)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    @unittest.skipUnless(HAS_LIBC_MEMFD, "requires libc.memfd_create")
    def test_libc_declares_the_memfd_prototype_once(self):
        create = store._libc().memfd_create
        self.assertEqual(create.argtypes, (ctypes.c_char_p, ctypes.c_uint))
        self.assertIs(create.restype, ctypes.c_int)
        self.assertIs(store._libc().memfd_create, create)

    def test_libc_errno_is_propagated(self):
        def denied(name, flags):
            ctypes.set_errno(errno.EPERM)
            return -1

        create = Mock(side_effect=denied)
        with (
            patch.dict(os.__dict__),
            patch.object(
                store, "_libc", return_value=SimpleNamespace(memfd_create=create)
            ),
        ):
            os.__dict__.pop("memfd_create", None)
            with self.assertRaises(OSError) as raised:
                store._memfd_create("experts")
            self.assertEqual(raised.exception.errno, errno.EPERM)
            create.assert_called_once_with(b"experts", getattr(os, "MFD_CLOEXEC", 1))

    def test_unavailable_symbol_reports_enosys(self):
        with (
            patch.dict(os.__dict__),
            patch.object(store, "_libc", return_value=SimpleNamespace()),
        ):
            os.__dict__.pop("memfd_create", None)
            with self.assertRaises(OSError) as raised:
                store._memfd_create("experts")
            self.assertEqual(raised.exception.errno, errno.ENOSYS)

    @unittest.skipUnless(HAS_LIBC_MEMFD, "requires libc.memfd_create")
    def test_failed_resize_closes_the_descriptor(self):
        fd = store._memfd_create("experts")
        with (
            patch.object(store, "_memfd_create", return_value=fd),
            patch.object(os, "ftruncate", side_effect=OSError(errno.ENOSPC, "full")),
        ):
            with self.assertRaises(OSError) as raised:
                store._pin_until_freed((4096,), torch.uint8)
            self.assertEqual(raised.exception.errno, errno.ENOSPC)
        with self.assertRaises(OSError) as closed:
            os.fstat(fd)
        self.assertEqual(closed.exception.errno, errno.EBADF)


if __name__ == "__main__":
    unittest.main()
