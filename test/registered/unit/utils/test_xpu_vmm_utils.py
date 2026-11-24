# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for srt/utils/xpu_vmm_utils.

The hardware round trip lives in test/registered/xpu/test_xpu_vmm_utils_xpu.py.
Everything here is the part that must hold without a Level Zero device: argument
validation that runs before the driver is touched, result decoding, and the
capability probe's answer when the loader is absent.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.utils import xpu_vmm_utils as vmm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestAlignment(CustomTestCase):
    def test_align_up_rounds_to_the_next_multiple(self):
        self.assertEqual(vmm.align_up(0, 4096), 0)
        self.assertEqual(vmm.align_up(1, 4096), 4096)
        self.assertEqual(vmm.align_up(4096, 4096), 4096)
        self.assertEqual(vmm.align_up(4097, 4096), 8192)
        self.assertEqual(vmm.align_up(100, 65536), 65536)

    def test_align_up_rejects_a_nonpositive_alignment(self):
        for alignment in (0, -4096):
            with self.assertRaises(ValueError):
                vmm.align_up(1, alignment)


class TestSizeTBounds(CustomTestCase):
    """ctypes truncates an out-of-range size_t silently and the driver has been
    seen to abort on the result, so the bound is checked in Python."""

    def test_values_inside_size_t_pass_through(self):
        for value in (0, 1, 1 << 20, vmm._SIZE_T_MAX):
            self.assertEqual(vmm._as_size_t("size", value), value)

    def test_values_outside_size_t_are_rejected(self):
        for value in (vmm._SIZE_T_MAX + 1, 1 << 64, 1 << 128, -1):
            with self.assertRaises(ValueError):
                vmm._as_size_t("size", value)


class TestResultDecoding(CustomTestCase):
    def test_known_results_decode_to_their_spec_names(self):
        self.assertEqual(vmm.result_name(0x78000003), "ERROR_UNSUPPORTED_FEATURE")
        self.assertEqual(vmm.result_name(0x70000003), "ERROR_OUT_OF_DEVICE_MEMORY")
        self.assertEqual(vmm.result_name(0x7800000A), "ERROR_UNSUPPORTED_ALIGNMENT")

    def test_an_unlisted_result_is_named_rather_than_dropped(self):
        self.assertEqual(vmm.result_name(0x12345678), "UNRECOGNISED_ZE_RESULT")

    def test_check_passes_on_success_and_raises_otherwise(self):
        vmm.check("zeNothing", vmm.ZE_RESULT_SUCCESS)
        with self.assertRaises(vmm.ZeVmmError) as caught:
            vmm.check("zeVirtualMemMap", 0x7800000A)
        message = str(caught.exception)
        self.assertIn("zeVirtualMemMap", message)
        self.assertIn("0x7800000a", message)
        self.assertIn("ERROR_UNSUPPORTED_ALIGNMENT", message)


class TestDmaBufIdentity(CustomTestCase):
    def test_identity_reads_dev_inode_and_size_from_the_fd(self):
        read_fd, write_fd = os.pipe()
        try:
            identity = vmm.dma_buf_identity(read_fd)
            expected = os.fstat(read_fd)
            self.assertEqual(identity.st_ino, expected.st_ino)
            self.assertEqual(identity.st_dev, expected.st_dev)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_two_fds_on_one_object_share_an_identity(self):
        read_fd, write_fd = os.pipe()
        duplicate = os.dup(read_fd)
        try:
            self.assertEqual(
                vmm.dma_buf_identity(read_fd), vmm.dma_buf_identity(duplicate)
            )
        finally:
            for fd in (read_fd, write_fd, duplicate):
                os.close(fd)

    def test_identity_rejects_a_negative_fd(self):
        with self.assertRaises(ValueError):
            vmm.dma_buf_identity(-1)


class TestValidationBeforeTheDriverIsTouched(CustomTestCase):
    """These must raise without a device, so they run on CPU CI.

    Each asserts the driver was never loaded: a size check that happens after
    zeInit would make the whole module unusable on a CPU-only host.
    """

    def setUp(self):
        patcher = patch.object(
            vmm._driver, "load", side_effect=AssertionError("driver was loaded")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_query_page_size_rejects_a_nonpositive_size(self):
        for size in (0, -1):
            with self.assertRaises(ValueError):
                vmm.query_page_size(size)

    def test_query_page_size_rejects_a_size_beyond_size_t(self):
        with self.assertRaises(ValueError):
            vmm.query_page_size(1 << 64)

    def test_physical_create_rejects_a_nonpositive_size(self):
        for size in (0, -4096):
            with self.assertRaises(ValueError):
                vmm.PhysicalMemory.create(size)

    def test_physical_import_rejects_a_negative_fd(self):
        with self.assertRaises(ValueError):
            vmm.PhysicalMemory.from_dma_buf(-1, 65536)

    def test_physical_import_rejects_a_nonpositive_size(self):
        with self.assertRaises(ValueError):
            vmm.PhysicalMemory.from_dma_buf(0, 0)

    def test_reservation_rejects_a_nonpositive_size(self):
        for size in (0, -65536):
            with self.assertRaises(ValueError):
                vmm.VirtualReservation(size)


class TestCapabilityProbe(CustomTestCase):
    def test_a_missing_loader_is_reported_not_raised(self):
        with patch.object(
            vmm._driver, "load", side_effect=OSError("libze_loader.so.1 not found")
        ):
            reason = vmm.vmm_unsupported_reason()
            self.assertIsNotNone(reason)
            self.assertIn("libze_loader", reason)
            self.assertFalse(vmm.is_vmm_supported())

    def test_a_driver_without_vmm_is_reported_not_raised(self):
        unsupported = vmm.ZeVmmError(
            "zeVirtualMemQueryPageSize failed with ze_result_t 0x78000003 "
            "(ERROR_UNSUPPORTED_FEATURE)"
        )
        with patch.object(vmm, "query_page_size", side_effect=unsupported):
            reason = vmm.vmm_unsupported_reason()
            self.assertIsNotNone(reason)
            self.assertIn("ERROR_UNSUPPORTED_FEATURE", reason)
            self.assertFalse(vmm.is_vmm_supported())


class TestCopyValidation(CustomTestCase):
    def setUp(self):
        patcher = patch.object(
            vmm._driver, "load", side_effect=AssertionError("driver was loaded")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_copy_rejects_a_null_pointer(self):
        with self.assertRaises(ValueError):
            vmm.copy_to_device(0, b"abcd")

    def test_copy_rejects_an_empty_payload(self):
        with self.assertRaises(ValueError):
            vmm.copy_to_device(0x1000, b"")

    def test_copy_from_device_rejects_a_zero_length(self):
        with self.assertRaises(ValueError):
            vmm.copy_from_device(0x1000, 0)


if __name__ == "__main__":
    unittest.main()
