import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.hardware_backend.npu import host_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNpuHostMemory(unittest.TestCase):
    PAGE_SIZE = host_memory.mmap.PAGESIZE
    HOST_PTR = PAGE_SIZE + 0x234
    NBYTES = PAGE_SIZE

    def test_borrows_existing_mapping_without_unregistering_it(self):
        acl_rt = SimpleNamespace(
            host_get_device_pointer=mock.Mock(return_value=(0x900000, 0)),
            host_register=mock.Mock(),
            host_unregister=mock.Mock(),
        )
        with mock.patch.object(host_memory, "_load_acl_rt", return_value=acl_rt):
            mapping = host_memory.register_host_memory(self.HOST_PTR, self.NBYTES)
            host_memory.unregister_host_memory(mapping)

        self.assertEqual(mapping.device_ptr, 0x900234)
        self.assertEqual(mapping.registered_host_ptr, self.PAGE_SIZE)
        self.assertFalse(mapping.owned)
        acl_rt.host_get_device_pointer.assert_called_once_with(self.PAGE_SIZE, 0)
        acl_rt.host_register.assert_not_called()
        acl_rt.host_unregister.assert_not_called()

    def test_registers_aligned_range_and_unregisters_owned_mapping(self):
        acl_rt = SimpleNamespace(
            host_get_device_pointer=mock.Mock(return_value=(None, 1)),
            host_register=mock.Mock(return_value=(0xA00000, 0)),
            host_unregister=mock.Mock(return_value=0),
        )
        with mock.patch.object(host_memory, "_load_acl_rt", return_value=acl_rt):
            mapping = host_memory.register_host_memory(self.HOST_PTR, self.NBYTES)
            host_memory.unregister_host_memory(mapping)

        self.assertEqual(mapping.device_ptr, 0xA00234)
        self.assertEqual(mapping.registered_host_ptr, self.PAGE_SIZE)
        self.assertTrue(mapping.owned)
        acl_rt.host_register.assert_called_once_with(
            self.PAGE_SIZE,
            2 * self.PAGE_SIZE,
            host_memory.ACL_HOST_REGISTER_MAPPED,
        )
        acl_rt.host_unregister.assert_called_once_with(self.PAGE_SIZE)

    def test_borrows_mapping_created_during_register_race(self):
        acl_rt = SimpleNamespace(
            host_get_device_pointer=mock.Mock(side_effect=[(None, 1), (0xB00000, 0)]),
            host_register=mock.Mock(return_value=(None, 507910)),
            host_unregister=mock.Mock(),
        )
        with mock.patch.object(host_memory, "_load_acl_rt", return_value=acl_rt):
            mapping = host_memory.register_host_memory(self.HOST_PTR, self.NBYTES)
            host_memory.unregister_host_memory(mapping)

        self.assertEqual(mapping.device_ptr, 0xB00234)
        self.assertFalse(mapping.owned)
        acl_rt.host_unregister.assert_not_called()


if __name__ == "__main__":
    unittest.main()
