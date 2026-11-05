"""Check allocator selection and ownership through actual CPU tensor views."""

import ctypes
import gc
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import npu_host_allocator as allocator
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestACLHostAllocator(CustomTestCase):
    def setUp(self):
        self.buffers = {}
        self.freed = []

        def allocate(output, size):
            buffer = ctypes.create_string_buffer(size)
            pointer = ctypes.addressof(buffer)
            self.buffers[pointer] = buffer
            output._obj.value = pointer
            return 0

        def free(pointer):
            self.freed.append(pointer.value)
            del self.buffers[pointer.value]
            return 0

        self.api = SimpleNamespace(
            aclrtMallocHost=Mock(side_effect=allocate),
            aclrtFreeHost=Mock(side_effect=free),
        )
        self.enterContext(
            patch.object(allocator, "_acl_host_api", return_value=self.api)
        )
        self.enterContext(
            patch.object(
                torch,
                "npu",
                SimpleNamespace(get_device_name=lambda: "Ascend910B4"),
                create=True,
            )
        )

    def test_view_keeps_allocation_alive_and_frees_once(self):
        tensor = allocator.allocate_acl_host((3, 5), torch.bfloat16, "cpu")
        pointer = tensor.data_ptr()
        tensor.fill_(7)
        view = tensor[1:]
        del tensor
        gc.collect()
        self.assertEqual(self.freed, [])
        self.assertTrue(torch.equal(view, torch.full((2, 5), 7, dtype=torch.bfloat16)))
        del view
        gc.collect()
        self.assertEqual(self.freed, [pointer])

    def test_exact_size_and_normal_entry_point(self):
        with patch.object(
            envs.SGLANG_NPU_HICACHE_HOST_ALLOCATOR, "get", return_value="aclrt"
        ):
            tensor = alloc_with_pin_memory((3, 5), torch.float32, "cpu", True, None)
        self.assertEqual(tensor.numel(), 15)
        self.assertEqual(self.api.aclrtMallocHost.call_args.args[1], 60)

    def test_empty_and_invalid_shape(self):
        self.assertEqual(
            allocator.allocate_acl_host((0, 3), torch.float32, "cpu").numel(), 0
        )
        self.api.aclrtMallocHost.assert_not_called()
        with self.assertRaises(ValueError):
            allocator.allocate_acl_host((-1, 3), torch.float32, "cpu")

    def test_native_failure_and_null_success(self):
        for result in (207001, 0):
            with self.subTest(result=result):
                self.api.aclrtMallocHost.side_effect = lambda pointer, size: result
                with self.assertRaises(RuntimeError):
                    allocator.allocate_acl_host((1,), torch.float32, "cpu")
        self.api.aclrtFreeHost.assert_not_called()

    def test_device_and_soc_guards(self):
        with self.assertRaises(ValueError):
            allocator.allocate_acl_host((1,), torch.float32, "npu")
        with patch.object(torch.npu, "get_device_name", return_value="Ascend910_9391"):
            with self.assertRaises(RuntimeError):
                allocator.allocate_acl_host((1,), torch.float32, "cpu")


if __name__ == "__main__":
    unittest.main()
