import builtins
import ctypes
import gc
import importlib
import sys
import types
import unittest
from enum import Enum
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# These tests stub out mori with a fake in-process module, so they need neither
# a real mori install nor a GPU and run on NVIDIA / CPU CI.


class FakeBacking(Enum):
    Anonymous = 0
    AnonymousHugetlb = 1


class FakeHandle:
    def __init__(
        self,
        ptr: int,
        requested_size: int,
        mapped_size: int,
        actual_backing: FakeBacking,
        actual_alignment: int,
    ) -> None:
        self.ptr = ptr
        self.requested_size = requested_size
        self.mapped_size = mapped_size
        self.actual_backing = actual_backing
        self.actual_alignment = actual_alignment

    def __bool__(self) -> bool:
        return self.ptr is not None


class FakeHostMemAllocator:
    def __init__(self) -> None:
        self.alloc_calls = []
        self.free_calls = []
        self._buffers = []

    def alloc(
        self,
        size: int,
        backing: FakeBacking,
        hugepage_size: int,
        numa_node: int,
        prefault: bool,
    ) -> FakeHandle:
        buf = (ctypes.c_byte * size)()
        self._buffers.append(buf)
        handle = FakeHandle(
            ptr=ctypes.addressof(buf),
            requested_size=size,
            mapped_size=size,
            actual_backing=backing,
            actual_alignment=(
                hugepage_size if backing == FakeBacking.AnonymousHugetlb else 4096
            ),
        )
        self.alloc_calls.append(
            {
                "size": size,
                "backing": backing,
                "hugepage_size": hugepage_size,
                "numa_node": numa_node,
                "prefault": prefault,
                "handle": handle,
            }
        )
        return handle

    def free(self, handle: FakeHandle) -> None:
        self.free_calls.append(handle)
        handle.ptr = None
        handle.requested_size = 0
        handle.mapped_size = 0


class TestUMBPHostAllocator(unittest.TestCase):
    def _save_mori_modules(self):
        """Snapshot and restore sys.modules entries for mori on cleanup."""
        saved = {name: sys.modules.get(name) for name in ("mori", "mori.umbp")}

        def restore():
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        self.addCleanup(restore)

    def _install_fake_mori(self):
        self._save_mori_modules()

        fake_umbp = types.ModuleType("mori.umbp")
        fake_umbp.UMBPHostBufferBacking = FakeBacking
        fake_umbp.UMBPHostBufferHandle = FakeHandle
        fake_umbp.UMBPHostMemAllocator = FakeHostMemAllocator

        fake_mori = types.ModuleType("mori")
        fake_mori.__path__ = []
        fake_mori.umbp = fake_umbp

        sys.modules["mori"] = fake_mori
        sys.modules["mori.umbp"] = fake_umbp
        return fake_umbp

    def test_umbp_allocator_dispatch_and_tensor_wrap(self):
        self._install_fake_mori()

        from sglang.srt.mem_cache.memory_pool_host import get_allocator_from_storage
        from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
            UMBPHostTensorAllocator,
        )

        allocator = get_allocator_from_storage("mori")
        self.assertIsInstance(allocator, UMBPHostTensorAllocator)

        tensor = allocator.allocate((2, 3), dtype=torch.float16, device="cpu")
        alloc_call = allocator._allocator.alloc_calls[0]

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.dtype, torch.float16)
        self.assertEqual(tensor.data_ptr(), alloc_call["handle"].ptr)
        self.assertEqual(alloc_call["size"], tensor.numel() * tensor.element_size())
        self.assertEqual(alloc_call["backing"], FakeBacking.AnonymousHugetlb)
        self.assertEqual(alloc_call["hugepage_size"], 2 * 1024 * 1024)
        self.assertEqual(alloc_call["numa_node"], -1)
        self.assertIs(alloc_call["prefault"], True)

        tensor.fill_(3.0)
        self.assertEqual(float(tensor[0, 0]), 3.0)

    def test_umbp_allocator_del_calls_free_once(self):
        self._install_fake_mori()

        module = importlib.import_module(
            "sglang.srt.mem_cache.storage.umbp.umbp_host_allocator"
        )
        allocator = module.UMBPHostTensorAllocator()
        tensor = allocator.allocate((16,), dtype=torch.uint8, device="cpu")

        del tensor
        gc.collect()

        fake_allocator = allocator._allocator
        handles = list(allocator._handles.values())
        self.assertEqual(len(handles), 1)
        handle = handles[0]
        allocator.__del__()

        self.assertEqual(len(fake_allocator.free_calls), 1)
        self.assertIs(fake_allocator.free_calls[0], handle)
        self.assertIsNone(handle.ptr)
        self.assertEqual(handle.requested_size, 0)
        self.assertEqual(handle.mapped_size, 0)

        allocator.__del__()
        self.assertEqual(len(fake_allocator.free_calls), 1)

    def test_get_allocator_from_storage_umbp_falls_back(self):
        self._save_mori_modules()
        sys.modules.pop("mori", None)
        sys.modules.pop("mori.umbp", None)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mori" or name.startswith("mori."):
                raise ImportError("mori unavailable in test")
            return real_import(name, globals, locals, fromlist, level)

        from sglang.srt.mem_cache.pool_host.common import HostTensorAllocator

        with mock.patch.object(builtins, "__import__", fake_import):
            with self.assertLogs(level="WARNING") as cm:
                from sglang.srt.mem_cache.memory_pool_host import (
                    get_allocator_from_storage,
                )

                allocator = get_allocator_from_storage("mori")

        self.assertIs(type(allocator), HostTensorAllocator)
        self.assertTrue(
            any("UMBPHostTensorAllocator unavailable" in msg for msg in cm.output),
            f"missing fallback warning in logs: {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
