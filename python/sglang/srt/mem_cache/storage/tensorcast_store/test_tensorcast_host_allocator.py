# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

# ruff: noqa: E402

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass

import torch

from sglang.test.mem_cache.host_pool_test_utils import install_memory_pool_host_stub


class _FakeHostTensorAllocator:
    def __init__(self) -> None:
        self.dtype = None
        self.dims = None


install_memory_pool_host_stub(host_tensor_allocator_cls=_FakeHostTensorAllocator)

from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHostAllocatorConfig,
    tensorcast_host_allocator_config_from_extra_config,
)
from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
    TensorcastHostTensorAllocator,
)


@dataclass
class _FakeRegionHandle:
    region_id: str


@dataclass
class _FakeAttachment:
    fd: int
    size_bytes: int


class _FakeTensorcastStore:
    def __init__(self) -> None:
        self.register_calls: list[dict[str, object]] = []
        self.release_calls: list[str] = []
        self.unregister_calls: list[str] = []
        self._file_by_region_id: dict[str, tempfile._TemporaryFileWrapper | object] = {}

    def register_region(self, **kwargs):
        region_id = f"region-{len(self.register_calls)}"
        self.register_calls.append(dict(kwargs))
        file_obj = tempfile.TemporaryFile()
        file_obj.truncate(int(kwargs["size_bytes"]))
        self._file_by_region_id[region_id] = file_obj
        return _FakeRegionHandle(region_id=region_id)

    def attach_host_shared_region(self, handle: _FakeRegionHandle):
        file_obj = self._file_by_region_id[handle.region_id]
        fd = int(file_obj.fileno())
        return _FakeAttachment(fd=os.dup(fd), size_bytes=os.fstat(fd).st_size)

    def release_host_shared_region(self, handle: _FakeRegionHandle) -> None:
        self.release_calls.append(handle.region_id)

    def unregister_region(self, region_id: str, *, force: bool | None = None) -> bool:
        self.unregister_calls.append(region_id)
        file_obj = self._file_by_region_id.pop(region_id, None)
        if file_obj is not None:
            file_obj.close()
        return True


class _FakeHostRegistrationOps:
    def __init__(self, *, available: bool = True, fail_register: bool = False) -> None:
        self._available = available
        self._fail_register = fail_register
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def register(self, ptr: int, size_bytes: int) -> None:
        self.register_calls.append((ptr, size_bytes))
        if self._fail_register:
            raise RuntimeError("synthetic register failure")

    def unregister(self, ptr: int) -> None:
        self.unregister_calls.append(ptr)


class TensorcastHostAllocatorTest(unittest.TestCase):
    def test_allocator_config_is_opt_in(self) -> None:
        self.assertIsNone(
            tensorcast_host_allocator_config_from_extra_config(
                {
                    "daemon_address": "unix:///tmp/tensorcast.sock",
                }
            )
        )
        config = tensorcast_host_allocator_config_from_extra_config(
            {
                "host_allocator_enabled": True,
                "daemon_address": "unix:///tmp/tensorcast.sock",
                "host_allocator_region_ttl_ms": 3210,
                "host_allocator_region_name": "rank0-host-pool",
            }
        )
        self.assertEqual(
            config,
            TensorcastHostAllocatorConfig(
                daemon_address="unix:///tmp/tensorcast.sock",
                region_ttl_ms=3210,
                region_name="rank0-host-pool",
            ),
        )

    def test_allocator_exports_allocator_backed_host_shared_slab(self) -> None:
        fake_store = _FakeTensorcastStore()
        host_registration_ops = _FakeHostRegistrationOps()
        allocator = TensorcastHostTensorAllocator(
            TensorcastHostAllocatorConfig(
                daemon_address="unix:///tmp/tensorcast.sock",
                region_ttl_ms=1234,
                region_name="rank0-host-pool",
            ),
            store_factory=lambda _: fake_store,
            host_registration_ops=host_registration_ops,
        )

        tensor = allocator.allocate((2, 4), torch.float32, "cpu")
        allocator.ensure_host_registration(tensor, pin_memory=True)

        self.assertEqual(tensor.shape, (2, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(len(fake_store.register_calls), 1)
        register_call = fake_store.register_calls[0]
        self.assertEqual(register_call["size_bytes"], 2 * 4 * 4)
        self.assertTrue(bool(register_call["daemon_managed"]))
        self.assertEqual(register_call["ttl_ms"], 1234)
        self.assertEqual(register_call["name"], "rank0-host-pool")
        self.assertEqual(
            str(register_call["memory_kind"]), "RegionMemoryKind.HOST_SHARED"
        )
        self.assertEqual(
            str(register_call["host_shared_region_class"]),
            "HostSharedRegionClass.ALLOCATOR",
        )
        binding = allocator.binding
        self.assertIsNotNone(binding)
        self.assertEqual(binding.capacity_bytes, 32)
        self.assertEqual(binding.region_name, "rank0-host-pool")
        self.assertEqual(binding.base_ptr, tensor.data_ptr())
        self.assertEqual(
            host_registration_ops.register_calls, [(tensor.data_ptr(), 32)]
        )

        tensor.fill_(3.0)
        self.assertTrue(torch.equal(tensor, torch.full((2, 4), 3.0)))

        allocator.close()
        self.assertEqual(host_registration_ops.unregister_calls, [tensor.data_ptr()])
        self.assertEqual(fake_store.release_calls, ["region-0"])
        self.assertEqual(fake_store.unregister_calls, ["region-0"])

    def test_allocator_host_registration_falls_back_cleanly(self) -> None:
        fake_store = _FakeTensorcastStore()
        host_registration_ops = _FakeHostRegistrationOps(fail_register=True)
        allocator = TensorcastHostTensorAllocator(
            TensorcastHostAllocatorConfig(
                daemon_address="unix:///tmp/tensorcast.sock",
                region_ttl_ms=1234,
                region_name="rank0-host-pool",
            ),
            store_factory=lambda _: fake_store,
            host_registration_ops=host_registration_ops,
        )

        tensor = allocator.allocate((2, 4), torch.float32, "cpu")
        allocator.ensure_host_registration(tensor, pin_memory=True)

        self.assertEqual(
            host_registration_ops.register_calls, [(tensor.data_ptr(), 32)]
        )
        self.assertEqual(host_registration_ops.unregister_calls, [])
        allocator.close()
        self.assertEqual(host_registration_ops.unregister_calls, [])
        self.assertEqual(fake_store.release_calls, ["region-0"])
        self.assertEqual(fake_store.unregister_calls, ["region-0"])


if __name__ == "__main__":
    unittest.main()
