import builtins
import ctypes
import gc
import importlib
import sys
import types
from enum import Enum

import pytest
import torch


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


def install_fake_mori(monkeypatch: pytest.MonkeyPatch):
    fake_umbp = types.ModuleType("mori.umbp")
    fake_umbp.UMBPHostBufferBacking = FakeBacking
    fake_umbp.UMBPHostBufferHandle = FakeHandle
    fake_umbp.UMBPHostMemAllocator = FakeHostMemAllocator

    fake_mori = types.ModuleType("mori")
    fake_mori.__path__ = []
    fake_mori.umbp = fake_umbp

    monkeypatch.setitem(sys.modules, "mori", fake_mori)
    monkeypatch.setitem(sys.modules, "mori.umbp", fake_umbp)
    return fake_umbp


def test_umbp_allocator_dispatch_and_tensor_wrap(monkeypatch: pytest.MonkeyPatch):
    install_fake_mori(monkeypatch)

    from sglang.srt.mem_cache.memory_pool_host import get_allocator_from_storage
    from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
        UMBPHostTensorAllocator,
    )

    allocator = get_allocator_from_storage("umbp")
    assert isinstance(allocator, UMBPHostTensorAllocator)

    tensor = allocator.allocate((2, 3), dtype=torch.float16, device="cpu")
    alloc_call = allocator._allocator.alloc_calls[0]

    assert tensor.shape == (2, 3)
    assert tensor.dtype == torch.float16
    assert tensor.data_ptr() == alloc_call["handle"].ptr
    assert alloc_call["size"] == tensor.numel() * tensor.element_size()
    assert alloc_call["backing"] == FakeBacking.AnonymousHugetlb
    assert alloc_call["hugepage_size"] == 2 * 1024 * 1024
    assert alloc_call["numa_node"] == -1
    assert alloc_call["prefault"] is True

    tensor.fill_(3.0)
    assert float(tensor[0, 0]) == 3.0


def test_umbp_allocator_del_calls_free_once(monkeypatch: pytest.MonkeyPatch):
    install_fake_mori(monkeypatch)

    module = importlib.import_module(
        "sglang.srt.mem_cache.storage.umbp.umbp_host_allocator"
    )
    allocator = module.UMBPHostTensorAllocator()
    tensor = allocator.allocate((16,), dtype=torch.uint8, device="cpu")

    del tensor
    gc.collect()

    fake_allocator = allocator._allocator
    handle = allocator._handle
    allocator.__del__()

    assert len(fake_allocator.free_calls) == 1
    assert fake_allocator.free_calls[0] is handle
    assert handle.ptr is None
    assert handle.requested_size == 0
    assert handle.mapped_size == 0

    allocator.__del__()
    assert len(fake_allocator.free_calls) == 1


def test_get_allocator_from_storage_umbp_falls_back(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mori" or name.startswith("mori."):
            raise ImportError("mori unavailable in test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "mori", raising=False)
    monkeypatch.delitem(sys.modules, "mori.umbp", raising=False)

    from sglang.srt.mem_cache.memory_pool_host import (
        HostTensorAllocator,
        get_allocator_from_storage,
    )

    allocator = get_allocator_from_storage("umbp")
    assert type(allocator) is HostTensorAllocator
    assert "UMBPHostTensorAllocator unavailable" in caplog.text
