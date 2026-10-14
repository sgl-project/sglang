"""Typed CUDA host-allocation admission and cleanup mocks."""

import ctypes
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.weight_sync.gpu_delta import memory as memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class Driver:
    CUdevice_attribute = SimpleNamespace(
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=141,
        CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID=134,
    )
    CUpointer_attribute = SimpleNamespace(
        CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE=21
    )
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemAllocationHandleType = SimpleNamespace(CU_MEM_HANDLE_TYPE_NONE=0)
    CUmemLocationType = SimpleNamespace(
        CU_MEM_LOCATION_TYPE_HOST_NUMA=3, CU_MEM_LOCATION_TYPE_DEVICE=1
    )
    CUmemAllocationGranularity_flags = SimpleNamespace(
        CU_MEM_ALLOC_GRANULARITY_MINIMUM=0
    )
    CUmemAccess_flags = SimpleNamespace(
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3, CU_MEM_ACCESS_FLAGS_PROT_READ=1
    )
    CU_MEM_CREATE_USAGE_HW_DECOMPRESS = 2

    def __init__(self, failure=None):
        self.calls = []
        self.backing = ctypes.create_string_buffer(8192)
        self.failure = failure

    @staticmethod
    def CUmemAllocationProp():
        return SimpleNamespace(location=SimpleNamespace(), allocFlags=SimpleNamespace())

    @staticmethod
    def CUmemAccessDesc():
        return SimpleNamespace(location=SimpleNamespace())

    def cuDeviceGetAttribute(self, attribute, device):
        assert attribute in (141, 134) and device == 2
        return 0, 1 if attribute == 141 else 7

    def cuMemGetAllocationGranularity(self, properties, mode):
        assert mode == 0
        assert (
            properties.type,
            properties.requestedHandleTypes,
            properties.location.type,
        ) == (
            1,
            0,
            3,
        )
        assert properties.location.id == 7 and properties.allocFlags.usage == 2
        return 0, 4096

    def cuMemCreate(self, size, properties, flags):
        assert size == 8192 and flags == 0
        self.calls.append("create")
        return 0, 11

    def cuMemGetAllocationPropertiesFromHandle(self, handle):
        assert handle == 11
        properties = self.CUmemAllocationProp()
        properties.location.type = 3
        # CUDA may not echo HW_DECOMPRESS; the mapped pointer still must admit it.
        properties.allocFlags.usage = 0
        return 0, properties

    def cuMemAddressReserve(self, size, alignment, hint, flags):
        assert (size, alignment, hint, flags) == (8192, 0, 0, 0)
        self.calls.append("reserve")
        return 0, ctypes.addressof(self.backing)

    def cuMemMap(self, address, size, offset, handle, flags):
        self.calls.append("map")
        return (1 if self.failure == "map" else 0,)

    def cuMemSetAccess(self, address, size, descriptors, count):
        assert count == 2
        assert [(x.location.type, x.location.id, x.flags) for x in descriptors] == [
            (3, 0, 3),
            (1, 2, 1),
        ]
        self.calls.append("access")
        return (0,)

    def cuPointerGetAttribute(self, attribute, address):
        assert attribute == 21
        self.calls.append("capability")
        return 0, self.failure != "capability"

    def cuMemUnmap(self, address, size):
        self.calls.append("unmap")
        return (0,)

    def cuMemAddressFree(self, address, size):
        self.calls.append("free_address")
        return (0,)

    def cuMemRelease(self, handle):
        self.calls.append("release")
        return (0,)


@pytest.mark.parametrize("failure", [None, "map", "capability"])
def test_original_host_vmm_admission_and_cleanup(failure):
    driver = Driver(failure)
    with patch.object(memory, "_driver", return_value=driver):
        if failure:
            error = (
                "cuMemMap" if failure == "map" else "not hardware-decompression capable"
            )
            with pytest.raises(RuntimeError, match=error):
                memory.HostAllocation(4097, 2)
        else:
            allocation = memory.HostAllocation(4097, 2)
            assert allocation.capacity == 8192
            allocation.view[:4] = b"test"
            assert bytes(driver.backing[:4]) == b"test"
            allocation.close()
            allocation.close()
    expected = ["create", "reserve", "map"]
    if failure != "map":
        expected += ["access", "capability", "unmap"]
    assert driver.calls == expected + ["free_address", "release"]


def test_cuda_driver_initialization_is_cached_and_checks_status(monkeypatch):
    calls = []
    driver = SimpleNamespace(cuInit=lambda flags: (calls.append(flags) or 0,))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(driver=driver))
    memory._driver.cache_clear()
    try:
        assert memory._driver() is driver
        assert memory._driver() is driver
        assert calls == [0]
        memory._driver.cache_clear()
        driver.cuInit = lambda flags: (1,)
        with pytest.raises(RuntimeError, match="cuInit failed"):
            memory._driver()
        assert memory._driver.cache_info().currsize == 0
    finally:
        memory._driver.cache_clear()


@pytest.mark.parametrize("status,capable", [(0, 1), (0, 0), (1, 1)])
def test_unsupported_typed_pointer_attribute_still_requires_driver_capability(
    monkeypatch, status, capable
):
    driver = Driver()
    calls = []

    def unsupported(attribute, pointer):
        assert attribute == 21 and pointer == 0x123456780000
        raise TypeError(
            "Unsupported attribute: CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE"
        )

    def native(output, attribute, pointer):
        calls.append((attribute, pointer))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint))[0] = capable
        return status

    driver.cuPointerGetAttribute = unsupported
    monkeypatch.setattr(memory, "_driver", lambda: driver)
    monkeypatch.setattr(memory, "_pointer_attribute_query", lambda: native)
    if status or not capable:
        with pytest.raises(
            RuntimeError,
            match="CUDA 1" if status else "not hardware-decompression capable",
        ):
            memory.require_de_capable(0x123456780000)
    else:
        memory.require_de_capable(0x123456780000)
    assert calls == [(21, 0x123456780000)]


@pytest.mark.parametrize("failure", [TypeError("invalid pointer"), (1, 1)])
def test_pointer_attribute_fallback_does_not_mask_other_failures(monkeypatch, failure):
    driver = Driver()

    def query(attribute, pointer):
        if isinstance(failure, Exception):
            raise failure
        return failure

    driver.cuPointerGetAttribute = query
    monkeypatch.setattr(memory, "_driver", lambda: driver)
    monkeypatch.setattr(
        memory, "_pointer_attribute_query", lambda: pytest.fail("unexpected fallback")
    )
    with pytest.raises((TypeError, RuntimeError), match="invalid pointer|CUDA 1"):
        memory.require_de_capable(0x123456780000)


def test_pointer_attribute_native_symbol_uses_cuda_abi_and_is_cached(monkeypatch):
    query = SimpleNamespace()
    libraries = []

    def load(name):
        libraries.append(name)
        return SimpleNamespace(cuPointerGetAttribute=query)

    memory._pointer_attribute_query.cache_clear()
    monkeypatch.setattr(memory.ctypes, "CDLL", load)
    try:
        assert memory._pointer_attribute_query() is query
        assert memory._pointer_attribute_query() is query
        assert libraries == ["libcuda.so.1"]
        assert query.argtypes == [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        assert query.restype is ctypes.c_int
    finally:
        memory._pointer_attribute_query.cache_clear()


class DeviceDriver:
    def __init__(self, events, context=13, device=2):
        self.events, self.context, self.device = events, context, device

    def cuCtxGetCurrent(self):
        return 0, self.context

    def cuCtxGetDevice(self):
        return 0, self.device

    def cuMemAlloc(self, size):
        self.events.append(("alloc", size))
        return 0, 4096

    def cuCtxPushCurrent(self, context):
        self.events.append(("push", context))
        return (0,)

    def cuMemFree(self, address):
        self.events.append(("free", address))
        return (0,)

    def cuCtxPopCurrent(self):
        self.events.append("pop")
        return 0, self.context


def test_device_owner_fences_both_streams_and_restores_context(monkeypatch):
    import torch

    events = []
    driver = DeviceDriver(events)
    stream = SimpleNamespace(
        cuda_stream=5, synchronize=lambda: events.append("producer")
    )
    monkeypatch.setattr(memory, "_driver", lambda: driver)
    monkeypatch.setattr(
        memory, "require_de_capable", lambda p: events.append(("admit", p))
    )
    owner = memory._DeviceAllocation(123, torch.device("cuda:2"), stream)
    assert owner.__cuda_array_interface__ == {
        "shape": (123,),
        "typestr": "|u1",
        "data": (4096, False),
        "version": 3,
    }
    tensor = SimpleNamespace(
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 4096)
    )
    reader = SimpleNamespace(
        device=torch.device("cuda:2"),
        cuda_stream=6,
        synchronize=lambda: events.append("reader"),
    )
    memory.record_device_stream(tensor, reader)
    memory.record_device_stream(tensor, reader)
    owner._release()
    owner._release()
    assert events == [
        ("alloc", 123),
        ("admit", 4096),
        ("push", 13),
        "producer",
        "reader",
        ("free", 4096),
        "pop",
    ]
    assert owner.address == 0


@pytest.mark.parametrize(
    "context,device,error",
    [(0, 2, "current context"), (13, 1, "context/device mismatch")],
)
def test_device_owner_rejects_wrong_context_before_allocation(
    monkeypatch, context, device, error
):
    import torch

    events = []
    monkeypatch.setattr(
        memory, "_driver", lambda: DeviceDriver(events, context, device)
    )
    with pytest.raises(RuntimeError, match=error):
        memory._DeviceAllocation(
            123, torch.device("cuda:2"), SimpleNamespace(cuda_stream=5)
        )
    assert events == []


def test_device_owner_admission_failure_frees_and_failed_fence_never_frees(monkeypatch):
    import torch

    events = []
    monkeypatch.setattr(memory, "_driver", lambda: DeviceDriver(events))
    stream = SimpleNamespace(cuda_stream=5, synchronize=lambda: events.append("fence"))

    def reject(pointer):
        raise RuntimeError("not DE capable")

    monkeypatch.setattr(memory, "require_de_capable", reject)
    with pytest.raises(RuntimeError, match="not DE capable"):
        memory._DeviceAllocation(123, torch.device("cuda:2"), stream)
    assert events == [("alloc", 123), ("push", 13), "fence", ("free", 4096), "pop"]
    events.clear()
    monkeypatch.setattr(memory, "require_de_capable", lambda _: None)
    owner = memory._DeviceAllocation(123, torch.device("cuda:2"), stream)

    def failed_fence():
        events.append("failed fence")
        raise RuntimeError("stream incomplete")

    stream.synchronize = failed_fence
    with pytest.raises(RuntimeError, match="stream incomplete"):
        owner._release()
    owner._release()  # No uncertain cleanup retry or premature free.
    assert events == [("alloc", 123), ("push", 13), "failed fence", "pop"]


def test_device_empty_validates_shape_dtype_device_and_handles_zero(monkeypatch):
    import torch

    monkeypatch.setattr(memory, "_driver", lambda: pytest.fail("driver called"))
    for shape, dtype, device, message in [
        ((1,), torch.uint8, "cpu", "explicit CUDA"),
        ((1,), torch.uint8, "cuda", "explicit CUDA"),
        ((-1,), torch.uint8, "cuda:0", "nonnegative"),
        ((1,), torch.float32, "cuda:0", "dtype"),
    ]:
        with pytest.raises(ValueError, match=message):
            memory.device_empty(shape, dtype=dtype, device=device)
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        torch, "empty", lambda shape, **kw: calls.append((shape, kw)) or sentinel
    )
    assert memory.device_empty((0, 4), dtype=torch.int64, device="cuda:0") is sentinel
    assert calls == [((0, 4), {"dtype": torch.int64, "device": torch.device("cuda:0")})]
