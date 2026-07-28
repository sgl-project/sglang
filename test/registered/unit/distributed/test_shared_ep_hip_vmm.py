"""CPU-only tests for the ROCm SharedEP VMM facade."""

import ctypes
import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from sglang.srt.distributed.device_communicators import hip_vmm, vmm_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# SharedEP's public package imports its GPU compute backend eagerly. Bypass that
# package initializer so these facade tests remain genuinely CPU-only.
_SHARED_EP_PACKAGE = "sglang.srt.layers.moe.shared_ep"
_created_package_stub = False
if _SHARED_EP_PACKAGE not in sys.modules:
    package = types.ModuleType(_SHARED_EP_PACKAGE)
    srt_root = Path(vmm_utils.__file__).resolve().parents[2]
    package.__path__ = [str(srt_root / "layers" / "moe" / "shared_ep")]
    sys.modules[_SHARED_EP_PACKAGE] = package
    _created_package_stub = True
vmm = importlib.import_module(f"{_SHARED_EP_PACKAGE}.vmm")
epoch = importlib.import_module(f"{_SHARED_EP_PACKAGE}.epoch")
if _created_package_stub:
    del sys.modules[_SHARED_EP_PACKAGE]


def test_pointer_table_epoch_api_is_rocm_only_and_validates_gpu_views():
    create = epoch.create_gpu_pointer_table_epoch
    with (
        patch.object(epoch, "_IS_HIP", False),
        pytest.raises(RuntimeError, match="ROCm"),
    ):
        create(peer_signals=[], rank=0)

    with (
        patch.object(epoch, "_IS_HIP", True),
        pytest.raises(ValueError, match="GPU tensors"),
    ):
        create(peer_signals=[torch.zeros(4, dtype=torch.uint8)], rank=0)

    source = inspect.getsource(create)
    assert "dtype=torch.uint64" in source
    assert "signals.data_ptr()" in source


def test_hip_allocation_properties_are_uncached_and_posix_shareable():
    prop = hip_vmm._allocation_prop(7)

    assert prop.type == hip_vmm._HIP_MEM_ALLOCATION_TYPE_UNCACHED
    assert prop.requested_handle_type == hip_vmm._HIP_MEM_HANDLE_TYPE_POSIX_FD
    assert prop.location.type == hip_vmm._HIP_MEM_LOCATION_TYPE_DEVICE
    assert prop.location.id == 7
    assert ctypes.sizeof(prop) == 32


def test_hip_capability_probe_exercises_full_primitive_chain_and_cleans_up():
    runtime = MagicMock()
    runtime.runtime_version.return_value = 70_200_000
    runtime.allocation_granularity.return_value = 65_536
    runtime.create.return_value = 0xA11
    runtime.import_fd.return_value = 0xB22
    runtime.reserve.return_value = 0xC000
    exported_fd, other_end = os.pipe()
    runtime.export_fd.return_value = exported_fd
    backend = hip_vmm.HipVmmBackend()

    try:
        with patch.object(hip_vmm, "_load_hip_runtime", return_value=runtime):
            backend.ensure_supported(3)
    finally:
        os.close(other_end)

    runtime.set_device.assert_called_once_with(3)
    runtime.allocation_granularity.assert_called_once_with(3)
    runtime.create.assert_called_once_with(65_536, 3)
    runtime.export_fd.assert_called_once_with(0xA11)
    imported_fd = runtime.import_fd.call_args.args[0]
    assert imported_fd != exported_fd
    runtime.reserve.assert_called_once_with(65_536, 65_536)
    runtime.map.assert_called_once_with(0xC000, 65_536, 0xB22)
    runtime.set_access.assert_called_once_with(0xC000, 65_536, 3)
    runtime.unmap.assert_called_once_with(0xC000, 65_536)
    runtime.address_free.assert_called_once_with(0xC000, 65_536)
    assert runtime.release.call_args_list == [call(0xB22), call(0xA11)]
    with pytest.raises(OSError):
        os.fstat(exported_fd)
    with pytest.raises(OSError):
        os.fstat(imported_fd)


def test_hip_capability_rejects_pre_72_runtime_clearly():
    runtime = MagicMock()
    runtime.runtime_version.return_value = 70_152_802
    backend = hip_vmm.HipVmmBackend()

    with (
        patch.object(hip_vmm, "_load_hip_runtime", return_value=runtime),
        pytest.raises(RuntimeError, match=r"ROCm 7\.2 or newer.*7\.1\.52802"),
    ):
        backend.ensure_supported(0)

    runtime.set_device.assert_not_called()
    runtime.create.assert_not_called()


def test_capability_result_preserves_backend_failure_reason():
    backend = MagicMock()
    backend.platform = "rocm"
    backend.ensure_supported.side_effect = RuntimeError(
        "hipMemCreate: HIP error 801 (operation not supported)"
    )

    capability = vmm_utils.query_vmm_capability(
        torch.device("cuda:2"),
        backend=backend,
    )

    assert not capability.supported
    assert capability.platform == "rocm"
    assert capability.device_id == 2
    assert "operation not supported" in capability.reason
    with pytest.raises(RuntimeError, match="SharedEP VMM.*operation not supported"):
        vmm_utils.require_vmm_backend(
            torch.device("cuda:2"),
            backend=backend,
        )


@pytest.mark.parametrize("device_type", [2, 10])
def test_dlpack_wrapper_publishes_platform_device_type(device_type):
    observed = {}
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    sentinel = object()

    def consume(capsule):
        pointer = get_pointer(capsule, b"dltensor")
        managed = ctypes.cast(pointer, ctypes.POINTER(vmm._DLManagedTensor))
        observed["device_type"] = managed.contents.dl_tensor.device.device_type
        observed["device_id"] = managed.contents.dl_tensor.device.device_id
        observed["length"] = managed.contents.dl_tensor.shape[0]
        managed.contents.deleter(managed)
        return sentinel

    refs = []
    with patch.object(vmm.torch, "from_dlpack", side_effect=consume):
        result = vmm._uint8_tensor_from_cuda_ptr(
            0x1000,
            4096,
            5,
            refs,
            dlpack_device_type=device_type,
        )

    assert result is sentinel
    assert observed == {
        "device_type": device_type,
        "device_id": 5,
        "length": 4096,
    }
    assert refs == []


def test_rank_major_allocator_uses_platform_neutral_backend():
    backend = MagicMock()
    backend.platform = "rocm"
    backend.supports_fabric = False
    backend.dlpack_device_type = 10
    backend.get_allocation_granularity.return_value = 64
    backend.create_allocation.return_value = 0x11
    backend.import_posix_fd.return_value = 0x22
    backend.reserve.return_value = 0x1000
    global_storage = torch.empty(128, dtype=torch.uint8)
    local_storage = global_storage[:10]

    with (
        patch.object(vmm.dist, "get_rank", return_value=0),
        patch.object(vmm.dist, "get_world_size", return_value=2),
        patch.object(vmm, "_validate_same_host_group"),
        patch.object(vmm, "require_vmm_backend", return_value=backend),
        patch.object(vmm, "_synchronize_vmm_stage"),
        patch.object(
            vmm,
            "_select_handle_transport",
            return_value=(False, [None, None], [], {(1, 0): 91}),
        ),
        patch.object(
            vmm,
            "_construct_rank_major_views",
            return_value=(global_storage, local_storage),
        ) as construct_views,
    ):
        allocation = vmm.allocate_rank_major_vmm(
            cpu_group="group",
            device=torch.device("cuda:0"),
            logical_rank_bytes=10,
        )

    backend.create_allocation.assert_called_once_with(
        64,
        0,
        allow_fabric=False,
    )
    backend.import_posix_fd.assert_called_once_with(91)
    assert backend.map.call_args_list == [
        call(0x1000, 64, 0x11),
        call(0x1040, 64, 0x22),
    ]
    assert backend.set_access.call_args_list == [
        call(0x1000, 64, 0),
        call(0x1040, 64, 0),
    ]
    assert backend.release.call_args_list == [call(0x11), call(0x22)]
    construct_views.assert_called_once_with(
        cpu_group="group",
        rank=0,
        base_va=0x1000,
        total_bytes=128,
        mapped_rank_bytes=64,
        logical_rank_bytes=10,
        device_id=0,
        refs=[],
        dlpack_device_type=10,
    )
    assert allocation._vmm_backend is backend


def test_allocation_close_unmaps_every_segment_then_frees_address():
    backend = MagicMock()
    storage = torch.empty(0, dtype=torch.uint8)
    allocation = vmm.SharedEpVmmAllocation(
        local_storage=storage,
        global_storage=storage,
        rank=0,
        world_size=2,
        logical_rank_bytes=32,
        mapped_rank_bytes=64,
        granularity=64,
        _base_va=0x1000,
        _total_bytes=128,
        _vmm_backend=backend,
    )

    with patch.object(vmm.torch.cuda, "synchronize"):
        allocation.close()
        allocation.close()

    assert backend.unmap.call_args_list == [
        call(0x1000, 64),
        call(0x1040, 64),
    ]
    backend.address_free.assert_called_once_with(0x1000, 128)
    assert allocation._base_va == 0
    assert allocation.local_storage.numel() == 0
    assert allocation.global_storage.numel() == 0
