"""ROCm HIP VMM primitive and lifecycle tests."""

from __future__ import annotations

import array
import gc
import importlib.util
import multiprocessing
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

_PROBE_BYTES = 256
_HIP_VMM_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/moe/dwdp/hip_vmm.py"
)
_HIP_VMM_MODULE_NAME = "_sglang_dwdp_hip_vmm_test_target"
_HIP_VMM_SPEC = importlib.util.spec_from_file_location(
    _HIP_VMM_MODULE_NAME, _HIP_VMM_PATH
)
if _HIP_VMM_SPEC is None or _HIP_VMM_SPEC.loader is None:
    raise ImportError(f"could not load HIP VMM wrapper from {_HIP_VMM_PATH}")
hip_vmm = importlib.util.module_from_spec(_HIP_VMM_SPEC)
sys.modules[_HIP_VMM_MODULE_NAME] = hip_vmm
_HIP_VMM_SPEC.loader.exec_module(hip_vmm)


@pytest.fixture(scope="module")
def hip_vmm_device() -> int:
    if torch.version.hip is None:
        pytest.skip("HIP VMM tests require a ROCm build of PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("HIP VMM tests require a visible ROCm GPU")

    available, reason = hip_vmm.extension_availability()
    if not available:
        pytest.skip(f"sgl_kernel ROCm HIP VMM extension unavailable: {reason}")

    device_id = torch.cuda.current_device()
    if not hip_vmm.is_supported(device_id):
        pytest.skip(f"HIP VMM is not supported by device {device_id}")
    return device_id


def _close_handles(handles: list[int]) -> None:
    errors = []
    for handle in reversed(handles):
        try:
            hip_vmm.release_handle(handle)
        except BaseException as error:
            errors.append(error)
    if errors:
        raise hip_vmm.HipVmmCleanupError("test handle cleanup failed", errors)


def _send_fd(sock: socket.socket, fd: int) -> None:
    descriptors = array.array("i", [fd])
    sent = sock.sendmsg(
        [b"F"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())],
    )
    if sent != 1:
        raise RuntimeError(f"short sendmsg while transferring VMM fd: {sent}")


def _recv_fd(sock: socket.socket) -> int:
    item_size = array.array("i").itemsize
    payload, ancillary, flags, _ = sock.recvmsg(1, socket.CMSG_SPACE(item_size))
    if payload != b"F":
        raise RuntimeError(f"invalid VMM fd payload: {payload!r}")
    if flags & socket.MSG_CTRUNC:
        raise RuntimeError("VMM fd ancillary data was truncated")

    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors = array.array("i")
            descriptors.frombytes(data[:item_size])
            if descriptors:
                return int(descriptors[0])
    raise RuntimeError("SCM_RIGHTS message contained no VMM fd")


def _fd_import_worker(
    fd_socket: socket.socket,
    result_pipe: Any,
    device_id: int,
    allocation_size: int,
    expected: int,
    replacement: int,
) -> None:
    received_fd = -1
    imported_handle = 0
    mapping = None
    tensor = None
    failure = None
    cleanup_errors: list[BaseException] = []
    try:
        torch.cuda.set_device(device_id)
        received_fd = _recv_fd(fd_socket)
        imported_handle = hip_vmm.import_fd(received_fd)

        # import_fd preserves its input. Close the SCM_RIGHTS descriptor now;
        # HIP owns an independent dma-buf reference associated with the handle.
        os.close(received_fd)
        received_fd = -1

        mapping = hip_vmm.map_handles(
            [imported_handle],
            [allocation_size],
            device_id,
            alignment=allocation_size,
        )
        tensor = hip_vmm.tensor_from_ptr(
            mapping.address, (_PROBE_BYTES,), torch.uint8, device_id
        )
        observed = tensor.cpu()
        if not torch.equal(
            observed, torch.full((_PROBE_BYTES,), expected, dtype=torch.uint8)
        ):
            raise AssertionError(
                f"imported rank read unexpected bytes: {observed.tolist()}"
            )
        tensor.fill_(replacement)
        torch.cuda.synchronize(device_id)
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if tensor is not None:
            del tensor
            try:
                torch.cuda.synchronize(device_id)
            except BaseException as error:
                cleanup_errors.append(error)
        if mapping is not None:
            try:
                mapping.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if imported_handle:
            try:
                hip_vmm.release_handle(imported_handle)
            except BaseException as error:
                cleanup_errors.append(error)
        if received_fd >= 0:
            try:
                os.close(received_fd)
            except BaseException as error:
                cleanup_errors.append(error)
        fd_socket.close()

    try:
        if failure is not None or cleanup_errors:
            result_pipe.send(
                (
                    "error",
                    failure,
                    [f"{type(error).__name__}: {error}" for error in cleanup_errors],
                )
            )
        else:
            result_pipe.send(("ok",))
    finally:
        result_pipe.close()


def test_capability_and_granularity(hip_vmm_device: int) -> None:
    device_id = hip_vmm_device
    assert hip_vmm.is_supported(device_id)

    minimum = hip_vmm.get_allocation_granularity(
        device_id, shareable=True, recommended=False
    )
    recommended = hip_vmm.get_allocation_granularity(device_id)
    local = hip_vmm.get_allocation_granularity(device_id, shareable=False)
    assert minimum > 0
    assert recommended >= minimum
    assert local > 0
    assert minimum & (minimum - 1) == 0
    assert recommended & (recommended - 1) == 0
    assert local & (local - 1) == 0


def test_single_process_tensor_read_write(hip_vmm_device: int) -> None:
    device_id = hip_vmm_device
    granularity = hip_vmm.get_allocation_granularity(device_id)
    handle = hip_vmm.create_shareable_handle(granularity, device_id)
    try:
        with hip_vmm.map_handles(
            [handle], [granularity], device_id, alignment=granularity
        ) as mapping:
            tensor = None
            replacement_view = None
            fp8_view = None
            try:
                expected = torch.arange(_PROBE_BYTES, dtype=torch.uint8)
                tensor = hip_vmm.tensor_from_ptr(
                    mapping.address, (_PROBE_BYTES,), torch.uint8, device_id
                )
                tensor.copy_(expected.to(device=f"cuda:{device_id}"))
                torch.cuda.synchronize(device_id)
                assert torch.equal(tensor.cpu(), expected)

                # at::from_blob uses a no-op deleter: dropping one view leaves
                # the VMM allocation intact and a new view sees the same bytes.
                pointer = tensor.data_ptr()
                tensor = None
                gc.collect()
                replacement_view = hip_vmm.tensor_from_ptr(
                    mapping.address, (_PROBE_BYTES,), torch.uint8, device_id
                )
                assert replacement_view.data_ptr() == pointer
                assert torch.equal(replacement_view.cpu(), expected)

                fp8_dtype = getattr(torch, "float8_e4m3fnuz", None)
                if fp8_dtype is not None:
                    fp8_view = hip_vmm.tensor_from_ptr(
                        mapping.address, (_PROBE_BYTES,), fp8_dtype, device_id
                    )
                    assert fp8_view.dtype == fp8_dtype
                    assert fp8_view.data_ptr() == pointer
                    assert torch.equal(fp8_view.view(torch.uint8).cpu(), expected)
            finally:
                fp8_view = None
                replacement_view = None
                tensor = None
                torch.cuda.synchronize(device_id)
    finally:
        hip_vmm.release_handle(handle)


def test_composite_va_from_two_handles(hip_vmm_device: int) -> None:
    device_id = hip_vmm_device
    granularity = hip_vmm.get_allocation_granularity(device_id, shareable=False)
    handles = []
    try:
        for _ in range(2):
            handles.append(hip_vmm.create_local_handle(granularity, device_id))
        with hip_vmm.map_handles(
            handles,
            [granularity, granularity],
            device_id,
            alignment=granularity,
        ) as mapping:
            tensor = None
            try:
                tensor = hip_vmm.tensor_from_ptr(
                    mapping.address,
                    (2 * granularity,),
                    torch.uint8,
                    device_id,
                )
                tensor[:granularity].fill_(0x31)
                tensor[granularity:].fill_(0xA7)
                torch.cuda.synchronize(device_id)
                boundary = tensor[granularity - 16 : granularity + 16].cpu()
                assert torch.equal(
                    boundary[:16], torch.full((16,), 0x31, dtype=torch.uint8)
                )
                assert torch.equal(
                    boundary[16:], torch.full((16,), 0xA7, dtype=torch.uint8)
                )
            finally:
                tensor = None
                torch.cuda.synchronize(device_id)
    finally:
        _close_handles(handles)


def test_repeated_setup_teardown(hip_vmm_device: int) -> None:
    device_id = hip_vmm_device
    granularity = hip_vmm.get_allocation_granularity(device_id, shareable=False)
    for iteration in range(8):
        handle = hip_vmm.create_local_handle(granularity, device_id)
        try:
            with hip_vmm.map_handles(
                [handle], [granularity], device_id, alignment=granularity
            ) as mapping:
                tensor = None
                try:
                    tensor = hip_vmm.tensor_from_ptr(
                        mapping.address, (64,), torch.uint8, device_id
                    )
                    tensor.fill_(iteration + 1)
                    assert torch.equal(
                        tensor.cpu(),
                        torch.full((64,), iteration + 1, dtype=torch.uint8),
                    )
                finally:
                    tensor = None
                    torch.cuda.synchronize(device_id)
        finally:
            hip_vmm.release_handle(handle)


def test_posix_fd_export_import_across_process(hip_vmm_device: int) -> None:
    device_id = hip_vmm_device
    granularity = hip_vmm.get_allocation_granularity(device_id)
    handle = hip_vmm.create_shareable_handle(granularity, device_id)
    exported_fd = -1
    process = None
    parent_socket = None
    child_socket = None
    parent_pipe = None
    child_pipe = None
    try:
        with hip_vmm.map_handles(
            [handle], [granularity], device_id, alignment=granularity
        ) as mapping:
            tensor = None
            try:
                tensor = hip_vmm.tensor_from_ptr(
                    mapping.address, (_PROBE_BYTES,), torch.uint8, device_id
                )
                tensor.fill_(0x2D)
                torch.cuda.synchronize(device_id)

                exported_fd = hip_vmm.export_fd(handle)
                context = multiprocessing.get_context("spawn")
                parent_socket, child_socket = socket.socketpair()
                parent_pipe, child_pipe = context.Pipe(duplex=False)
                process = context.Process(
                    target=_fd_import_worker,
                    args=(
                        child_socket,
                        child_pipe,
                        device_id,
                        granularity,
                        0x2D,
                        0x6B,
                    ),
                )
                process.start()
                child_socket.close()
                child_socket = None
                child_pipe.close()
                child_pipe = None

                _send_fd(parent_socket, exported_fd)
                os.close(exported_fd)
                exported_fd = -1
                parent_socket.close()
                parent_socket = None

                if not parent_pipe.poll(60):
                    raise TimeoutError("timed out waiting for HIP VMM import rank")
                result = parent_pipe.recv()
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    raise TimeoutError("HIP VMM import rank did not exit")
                assert process.exitcode == 0
                assert result[0] == "ok", result

                torch.cuda.synchronize(device_id)
                assert torch.equal(
                    tensor.cpu(),
                    torch.full((_PROBE_BYTES,), 0x6B, dtype=torch.uint8),
                )
            finally:
                if process is not None and process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                tensor = None
                torch.cuda.synchronize(device_id)
    finally:
        if exported_fd >= 0:
            os.close(exported_fd)
        if parent_socket is not None:
            parent_socket.close()
        if parent_pipe is not None:
            parent_pipe.close()
        if child_socket is not None:
            child_socket.close()
        if child_pipe is not None:
            child_pipe.close()
        hip_vmm.release_handle(handle)


def test_composite_mapping_rolls_back_partial_failure(monkeypatch) -> None:
    events = []

    monkeypatch.setattr(
        hip_vmm,
        "reserve_va",
        lambda size, alignment=0, requested_address=0: (
            events.append(("reserve", size, alignment, requested_address)) or 0x10000
        ),
    )

    def fake_map(address, size, handle, offset=0):
        events.append(("map", address, size, handle, offset))
        if handle == 22:
            raise RuntimeError("injected second-map failure")

    monkeypatch.setattr(hip_vmm, "map_handle", fake_map)
    monkeypatch.setattr(
        hip_vmm,
        "set_access",
        lambda address, size, device_id: events.append(
            ("access", address, size, device_id)
        ),
    )
    monkeypatch.setattr(
        hip_vmm,
        "unmap_va",
        lambda address, size: events.append(("unmap", address, size)),
    )
    monkeypatch.setattr(
        hip_vmm,
        "free_va",
        lambda address, size: events.append(("free", address, size)),
    )

    with pytest.raises(RuntimeError, match="injected second-map failure"):
        hip_vmm.map_handles([11, 22], [4096, 4096], 0, alignment=4096)

    assert events == [
        ("reserve", 8192, 4096, 0),
        ("map", 0x10000, 4096, 11, 0),
        ("map", 0x11000, 4096, 22, 0),
        ("unmap", 0x10000, 4096),
        ("free", 0x10000, 8192),
    ]


def test_composite_mapping_rolls_back_access_failure(monkeypatch) -> None:
    events = []

    monkeypatch.setattr(
        hip_vmm,
        "reserve_va",
        lambda size, alignment=0, requested_address=0: 0x20000,
    )
    monkeypatch.setattr(
        hip_vmm,
        "map_handle",
        lambda address, size, handle, offset=0: events.append(
            ("map", address, size, handle)
        ),
    )

    def fail_access(address, size, device_id):
        events.append(("access", address, size, device_id))
        raise RuntimeError("injected access failure")

    monkeypatch.setattr(hip_vmm, "set_access", fail_access)
    monkeypatch.setattr(
        hip_vmm,
        "unmap_va",
        lambda address, size: events.append(("unmap", address, size)),
    )
    monkeypatch.setattr(
        hip_vmm,
        "free_va",
        lambda address, size: events.append(("free", address, size)),
    )

    with pytest.raises(RuntimeError, match="injected access failure"):
        hip_vmm.map_handles([11, 22], [4096, 4096], 3, alignment=4096)

    assert events == [
        ("map", 0x20000, 4096, 11),
        ("map", 0x21000, 4096, 22),
        ("access", 0x20000, 8192, 3),
        ("unmap", 0x21000, 4096),
        ("unmap", 0x20000, 4096),
        ("free", 0x20000, 8192),
    ]
