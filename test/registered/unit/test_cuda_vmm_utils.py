"""Unit tests for the VMM cross-process handle helpers in ``vmm_utils``.

Round-trips export -> exchange -> import/map across ranks for both transports
(POSIX, FABRIC) and both mapping shapes (single base, multi-chunk span). The
only in-tree consumer, ``register_graph_inputs``, reaches this path only under
``expandable_segments``, so the tests allocate shareable buffers directly. A
POSIX-only allocation forces ``export_shareable_handles`` down its POSIX
fallback (otherwise unreachable on FABRIC hardware); FABRIC cases need an
NVLink fabric (GB200/GB300) and skip elsewhere.
"""

from __future__ import annotations

import atexit
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
from cuda.bindings import driver as drv

from sglang.kernels.jit.utils import cache_once
from sglang.srt.utils import cuda_vmm_utils
from sglang.srt.utils.cuda_vmm_utils import (
    check_drv,
    exchange_posix_fds,
    export_shareable_handles,
    get_allocation_granularity,
    get_device_allocation_handle_type,
    import_and_map_alloc,
    make_device_allocation_prop,
    make_rw_access_desc,
    map_chunk_into_span,
    release_mappings,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=15, stage="base-b", runner_config="2-gpu-large")

_FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
_POSIX_FD = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
_RECOMMENDED = drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
_ALLOC_BYTES = 2 * 1024 * 1024


@cache_once
def _gloo_group() -> dist.ProcessGroup:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo")
    atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


def _make_prop(handle_type, device_id: int):
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = handle_type
    return prop


@cache_once
def _fabric_available() -> bool:
    """True if this device can create + export FABRIC handles (GB200/GB300)."""
    prop = _make_prop(_FABRIC, torch.cuda.current_device())
    err, gran = drv.cuMemGetAllocationGranularity(prop, _RECOMMENDED)
    if err != drv.CUresult.CUDA_SUCCESS:
        return False
    err, handle = drv.cuMemCreate(int(gran), prop, 0)
    if err != drv.CUresult.CUDA_SUCCESS:
        return False
    err, _ = drv.cuMemExportToShareableHandle(handle, _FABRIC, 0)
    drv.cuMemRelease(handle)
    return err == drv.CUresult.CUDA_SUCCESS


def _create_alloc(handle_type, size_hint: int):
    """Create a mapped, RW, shareable VMM allocation. Returns (handle, va, size)."""
    device_id = torch.cuda.current_device()
    prop = _make_prop(handle_type, device_id)
    gran = check_drv(
        drv.cuMemGetAllocationGranularity(prop, _RECOMMENDED),
        "cuMemGetAllocationGranularity",
    )
    size = ((size_hint + gran - 1) // gran) * gran
    handle = check_drv(drv.cuMemCreate(size, prop, 0), "cuMemCreate")
    va = check_drv(drv.cuMemAddressReserve(size, gran, 0, 0), "cuMemAddressReserve")
    check_drv(drv.cuMemMap(int(va), size, 0, handle, 0), "cuMemMap")
    check_drv(
        drv.cuMemSetAccess(int(va), size, [make_rw_access_desc(device_id)], 1),
        "cuMemSetAccess",
    )
    return handle, int(va), size


def _byte(rank: int, chunk: int) -> int:
    """A distinct nonzero fill byte per (rank, chunk)."""
    return (rank * 16 + chunk + 1) & 0xFF


def _assert_region(va: int, expected: int, peer: int, chunk: int) -> None:
    host = np.empty(16, dtype=np.uint8)
    check_drv(drv.cuMemcpyDtoH(host.ctypes.data, va, host.nbytes), "cuMemcpyDtoH")
    assert (host == expected).all(), (
        f"read {host.tolist()} from peer {peer} chunk {chunk}, expected all {expected}"
    )


@pytest.mark.parametrize(
    ("rejected", "expected"),
    [
        ((_FABRIC,), _POSIX_FD),
        ((_FABRIC, _POSIX_FD), 0),
    ],
)
def test_default_handle_type_fallback(monkeypatch, rejected, expected) -> None:
    device_id = torch.cuda.current_device()
    create = drv.cuMemCreate

    def reject_selected(size, prop, flags):
        if prop.requestedHandleTypes in rejected:
            return (drv.CUresult.CUDA_ERROR_NOT_SUPPORTED, None)
        return create(size, prop, flags)

    get_device_allocation_handle_type.cache_clear()
    monkeypatch.setattr(cuda_vmm_utils, "is_gpu_fabric_ready", lambda _device: True)
    monkeypatch.setattr(drv, "cuMemCreate", reject_selected)
    try:
        selected = get_device_allocation_handle_type(device_id)
        prop = make_device_allocation_prop(device_id)
        assert selected == expected
        assert prop.requestedHandleTypes == expected
        assert prop.allocFlags.gpuDirectRDMACapable == 0

        explicit = make_device_allocation_prop(
            device_id,
            handle_types=_FABRIC,
            gpu_direct_rdma=True,
        )
        assert explicit.requestedHandleTypes == _FABRIC
        assert explicit.allocFlags.gpuDirectRDMACapable == 1

        non_exportable = make_device_allocation_prop(device_id, handle_types=None)
        assert non_exportable.requestedHandleTypes == 0

        explicit_none = make_device_allocation_prop(device_id, handle_types=0)
        assert explicit_none.requestedHandleTypes == 0

        with pytest.raises(ValueError, match="handle_types must be"):
            make_device_allocation_prop(device_id, handle_types="fabric")
        with pytest.raises(ValueError, match="invalid CUDA handle-type value"):
            make_device_allocation_prop(device_id, handle_types=42)
    finally:
        get_device_allocation_handle_type.cache_clear()


@pytest.mark.parametrize(
    ("handle_types", "expected"),
    [
        (None, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE),
        (0, drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE),
        (_POSIX_FD, _POSIX_FD),
        (_FABRIC, _FABRIC),
    ],
)
def test_allocation_prop_assigns_handle_type_enum(
    monkeypatch, handle_types, expected
) -> None:
    """CUDA bindings 13.0.x reject integer VMM handle types at assignment."""

    class Fields:
        pass

    class StrictAllocationProp:
        def __init__(self):
            self.location = Fields()
            self.allocFlags = Fields()
            self._requested_handle_types = None

        @property
        def requestedHandleTypes(self):
            return self._requested_handle_types

        @requestedHandleTypes.setter
        def requestedHandleTypes(self, value):
            if not isinstance(value, drv.CUmemAllocationHandleType):
                raise TypeError("requestedHandleTypes requires a CUDA enum")
            self._requested_handle_types = value

    monkeypatch.setattr(drv, "CUmemAllocationProp", StrictAllocationProp)

    prop = make_device_allocation_prop(0, handle_types=handle_types)

    assert prop.requestedHandleTypes is expected


def test_granularity_defaults_to_recommended(monkeypatch) -> None:
    prop = make_device_allocation_prop(0, handle_types=None)
    seen = []

    def granularity(_prop, flag):
        seen.append(flag)
        return (drv.CUresult.CUDA_SUCCESS, _ALLOC_BYTES)

    monkeypatch.setattr(drv, "cuMemGetAllocationGranularity", granularity)
    assert get_allocation_granularity(prop) == _ALLOC_BYTES
    assert seen == [_RECOMMENDED]


@pytest.mark.parametrize("n_chunks", [1, 3])
@pytest.mark.parametrize("transport", ["posix", "fabric"])
def test_handle_roundtrip(transport: str, n_chunks: int) -> None:
    group = _gloo_group()
    if transport == "fabric" and not _fabric_available():
        pytest.skip("FABRIC handles require an NVLink fabric (GB200/GB300)")
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    device_id = torch.cuda.current_device()
    handle_type = _FABRIC if transport == "fabric" else _POSIX_FD

    handles, vas, sizes = [], [], []
    for chunk in range(n_chunks):
        handle, va, size = _create_alloc(handle_type, _ALLOC_BYTES)
        check_drv(drv.cuMemsetD8(va, _byte(rank, chunk), size), "cuMemsetD8")
        handles.append(handle)
        vas.append(va)
        sizes.append(size)
    torch.cuda.synchronize()

    posix_fds, peer_fds, mappings = [], {}, []
    try:
        fabric_handles, posix_fds, use_fabric = export_shareable_handles(
            handles, group, rank
        )
        assert use_fabric == (transport == "fabric")

        # FABRIC handles travel inline; POSIX fds are exchanged out-of-band
        # (process-local).
        local_meta = [
            (sizes[c], fabric_handles[c] if use_fabric else None)
            for c in range(n_chunks)
        ]
        all_meta = [None] * world
        dist.all_gather_object(all_meta, local_meta, group=group)
        if not use_fabric:
            peer_fds = exchange_posix_fds(
                group, rank, world, posix_fds, [n_chunks] * world
            )

        for peer in range(world):
            if peer == rank:
                continue
            peer_meta = all_meta[peer]
            if n_chunks == 1:
                size, fabric_handle = peer_meta[0]
                fd = None if use_fabric else peer_fds[(peer, 0)]
                peer_va = import_and_map_alloc(
                    fabric_handle,
                    fd,
                    size,
                    device_id,
                    use_fabric=use_fabric,
                    peer_rank=peer,
                )
                mappings.append((peer_va, size, [(0, size)]))
                _assert_region(peer_va, _byte(peer, 0), peer, 0)
                continue

            span_size = sum(size for size, _ in peer_meta)
            span_va = int(
                check_drv(
                    drv.cuMemAddressReserve(span_size, 0, 0, 0),
                    "cuMemAddressReserve(span)",
                )
            )
            rel, mapped = 0, []
            for chunk, (size, fabric_handle) in enumerate(peer_meta):
                fd = None if use_fabric else peer_fds[(peer, chunk)]
                map_chunk_into_span(
                    fabric_handle,
                    fd,
                    span_va,
                    rel,
                    size,
                    device_id,
                    use_fabric=use_fabric,
                    peer_rank=peer,
                )
                mapped.append((rel, size))
                rel += size
            mappings.append((span_va, span_size, mapped))
            rel = 0
            for chunk, (size, _) in enumerate(peer_meta):
                _assert_region(span_va + rel, _byte(peer, chunk), peer, chunk)
                rel += size
    finally:
        release_mappings(mappings)
        for fd in peer_fds.values():
            os.close(fd)
        for fd in posix_fds:
            os.close(fd)
        for handle, va, size in zip(handles, vas, sizes):
            check_drv(drv.cuMemUnmap(va, size), "cuMemUnmap")
            check_drv(drv.cuMemAddressFree(va, size), "cuMemAddressFree")
            check_drv(drv.cuMemRelease(handle), "cuMemRelease")


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
