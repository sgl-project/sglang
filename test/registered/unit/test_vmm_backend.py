"""VMM/IPC properties DWDP's composite address space relies on, on either driver.

Each case pins a conclusion that had to be established against a real driver
rather than read off a spec, and that a driver or torch upgrade could silently
take away:

  - the reservations live in the backend's own context, so torch must still see
    them and its allocator must not hand back a reserved range;
  - one VA range can hold several distinct physical objects and a kernel reading
    across a seam must get the same answer as a contiguous copy;
  - a physical object survives an export/import round trip and re-maps at a
    caller-chosen offset, which is what makes peer weight prefetch work;
  - the memory these reservations hold is visible to SGLang's free-memory query,
    which torch's allocator alone cannot report.

CUDA and Intel XPU run the same cases through ``get_vmm_backend``. Where the two
drivers genuinely differ the case asks the backend -- ``owns_exported_fds`` for
who closes an fd, ``export_handles`` for fabric versus POSIX fd -- rather than
branching on the device type.

Run with ``python test_vmm_backend.py`` (relaunches under torchrun for the
cross-process cases).
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

from sglang.srt.utils import get_device, get_device_module, is_cuda, is_xpu
from sglang.srt.utils.common import get_available_gpu_memory
from sglang.srt.utils.vmm_backend import get_vmm_backend
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=40, stage="base-b", runner_config="2-gpu-large")
register_xpu_ci(est_time=30, suite="nightly-xpu-2-gpu", nightly=True)

# the two backends get_vmm_backend() knows
pytestmark = pytest.mark.skipif(
    not (is_cuda() or is_xpu()), reason="requires a CUDA or Intel XPU device"
)

_NUM_OBJECTS = 8


def _local_device_id() -> int:
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    get_device_module().set_device(device_id)
    return device_id


def _gloo_group() -> dist.ProcessGroup:
    if not dist.is_initialized():
        _local_device_id()
        dist.init_process_group(backend="gloo")
        atexit.register(dist.destroy_process_group)
    return dist.group.WORLD


def test_an_odd_multiple_of_the_granularity_can_be_created_and_mapped() -> None:
    """DWDP sizes each physical object from the expert bytes it holds, so the
    object is an arbitrary multiple of the granularity rather than a round one.
    Level Zero rejected such a size outright when the granularity was queried
    below its own step, with UNSUPPORTED_SIZE from zePhysicalMemCreate."""
    device_id = _local_device_id()
    backend = get_vmm_backend(device_id)
    granularity = backend.granularity
    assert granularity & (granularity - 1) == 0, f"{granularity} is not a power of two"

    # an odd multiple of the granularity, the shape the failing shard had
    size = 33 * granularity
    reservation = backend.make_reservation(size, exportable=True, alignment=granularity)
    try:
        reservation.map(0, size, retain_handle=False)
    finally:
        reservation.close()


@pytest.mark.skipif(
    not is_xpu(), reason="only Level Zero reports a page size per allocation size"
)
def test_level_zero_granularity_covers_every_allocation_size() -> None:
    """Level Zero reports a page size per allocation size rather than per device,
    and rejects a physical object whose size is not a multiple of the page it
    reports for that size. So the granularity DWDP aligns everything to has to be
    the coarse page, not whatever the smallest query returns."""
    from sglang.srt.utils.xpu_vmm_utils import get_device_granularity, query_page_size

    device_id = _local_device_id()
    granularity = get_device_granularity(device_id)
    for multiplier in (1, 2, 3, 33, 512):
        size = multiplier * granularity
        assert size % query_page_size(device_id, size) == 0, (
            f"{size} B needs a {query_page_size(device_id, size)} B page but the "
            f"granularity is {granularity} B"
        )


def test_reserved_va_is_invisible_to_the_torch_allocator() -> None:
    """The context behind these reservations is not the one torch's stream uses.
    Torch must still read the mapping, and its allocator must never hand back a
    range we reserved -- otherwise a DWDP weight and a live activation would
    occupy the same pages."""
    device_id = _local_device_id()
    device = get_device(device_id)
    backend = get_vmm_backend(device_id)
    before = [torch.randn(1 << 20, device=device) for _ in range(4)]

    size = 16 * backend.granularity
    reservation = backend.make_reservation(size, exportable=False)
    try:
        reservation.map(0, size, retain_handle=False)
        span = (reservation.base, reservation.base + size)
        for tensor in before:
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            assert end <= span[0] or span[1] <= start, (
                "reservation overlaps live tensor"
            )

        view = backend.tensor_from_pointer(
            reservation.base, size, shape=(size // 2,), dtype=torch.bfloat16
        )
        expected = torch.randn(size // 2, dtype=torch.bfloat16, device=device)
        view.copy_(expected)
        backend.synchronize()
        assert torch.equal(view, expected), "torch cannot read the foreign-context map"

        after = [torch.randn(1 << 20, device=device) for _ in range(4)]
        for tensor in after:
            assert not span[0] <= tensor.data_ptr() < span[1], "allocator reused our VA"
        assert torch.equal(view, expected), "later allocations clobbered the mapping"
    finally:
        reservation.close()


def test_gemm_across_composite_va_seams_matches_contiguous() -> None:
    """A composite VA is only useful if a kernel cannot tell it from one
    allocation. Reading B across 7 physical-object boundaries must be bit-exact
    against the same weights in a normal tensor."""
    device_id = _local_device_id()
    device = get_device(device_id)
    backend = get_vmm_backend(device_id)
    page = backend.granularity
    span = _NUM_OBJECTS * page

    reservation = backend.make_reservation(span, exportable=False)
    try:
        for i in range(_NUM_OBJECTS):
            reservation.map(i * page, page, retain_handle=False)

        k_dim = 1024
        rows = span // (k_dim * 2)
        weight = backend.tensor_from_pointer(
            reservation.base, span, shape=(rows, k_dim), dtype=torch.bfloat16
        )
        reference = torch.randn(rows, k_dim, dtype=torch.bfloat16, device=device)
        weight.copy_(reference)
        x = torch.randn(64, k_dim, dtype=torch.bfloat16, device=device)

        composite = x @ weight.t()
        expected = x @ reference.t()
        backend.synchronize()
        assert torch.equal(weight, reference), "readback across seams differs"
        assert torch.equal(composite, expected), "GEMM over the composite VA differs"
    finally:
        reservation.close()


def test_prefetch_event_protocol_orders_copy_before_compute() -> None:
    """DWDP overlaps the peer-weight copy with compute on a second stream and
    orders them with events only. A stream/event regression would surface as
    torn weights rather than a failure, so pin the ordering."""
    device_id = _local_device_id()
    device = get_device(device_id)
    device_module = get_device_module()
    backend = get_vmm_backend(device_id)
    page = backend.granularity
    span = 4 * page

    reservation = backend.make_reservation(span, exportable=False)
    try:
        reservation.map(0, span, retain_handle=False)
        k_dim = 512
        rows = span // (k_dim * 2)
        weight = backend.tensor_from_pointer(
            reservation.base, span, shape=(rows, k_dim), dtype=torch.bfloat16
        )
        source = torch.randn(rows, k_dim, dtype=torch.bfloat16, device=device)
        x = torch.randn(32, k_dim, dtype=torch.bfloat16, device=device)

        copy_stream = device_module.Stream(device=torch.device(device))
        done = device_module.Event()
        with device_module.stream(copy_stream):
            weight.copy_(source)
            done.record(copy_stream)
        device_module.current_stream().wait_event(done)
        out = x @ weight.t()
        backend.synchronize()
        assert torch.equal(out, x @ source.t()), "compute raced the prefetch copy"
    finally:
        reservation.close()


def test_export_import_round_trip_on_one_device() -> None:
    """Reserve-and-alias only works if an exported physical object can be mapped
    at a caller-chosen VA offset. Level Zero's zeMemGetIpcHandle cannot do this
    (it rejects physical-memory handles), so the export path is load-bearing on
    both drivers even within one process."""
    device_id = _local_device_id()
    device = get_device(device_id)
    backend = get_vmm_backend(device_id)
    group = _gloo_group()
    rank = dist.get_rank()
    size = 4 * backend.granularity

    source = backend.make_reservation(size, exportable=True)
    handle = int(source.map(0, size, retain_handle=True))
    fill = torch.full((size // 2,), 0.5, dtype=torch.bfloat16, device=device)
    backend.copy_tensor_to_pointer(source.base, fill)
    backend.synchronize()
    source.close(release_handles=False)

    fabric_handles, fds, use_fabric = backend.export_handles([handle], group, rank)
    if use_fabric:
        assert len(fabric_handles) == 1
    else:
        assert len(fds) == 1 and fds[0] >= 0

    imported = backend.import_handle(
        fabric_handles[0] if use_fabric else None,
        None if use_fabric else fds[0],
        use_fabric=use_fabric,
        peer_rank=rank,
        size=size,
    )
    # Map the imported object at a nonzero offset in a larger reservation --
    # exactly the [pre pages | peer handle | post pages] shape DWDP builds.
    composite = backend.make_reservation(3 * size, exportable=False)
    try:
        composite.map(0, size, retain_handle=False)
        composite.map_existing(size, size, imported)
        composite.map(2 * size, size, retain_handle=False)
        view = backend.tensor_from_pointer(
            composite.base + size, size, shape=(size // 2,), dtype=torch.bfloat16
        )
        backend.synchronize()
        assert torch.equal(view, fill), "the imported object read back wrong"
    finally:
        composite.close()
        backend.release_handle(imported)
        if not use_fabric and backend.owns_exported_fds():
            os.close(fds[0])
        backend.release_handle(handle)


def test_memory_accounting_sees_vmm_allocations() -> None:
    """A DWDP rank copies each expert shard into driver memory and then frees
    torch's copy, so torch's memory_allocated *drops* while ~16 GiB stays
    resident. Sizing the KV pool against that figure made all four ranks die with
    OutOfMemoryError while reporting 23.91 GiB of 23.91 GiB free -- on XPU the
    driver query that looks like cuMemGetInfo is a stub, and torch forwards it."""
    device_id = _local_device_id()
    device_module = get_device_module()
    backend = get_vmm_backend(device_id)
    size = 512 * backend.granularity

    before = get_available_gpu_memory(get_device(), device_id)
    torch_before = device_module.memory_allocated(device_id)
    reservation = backend.make_reservation(size, exportable=True)
    try:
        reservation.map(0, size, retain_handle=False)
        after = get_available_gpu_memory(get_device(), device_id)
        assert device_module.memory_allocated(device_id) == torch_before, (
            "torch's allocator saw the driver mapping, so this no longer "
            "exercises the invisible-footprint path"
        )
        # get_available_gpu_memory reports GiB; the driver rounds to whole pages
        # and other ranks may allocate concurrently, so bound rather than equate.
        freed = (before - after) * (1 << 30)
        assert freed >= size, (
            f"{size} B of driver memory only moved the free figure by {freed} B"
        )
    finally:
        reservation.close()


def test_cross_rank_peer_weight_alias() -> None:
    """The DWDP transport itself: every rank exports its shard, every peer
    imports it, maps it into its own address space, and reads the peer's data off
    a different device."""
    group = _gloo_group()
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("needs at least 2 ranks")
    rank = dist.get_rank()
    device_id = _local_device_id()
    device = get_device(device_id)
    backend = get_vmm_backend(device_id)
    size = 2 * backend.granularity

    reservation = backend.make_reservation(size, exportable=True)
    handle = int(reservation.map(0, size, retain_handle=True))
    mine = torch.full(
        (size // 2,), float(rank + 1), dtype=torch.bfloat16, device=device
    )
    backend.copy_tensor_to_pointer(reservation.base, mine)
    backend.synchronize()
    reservation.close(release_handles=False)

    fabric_handles, fds, use_fabric = backend.export_handles([handle], group, rank)
    peer_fds = {}
    all_fabric = None
    if use_fabric:
        all_fabric = [None] * world_size
        dist.all_gather_object(all_fabric, fabric_handles, group=group)
    else:
        peer_fds = backend.exchange_fds(group, rank, world_size, fds, [1] * world_size)

    imported = []
    peer_reservations = []
    try:
        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue
            _assert_peer_is_reachable(device_id, peer_rank)
            peer_handle = backend.import_handle(
                all_fabric[peer_rank][0] if use_fabric else None,
                None if use_fabric else peer_fds[(peer_rank, 0)],
                use_fabric=use_fabric,
                peer_rank=peer_rank,
                size=size,
            )
            imported.append(peer_handle)
            peer_reservation = backend.make_reservation(size, exportable=True)
            peer_reservation.map_existing(0, size, peer_handle)
            peer_reservations.append(peer_reservation)

            view = backend.tensor_from_pointer(
                peer_reservation.base, size, shape=(size // 2,), dtype=torch.bfloat16
            )
            # Copy out on the local device, the way prefetch_layer does.
            staged = torch.empty_like(view)
            staged.copy_(view)
            backend.synchronize()
            assert staged.min().item() == staged.max().item() == float(peer_rank + 1), (
                f"rank {rank} read {staged[0].item()} from peer {peer_rank}"
            )
        dist.barrier(group=group)
    finally:
        for peer_reservation in peer_reservations:
            peer_reservation.close()
        for peer_handle in imported:
            backend.release_handle(peer_handle)
        for fd in peer_fds.values():
            os.close(fd)
        if not use_fabric and backend.owns_exported_fds():
            for fd in fds:
                os.close(fd)
        backend.release_handle(handle)


def _assert_peer_is_reachable(device_id: int, peer_rank: int) -> None:
    # Only the Level Zero backend exposes a peer-access query; on CUDA an
    # unreachable peer surfaces as a failure from the import itself.
    if not is_xpu():
        return

    from sglang.srt.utils.xpu_vmm_utils import can_access_peer

    assert can_access_peer(device_id, peer_rank), (
        f"device {device_id} cannot reach peer device {peer_rank}"
    )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2,))
