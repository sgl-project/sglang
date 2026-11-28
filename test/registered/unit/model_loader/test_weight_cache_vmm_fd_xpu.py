"""CPU-only unit tests for the XPU half of the weight cache's vmm_fd transport.

The Level Zero calls need an Intel GPU and are left to
test/registered/xpu/test_weight_cache_transport_xpu.py. What runs here is the
policy around them, with host memory standing in for the allocations:

  - the one alignment probed for the reservation and every allocation in it,
    and the refusals when a driver's page size does not fit it
  - the daemon's export: a create whose map fails leaves no allocation behind,
    and an out-of-memory map says that the create was not a capacity check
  - the engine's import: misaligned allocation sizes are refused before any
    handle is imported, a failed map leaves neither handle nor reservation, and
    a successful one retains the handles that keep the dma-buf alive
  - teardown order (reservation, then handles), idempotency, use after close
  - the whole daemon-to-engine round trip through these hooks, over a
    socketpair, on host memory

The device-neutral half of the transport is covered in
test_weight_cache_vmm_fd.py, through the CUDA backend.
"""

import contextlib
import os
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils.xpu_vmm_utils import ZeVmmError
from sglang.srt.weight_cache import vmm_fd_xpu
from sglang.srt.weight_cache.protocol import recv_msg
from sglang.srt.weight_cache.transport import VMM_FD_BACKEND
from sglang.srt.weight_cache.vmm_fd_xpu import (
    XpuVmmFdTransportBackend,
    allocation_granularity,
    check_device_indices,
)
from sglang.srt.weight_cache.vmm_layout import SLOT_ALIGNMENT
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CPU = torch.device("cpu")
# The page size Arc Pro B60 reports for a large request, and the alignment the
# backend uses for every allocation.
PAGE_SIZE = 2 << 20


class _FakePhysical:
    """Host-memory stand-in for xpu_vmm_utils.PhysicalMemory.

    ``view`` is the allocation's bytes, carved out of the fake's one device
    buffer, so an importer that looks the allocation up by its dma-buf identity
    reads exactly what the exporter wrote.
    """

    def __init__(self, view, *, device_index, exportable, dma_buf_fd=None):
        self.view = view
        self.size = view.numel()
        self.page_size = PAGE_SIZE
        self.device_index = device_index
        self.exportable = exportable
        self.closed = False
        self.mapped_refs = 0
        # The driver hands out the same fd every time and dups it per caller.
        self._dma_buf_fd = dma_buf_fd
        self.exported_fds = 0

    def export_dma_buf(self):
        if not self.exportable:
            raise ZeVmmError("allocation was created with exportable=False")
        self.exported_fds += 1
        return os.dup(self._dma_buf_fd)

    def close(self):
        if self.mapped_refs:
            raise ZeVmmError("refusing to destroy a mapped physical allocation")
        self.closed = True


class _FakeReservation:
    """Host-memory stand-in for xpu_vmm_utils.VirtualReservation.

    Addresses come from the allocations themselves rather than from a range of
    this object's own, which is the one thing host memory cannot reproduce: a
    real importer picks its own addresses. The hardware test covers that; what
    this preserves is that offsets, sizes and bytes line up.
    """

    def __init__(self, size, *, device_index, page_size=PAGE_SIZE):
        self.size = size
        self.page_size = page_size
        self.device_index = device_index
        self.closed = False
        self.mapped = []

    @property
    def ptr(self) -> int:
        if self.closed:
            raise ZeVmmError("virtual reservation is closed")
        offset, _, physical = self.mapped[0]
        return physical.view.data_ptr() - offset

    def map(self, physical, *, offset=0, size=None, access=1):
        span = physical.size if size is None else size
        if self.closed:
            raise ZeVmmError("virtual reservation is closed")
        self.mapped.append((offset, span, physical))
        physical.mapped_refs += 1
        return physical.view.data_ptr()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for _, _, physical in self.mapped:
            physical.mapped_refs -= 1
        self.mapped = []


class _FakeLevelZero:
    """The subset of xpu_vmm_utils the backend calls, on host memory.

    One buffer stands in for device memory; allocations carve spans out of it
    back to back. Sharing goes through the dma-buf's ``(st_dev, st_ino)``, which
    is how ``xpu_vmm_utils.dma_buf_identity`` identifies a buffer too, so an
    allocation received over SCM_RIGHTS is recognised as the one exported.
    """

    ZeVmmError = ZeVmmError
    CAPACITY = 64 << 20

    def __init__(self, *, page_size=PAGE_SIZE, device_count=1, create_fails_after=None):
        self.page_size = page_size
        self._device_count = device_count
        # map() raises out-of-device-memory from this mapping on, as Level Zero
        # does when the card is full: the create before it still succeeds.
        self.create_fails_after = create_fails_after
        self.pad_creates_to = None
        self.unsupported_reason = None
        self.physicals = []
        self.reservations = []
        self.imported = []
        self._memory = torch.zeros(self.CAPACITY, dtype=torch.uint8)
        self._used = 0
        self._by_dma_buf = {}
        self._owned_fds = []

    def close(self):
        for fd in self._owned_fds:
            os.close(fd)
        self._owned_fds = []

    def device_count(self):
        return self._device_count

    def query_page_size(self, size, *, device_index=0):
        return self.page_size

    def vmm_unsupported_reason(self, *, device_index=0):
        return self.unsupported_reason

    def _carve(self, size):
        if self._used + size > self.CAPACITY:
            raise ZeVmmError(
                "zePhysicalMemCreate failed with ze_result_t 0x70000003 "
                "(ERROR_OUT_OF_DEVICE_MEMORY)"
            )
        view = self._memory[self._used : self._used + size]
        self._used += size
        return view

    def _create(self, size, *, device_index, exportable):
        if self.pad_creates_to is not None:
            size = self.pad_creates_to
        fd = os.memfd_create(f"fake-dma-buf-{len(self.physicals)}")
        self._owned_fds.append(fd)
        physical = _FakePhysical(
            self._carve(size),
            device_index=device_index,
            exportable=exportable,
            dma_buf_fd=fd,
        )
        self._by_dma_buf[os.fstat(fd).st_ino] = physical
        self.physicals.append(physical)
        return physical

    def _from_dma_buf(self, fd, size, *, device_index):
        exported = self._by_dma_buf.get(os.fstat(fd).st_ino)
        if exported is None:
            raise ZeVmmError("zePhysicalMemCreate(import_fd): unknown dma-buf")
        physical = _FakePhysical(
            exported.view[:size],
            device_index=device_index,
            exportable=False,
            dma_buf_fd=fd,
        )
        self.imported.append(physical)
        self.physicals.append(physical)
        return physical

    @property
    def PhysicalMemory(self):
        fake = self
        return SimpleNamespace(
            create=lambda size, *, device_index=0, exportable=True: fake._create(
                size, device_index=device_index, exportable=exportable
            ),
            from_dma_buf=lambda fd, size, *, device_index=0: fake._from_dma_buf(
                fd, size, device_index=device_index
            ),
        )

    def VirtualReservation(self, size, *, device_index=0):
        reservation = _FakeReservation(
            size, device_index=device_index, page_size=self.page_size
        )
        if self.create_fails_after is not None:
            limit = self.create_fails_after
            inner = reservation.map

            def failing_map(physical, *, offset=0, size=None, access=1):
                if len(reservation.mapped) >= limit:
                    raise ZeVmmError(
                        "zeVirtualMemMap failed with ze_result_t 0x70000003 "
                        "(ERROR_OUT_OF_DEVICE_MEMORY)"
                    )
                return inner(physical, offset=offset, size=size, access=access)

            reservation.map = failing_map
        self.reservations.append(reservation)
        return reservation


class _HostXpuBackend(XpuVmmFdTransportBackend):
    """The XPU backend with its torch.xpu calls answered on the host.

    Only the device calls and the storage construction are replaced, so that
    this runs where no XPU exists; every Level Zero decision stays real code.
    """

    free_bytes = 1 << 40

    def _empty_cache(self) -> None:
        pass

    def _synchronize(self) -> None:
        pass

    def _free_device_bytes(self) -> int:
        return self.free_bytes

    @property
    def _device_index(self) -> int:
        return 0

    def _current_device(self) -> torch.device:
        return CPU

    def _tensor_from_pointer(self, pointer: int, nbytes: int) -> torch.Tensor:
        storage = torch._C._construct_storage_from_data_pointer(pointer, CPU, nbytes)
        return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (nbytes,))


@contextlib.contextmanager
def _level_zero(**kwargs):
    fake = _FakeLevelZero(**kwargs)
    try:
        with (
            mock.patch.object(vmm_fd_xpu, "xpu_vmm", fake),
            mock.patch.object(
                torch.xpu, "device_count", return_value=fake.device_count()
            ),
        ):
            yield fake
    finally:
        fake.close()


def _backend(fake, *, max_alloc_bytes=1 << 30) -> _HostXpuBackend:
    return _HostXpuBackend(
        device=CPU, granularity=fake.page_size, max_alloc_bytes=max_alloc_bytes
    )


def _state(**tensors):
    return {name: (tensor, True) for name, tensor in tensors.items()}


class TestGranularity(CustomTestCase):
    """One alignment covers the reservation and every allocation in it, because
    the driver's page size grows with the requested size and the reservation's
    bounds every map offset inside it."""

    def test_the_probe_asks_for_a_size_past_the_page_size_growth(self):
        with _level_zero() as fake:
            sizes = []
            fake.query_page_size = lambda size, *, device_index=0: (
                sizes.append(size) or PAGE_SIZE
            )
            self.assertEqual(allocation_granularity(0), PAGE_SIZE)
        self.assertEqual(sizes, [vmm_fd_xpu._GRANULARITY_PROBE_BYTES])

    def test_a_page_size_that_breaks_slot_alignment_is_refused(self):
        with _level_zero(page_size=SLOT_ALIGNMENT + 1):
            with self.assertRaisesRegex(RuntimeError, "slot alignment"):
                allocation_granularity(0)

    def test_a_reservation_whose_pages_do_not_divide_the_alignment_is_refused(self):
        with _level_zero() as fake:
            backend = _backend(fake)
            fake.page_size = 3 << 20  # does not divide the 2 MiB alignment
            with self.assertRaisesRegex(RuntimeError, "does not divide"):
                backend._reserve(6 << 20)
        self.assertTrue(fake.reservations[-1].closed)


class TestDeviceIndices(CustomTestCase):
    def test_a_torch_and_level_zero_device_count_mismatch_is_refused(self):
        """Allocations are addressed by the torch device index while the driver
        layer indexes its own list; ZE_AFFINITY_MASK can shorten one of them
        and silently put the weights on another card."""
        with _level_zero(device_count=8):
            with mock.patch.object(torch.xpu, "device_count", return_value=4):
                with self.assertRaisesRegex(RuntimeError, "ZE_AFFINITY_MASK"):
                    check_device_indices()

    def test_matching_counts_pass(self):
        with _level_zero(device_count=8):
            with mock.patch.object(torch.xpu, "device_count", return_value=8):
                check_device_indices()


class TestForExport(CustomTestCase):
    def test_an_unsupported_driver_is_refused_before_the_model_loads(self):
        with _level_zero() as fake:
            fake.unsupported_reason = "no dma-buf fd"
            with self.assertRaisesRegex(RuntimeError, "no dma-buf fd"):
                XpuVmmFdTransportBackend.for_export(0)

    def test_a_supported_driver_yields_a_backend_on_that_device(self):
        with _level_zero() as fake:
            backend = XpuVmmFdTransportBackend.for_export(3)
        self.assertEqual(backend.device_type, "xpu")
        self.assertEqual(backend.name, VMM_FD_BACKEND)
        self.assertEqual(backend._device, torch.device("xpu", 3))
        self.assertEqual(backend._granularity, fake.page_size)


class TestDaemonExport(CustomTestCase):
    def test_a_create_whose_map_fails_leaves_no_allocation_behind(self):
        """Level Zero commits memory at the map, so that is where a full card
        reports itself; the allocation created just before must not be left
        holding a handle."""
        with _level_zero(create_fails_after=0) as fake:
            backend = _backend(fake)
            backend.free_bytes = 1 << 20
            with self.assertRaisesRegex(RuntimeError, "ERROR_OUT_OF_DEVICE_MEMORY"):
                backend.prepare_export(_state(w=torch.ones(1024)))
        self.assertEqual(len(fake.physicals), 1)
        self.assertTrue(fake.physicals[0].closed)

    def test_the_out_of_memory_message_names_the_free_memory_and_the_cause(self):
        with _level_zero(create_fails_after=0) as fake:
            backend = _backend(fake)
            backend.free_bytes = 4096
            with self.assertRaises(RuntimeError) as caught:
                backend.prepare_export(_state(w=torch.ones(1024)))
        message = str(caught.exception)
        self.assertIn("4096 bytes free", message)
        self.assertIn("zePhysicalMemCreate", message)

    def test_an_allocation_the_driver_pads_is_refused(self):
        """The padded bytes would be allocated but never mapped, and the fd
        would carry a buffer larger than the size the engine imports it with."""
        with _level_zero() as fake:
            backend = _backend(fake)
            fake.pad_creates_to = 4 << 20
            with self.assertRaisesRegex(RuntimeError, "padded"):
                backend.prepare_export(_state(w=torch.ones(1024)))
        self.assertTrue(fake.physicals[0].closed)

    def test_each_fetch_exports_a_fresh_fd_per_allocation(self):
        with _level_zero() as fake:
            backend = _backend(fake, max_alloc_bytes=PAGE_SIZE)
            backend.prepare_export(
                _state(a=torch.ones(PAGE_SIZE // 4), b=torch.ones(PAGE_SIZE // 4))
            )
            self.assertEqual(backend.num_allocations, 2)
            first = backend._export_fds()
            second = backend._export_fds()
            try:
                self.assertEqual(len(first), 2)
                self.assertEqual(len(second), 2)
                self.assertFalse(set(first) & set(second))
            finally:
                for fd in first + second:
                    os.close(fd)
            backend.close()


@contextlib.contextmanager
def _exported_allocations(count=2, **kwargs):
    """A fake driver, ``count`` exported allocations, and one fd each.

    The fds are what an engine receives over SCM_RIGHTS; they stay the test's
    to close, as they are the transport's.
    """
    with _level_zero(**kwargs) as fake:
        physicals = [
            fake.PhysicalMemory.create(PAGE_SIZE, device_index=0) for _ in range(count)
        ]
        fds = [physical.export_dma_buf() for physical in physicals]
        try:
            yield fake, [physical.size for physical in physicals], fds
        finally:
            for fd in fds:
                os.close(fd)


class TestEngineImport(CustomTestCase):
    def test_misaligned_allocation_sizes_are_refused_before_any_import(self):
        with _exported_allocations(1) as (fake, sizes, fds):
            client = _backend(fake)
            with self.assertRaisesRegex(RuntimeError, "not multiples of"):
                client._map_imported(fds, [sizes[0] + SLOT_ALIGNMENT])
            self.assertEqual(fake.imported, [])

    def test_a_failed_map_leaves_neither_a_handle_nor_a_reservation(self):
        with _exported_allocations(2, create_fails_after=1) as (fake, sizes, fds):
            client = _backend(fake)
            with self.assertRaises(ZeVmmError):
                client._map_imported(fds, sizes)
            self.assertEqual(len(fake.imported), 2)
            self.assertTrue(all(p.closed for p in fake.imported))
            self.assertTrue(fake.reservations[-1].closed)
            self.assertIsNone(client._reservation)

    def test_a_successful_import_retains_the_handles_that_keep_the_buffer_alive(self):
        """On Level Zero the importer's own physical handle is what keeps the
        dma-buf alive, so unlike CUDA it cannot be released after mapping: that
        is what lets the engine outlive the daemon."""
        with _exported_allocations(2) as (fake, sizes, fds):
            client = _backend(fake)
            client._map_imported(fds, sizes)
            self.assertEqual(client._handles, fake.imported)
            self.assertFalse(any(p.closed for p in fake.imported))
            self.assertEqual(client._alloc_offsets, [0, PAGE_SIZE])
            with self.assertRaisesRegex(RuntimeError, "already holds a mapping"):
                client._map_imported(fds, sizes)
            client.close()
            self.assertTrue(fake.reservations[-1].closed)
            self.assertTrue(all(p.closed for p in fake.imported))

    def test_the_imported_fds_stay_the_callers_to_close(self):
        """from_dma_buf imports a dup, so the backend must leave the received
        fd open; its caller closes it, and a second close would be an error."""
        with _exported_allocations(1) as (fake, sizes, fds):
            client = _backend(fake)
            client._map_imported(fds, sizes)
            client.close()
            os.fstat(fds[0])  # raises if the backend closed it

    def test_close_is_idempotent_and_the_mapping_is_unusable_afterwards(self):
        with _exported_allocations(1) as (fake, sizes, fds):
            client = _backend(fake)
            client._map_imported(fds, sizes)
            client.close()
            client.close()
            with self.assertRaisesRegex(RuntimeError, "no allocation is mapped"):
                client.import_tensor(
                    {"alloc_index": 0, "offset": 0, "storage_nbytes": 4}
                )
            self.assertEqual(len(fake.reservations), 1)

    def test_a_handle_that_refuses_to_go_does_not_strand_the_others(self):
        with _exported_allocations(3) as (fake, sizes, fds):
            client = _backend(fake)
            client._map_imported(fds, sizes)
            fake.imported[1].close = mock.Mock(side_effect=ZeVmmError("handle in use"))
            with self.assertLogs(vmm_fd_xpu.logger, "WARNING"):
                client.close()
            self.assertTrue(fake.imported[0].closed)
            self.assertTrue(fake.imported[2].closed)


class TestRoundTripThroughTheXpuHooks(CustomTestCase):
    """The daemon's export, the fetch_state exchange and the engine's import,
    all through the XPU backend, with host memory behind the allocations.

    The engine maps the same host buffers the daemon packed, so the tensors it
    rebuilds must carry the daemon's values with the shape, dtype, stride and
    aliasing they were exported with.
    """

    def test_round_trip(self):
        torch.manual_seed(0)
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        state = {
            "w": (base, True),
            "w_t": (base.t(), True),  # a transposed view of the same storage
            "bias": (torch.randn(6), True),
            "cache": (torch.arange(12, dtype=torch.bfloat16).reshape(3, 4), False),
        }
        expected = {
            name: tensor.detach().clone() for name, (tensor, _) in state.items()
        }

        with _level_zero() as fake:
            daemon = _backend(fake, max_alloc_bytes=PAGE_SIZE)
            entries = daemon.prepare_export(dict(state))
            self.assertGreaterEqual(daemon.num_allocations, 1)

            a, b = socket.socketpair()
            try:
                daemon.send_fetch_state_response(
                    a,
                    config={"k": "v"},
                    entries=entries,
                    pid=11,
                )
                response = recv_msg(b)
                self.assertEqual(response["transport_backend"], VMM_FD_BACKEND)
                self.assertEqual(response["device_type"], "xpu")
                client = _backend(fake)
                response = client.recv_fetch_state_response(b, response)
                imported = {
                    name: client.import_tensor(entry)
                    for name, entry in response["entries"].items()
                }
                for name, tensor in imported.items():
                    self.assertEqual(tensor.shape, expected[name].shape, name)
                    self.assertEqual(tensor.dtype, expected[name].dtype, name)
                    self.assertEqual(tensor.stride(), expected[name].stride(), name)
                    torch.testing.assert_close(
                        tensor, expected[name], rtol=0, atol=0, check_device=False
                    )
                self.assertEqual(
                    imported["w"].untyped_storage().data_ptr(),
                    imported["w_t"].untyped_storage().data_ptr(),
                    "a transposed view must share its storage again",
                )
            finally:
                a.close()
                b.close()
            client.close()
            daemon.close()

    def test_an_engine_can_attach_again_after_detaching(self):
        """A restarting engine fetches from a daemon that stayed up, so each
        fetch must export fresh fds and map into a reservation of its own."""
        with _level_zero() as fake:
            daemon = _backend(fake)
            entries = daemon.prepare_export(_state(w=torch.ones(1024)))
            bases = []
            for _ in range(2):
                a, b = socket.socketpair()
                try:
                    daemon.send_fetch_state_response(
                        a, config={}, entries=entries, pid=11
                    )
                    client = _backend(fake)
                    client.recv_fetch_state_response(b, recv_msg(b))
                    bases.append(client._reservation_base)
                    torch.testing.assert_close(
                        client.import_tensor(entries["w"]),
                        torch.ones(1024),
                        rtol=0,
                        atol=0,
                        check_device=False,
                    )
                    client.close()
                finally:
                    a.close()
                    b.close()
            self.assertEqual(len(bases), 2)
            self.assertEqual(daemon.num_allocations, 1)
            self.assertFalse(fake.physicals[0].closed, "the daemon kept its allocation")
            daemon.close()


if __name__ == "__main__":
    unittest.main()
