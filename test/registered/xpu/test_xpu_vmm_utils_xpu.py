# SPDX-License-Identifier: Apache-2.0
"""Level Zero VMM hardware tests on Intel XPU.

Two processes share one physical allocation with no model loaded: the exporter
creates it, maps it at an address of its own choosing and writes a pattern; the
importer receives the dma-buf fd over a unix socket, reserves its own address
range and maps the same allocation there.

Byte equality alone would not show zero copy -- a copy satisfies it too. Two
further checks do:

- the dma-buf's ``(st_dev, st_ino)`` is the same in both processes, which is the
  kernel's own identity for the buffer, and
- a write one process makes after the other has already mapped is visible to the
  other, in both directions.

Sizes travel as ordinary metadata and only the fds go through ``send_fds``,
matching how ``CudaVmmFdTransportBackend`` splits the two on CUDA.

The CPU-only cases live in test/registered/unit/utils/test_xpu_vmm_utils.py.
"""

import multiprocessing as mp
import os
import socket
import tempfile
import unittest

from sglang.srt.utils import xpu_vmm_utils as vmm
from sglang.srt.weight_cache.transport import recv_fds, send_fds
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="extra-test-1-gpu-xpu")

_SIZE = 1 << 20
_CONNECT_TIMEOUT_S = 60.0
_JOIN_TIMEOUT_S = 180.0

_EXPORTER_PATTERN = bytes((i * 37 + 11) & 0xFF for i in range(4096))
_IMPORTER_PATTERN = bytes((i * 91 + 5) & 0xFF for i in range(4096))
_LATE_PATTERN = bytes((i * 13 + 3) & 0xFF for i in range(4096))


def _unsupported_reason():
    try:
        return vmm.vmm_unsupported_reason()
    except Exception as exc:  # the loader itself may be missing
        return str(exc)


_SKIP_REASON = _unsupported_reason()
requires_vmm = unittest.skipIf(
    _SKIP_REASON is not None, f"Level Zero VMM unavailable: {_SKIP_REASON}"
)


def _open_fds() -> int:
    # listdir allocates one descriptor for the directory itself; discount it.
    return len(os.listdir("/proc/self/fd")) - 1


def _receive_one_fd(path: str) -> int:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_TIMEOUT_S)
    sock.connect(path)
    try:
        (fd,) = recv_fds(sock, 1)
        return fd
    finally:
        sock.close()


def _importer(path: str, size: int, out, mapped, late_written) -> None:
    """Import the exporter's allocation, read it, write back, then tear down.

    ``mapped`` is set once the mapping exists and the reply has been written, so
    the exporter's late write provably happens after this process mapped.
    """
    fd = _receive_one_fd(path)
    physical = None
    reservation = None
    try:
        identity = vmm.dma_buf_identity(fd)
        out["identity"] = (identity.st_dev, identity.st_ino, identity.size)

        physical = vmm.PhysicalMemory.from_dma_buf(fd, size)
        reservation = vmm.VirtualReservation(size)
        address = reservation.map(physical)

        out["va"] = address
        out["access"] = reservation.access_at()[0]
        out["sees_exporter_pattern"] = (
            vmm.copy_from_device(address, len(_EXPORTER_PATTERN)) == _EXPORTER_PATTERN
        )
        vmm.copy_to_device(address, _IMPORTER_PATTERN)
        mapped.set()

        if late_written.wait(timeout=_JOIN_TIMEOUT_S):
            out["sees_late_pattern"] = (
                vmm.copy_from_device(address, len(_LATE_PATTERN)) == _LATE_PATTERN
            )
    finally:
        if reservation is not None:
            reservation.close()
        if physical is not None:
            physical.close()
        os.close(fd)
        out["cleaned_up"] = True
        mapped.set()


def _importer_once(path: str, size: int, out) -> None:
    """Receive, import, map, verify and tear down, reporting what happened."""
    fd = _receive_one_fd(path)
    physical = None
    reservation = None
    try:
        physical = vmm.PhysicalMemory.from_dma_buf(fd, size)
        reservation = vmm.VirtualReservation(size)
        address = reservation.map(physical)
        out["sees_exporter_pattern"] = (
            vmm.copy_from_device(address, len(_EXPORTER_PATTERN)) == _EXPORTER_PATTERN
        )
        out["error"] = None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if reservation is not None:
            reservation.close()
        if physical is not None:
            physical.close()
        os.close(fd)


class _Exporter:
    """An exported allocation holding ``_EXPORTER_PATTERN``, and a socket
    serving its fd."""

    def __init__(self, size: int = _SIZE):
        self.directory = tempfile.mkdtemp(prefix="sglang-xpu-vmm-")
        self.path = os.path.join(self.directory, "fd.sock")
        self.size = size
        self.physical = vmm.PhysicalMemory.create(size)
        self.reservation = vmm.VirtualReservation(size)
        self.address = self.reservation.map(self.physical)
        vmm.copy_to_device(self.address, _EXPORTER_PATTERN)
        self.fd = self.physical.export_dma_buf()
        self.identity = vmm.dma_buf_identity(self.fd)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.settimeout(_CONNECT_TIMEOUT_S)
        self.server.listen(1)

    def serve_one(self) -> None:
        conn, _ = self.server.accept()
        try:
            send_fds(conn, [self.fd])
        finally:
            conn.close()

    def close(self) -> None:
        self.reservation.close()
        self.physical.close()
        os.close(self.fd)
        self.server.close()
        if os.path.exists(self.path):
            os.unlink(self.path)
        os.rmdir(self.directory)


@requires_vmm
class TestXpuVmmTwoProcessRoundTrip(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = mp.get_context("spawn")
        cls.manager = cls.ctx.Manager()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "manager") and cls.manager is not None:
            cls.manager.shutdown()

    def _run_importer(self, exporter, target, args):
        proc = self.ctx.Process(target=target, args=args)
        proc.start()
        try:
            exporter.serve_one()
        except BaseException:
            proc.terminate()
            proc.join(_JOIN_TIMEOUT_S)
            raise
        return proc

    def test_two_processes_map_one_physical_allocation_at_different_addresses(self):
        exporter = _Exporter()
        out = self.manager.dict()
        mapped = self.ctx.Event()
        late_written = self.ctx.Event()
        proc = None
        try:
            proc = self._run_importer(
                exporter,
                _importer,
                (exporter.path, exporter.size, out, mapped, late_written),
            )
            self.assertTrue(
                mapped.wait(timeout=_JOIN_TIMEOUT_S),
                f"importer never mapped; exitcode {proc.exitcode}, got {dict(out)}",
            )

            self.assertEqual(
                tuple(out["identity"]),
                (
                    exporter.identity.st_dev,
                    exporter.identity.st_ino,
                    exporter.identity.size,
                ),
                "importer holds a different dma-buf, so the allocation is not shared",
            )
            self.assertNotEqual(
                out["va"],
                exporter.address,
                "both processes mapped at the same address, so the test shows "
                "nothing about independent reservations",
            )
            self.assertTrue(out["sees_exporter_pattern"])
            self.assertEqual(out["access"], vmm.ACCESS_READWRITE)
            self.assertEqual(
                vmm.copy_from_device(exporter.address, len(_IMPORTER_PATTERN)),
                _IMPORTER_PATTERN,
                "the importer's write is not visible to the exporter",
            )

            # Written after the importer mapped, so a one-time copy at import
            # time cannot account for the importer seeing it.
            vmm.copy_to_device(exporter.address, _LATE_PATTERN)
            late_written.set()

            proc.join(_JOIN_TIMEOUT_S)
            self.assertEqual(proc.exitcode, 0)
            self.assertTrue(
                out.get("sees_late_pattern"),
                "a write made after the importer mapped was not visible to it",
            )
            self.assertTrue(out["cleaned_up"])
        finally:
            late_written.set()
            if proc is not None:
                proc.join(_JOIN_TIMEOUT_S)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(_JOIN_TIMEOUT_S)
            exporter.close()

    def test_repeated_round_trips_leak_no_descriptors(self):
        # One dict and one warm-up round, so the baseline excludes the
        # multiprocessing manager's own connections and leaves only this
        # module's descriptors to account for.
        out = self.manager.dict()
        baseline = None
        for round_index in range(8):
            exporter = _Exporter()
            proc = None
            try:
                proc = self._run_importer(
                    exporter, _importer_once, (exporter.path, exporter.size, out)
                )
                proc.join(_JOIN_TIMEOUT_S)
                self.assertEqual(proc.exitcode, 0)
                self.assertIsNone(out["error"])
                self.assertTrue(out["sees_exporter_pattern"])
            finally:
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(_JOIN_TIMEOUT_S)
                exporter.close()
            if round_index == 0:
                baseline = _open_fds()
        self.assertLessEqual(
            _open_fds() - baseline,
            2,
            "repeated export/import cycles are leaking descriptors",
        )

    def test_importer_refuses_a_size_the_exporter_did_not_allocate(self):
        exporter = _Exporter()
        out = self.manager.dict()
        proc = None
        try:
            proc = self._run_importer(
                exporter, _importer_once, (exporter.path, exporter.size + 1, out)
            )
            proc.join(_JOIN_TIMEOUT_S)
            self.assertEqual(proc.exitcode, 0)
            self.assertIsNotNone(out["error"], "an unaligned size was accepted")
            self.assertIn("page size", out["error"])
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(_JOIN_TIMEOUT_S)
            exporter.close()

    def test_importer_sees_a_closed_socket_as_an_error(self):
        exporter = _Exporter()
        try:
            exporter.server.close()
            with self.assertRaises(
                (ConnectionRefusedError, FileNotFoundError, OSError)
            ):
                _receive_one_fd(exporter.path)
        finally:
            exporter.reservation.close()
            exporter.physical.close()
            os.close(exporter.fd)
            if os.path.exists(exporter.path):
                os.unlink(exporter.path)
            os.rmdir(exporter.directory)


@requires_vmm
class TestXpuVmmLocalContracts(CustomTestCase):
    def test_page_size_is_reported_and_sizes_round_up_to_it(self):
        page = vmm.query_page_size(_SIZE)
        self.assertGreater(page, 0)
        self.assertEqual(page & (page - 1), 0, "page size is not a power of two")
        self.assertEqual(vmm.aligned_size(1), page)
        self.assertEqual(vmm.aligned_size(page + 1), 2 * page)

    def test_create_pads_the_allocation_to_a_whole_page(self):
        page = vmm.query_page_size(_SIZE)
        with vmm.PhysicalMemory.create(page + 1) as physical:
            self.assertEqual(physical.size, 2 * page)
            self.assertEqual(physical.reported_size(), 2 * page)

    def test_a_non_exportable_allocation_refuses_to_export(self):
        with vmm.PhysicalMemory.create(_SIZE, exportable=False) as physical:
            with self.assertRaises(vmm.ZeVmmError):
                physical.export_dma_buf()

    def test_each_export_is_a_distinct_fd_onto_one_buffer(self):
        with vmm.PhysicalMemory.create(_SIZE) as physical:
            first = physical.export_dma_buf()
            second = physical.export_dma_buf()
            try:
                self.assertNotEqual(first, second)
                self.assertEqual(
                    vmm.dma_buf_identity(first), vmm.dma_buf_identity(second)
                )
            finally:
                os.close(first)
                os.close(second)

    def test_access_attribute_is_set_and_read_back(self):
        with vmm.PhysicalMemory.create(_SIZE) as physical:
            with vmm.VirtualReservation(_SIZE) as reservation:
                reservation.map(physical, access=vmm.ACCESS_READWRITE)
                self.assertEqual(reservation.access_at()[0], vmm.ACCESS_READWRITE)
                reservation.set_access(vmm.ACCESS_READONLY)
                self.assertEqual(reservation.access_at()[0], vmm.ACCESS_READONLY)
                reservation.set_access(vmm.ACCESS_READWRITE)
                self.assertEqual(reservation.access_at()[0], vmm.ACCESS_READWRITE)

    def test_several_allocations_map_into_one_reservation(self):
        page = vmm.query_page_size(_SIZE)
        parts = [vmm.PhysicalMemory.create(page) for _ in range(4)]
        reservation = vmm.VirtualReservation(4 * page)
        try:
            for index, part in enumerate(parts):
                reservation.map(part, offset=index * page, size=page)
            self.assertEqual(reservation.num_mappings, 4)
            payload = bytes((i * 7 + 1) & 0xFF for i in range(4 * page))
            vmm.copy_to_device(reservation.ptr, payload)
            self.assertEqual(vmm.copy_from_device(reservation.ptr, 4 * page), payload)
        finally:
            reservation.close()
            for part in parts:
                part.close()

    def test_mapping_rejects_an_unaligned_or_out_of_range_span(self):
        with vmm.PhysicalMemory.create(_SIZE) as physical:
            with vmm.VirtualReservation(_SIZE) as reservation:
                with self.assertRaises(ValueError):
                    reservation.map(physical, offset=1)
                with self.assertRaises(ValueError):
                    reservation.map(physical, offset=_SIZE)
                with self.assertRaises(ValueError):
                    reservation.map(physical, size=2 * _SIZE)
                with self.assertRaises(ValueError):
                    reservation.map(physical, physical_offset=_SIZE)

    def test_importing_a_descriptor_that_is_not_a_dma_buf_fails(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(vmm.ZeVmmError):
                vmm.PhysicalMemory.from_dma_buf(read_fd, _SIZE)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_import_leaves_the_callers_descriptor_usable(self):
        """The driver takes the fd it is given, so from_dma_buf must pass a dup."""
        with vmm.PhysicalMemory.create(_SIZE) as physical:
            fd = physical.export_dma_buf()
            try:
                identity = vmm.dma_buf_identity(fd)
                imported = vmm.PhysicalMemory.from_dma_buf(fd, _SIZE)
                imported.close()
                self.assertEqual(vmm.dma_buf_identity(fd), identity)
            finally:
                os.close(fd)

    def test_a_size_the_driver_would_abort_on_is_refused(self):
        """Above 16 TiB zePhysicalMemCreate calls abort() instead of returning
        an error, which would take the daemon down with SIGABRT."""
        with self.assertRaises(ValueError):
            vmm.PhysicalMemory.create(vmm._PHYSICAL_MEM_MAX_BYTES + 1)

    def test_a_size_past_device_memory_is_a_clean_error(self):
        """A create success is not a capacity check, but an unbackable size must
        still come back as an exception rather than a crash."""
        with self.assertRaises(vmm.ZeVmmError):
            vmm.PhysicalMemory.create(1 << 42)

    def test_an_out_of_range_device_index_is_refused(self):
        with self.assertRaises(vmm.ZeVmmError):
            vmm.PhysicalMemory.create(_SIZE, device_index=vmm.device_count() + 16)

    def test_closed_objects_refuse_use_and_close_is_idempotent(self):
        physical = vmm.PhysicalMemory.create(_SIZE)
        reservation = vmm.VirtualReservation(_SIZE)
        reservation.map(physical)
        reservation.close()
        physical.close()

        reservation.close()
        physical.close()

        self.assertTrue(physical.closed)
        self.assertTrue(reservation.closed)
        with self.assertRaises(vmm.ZeVmmError):
            physical.handle
        with self.assertRaises(vmm.ZeVmmError):
            reservation.ptr

    def test_an_importer_mapping_outlives_the_exporters_teardown(self):
        exporter = vmm.PhysicalMemory.create(_SIZE)
        exporter_reservation = vmm.VirtualReservation(_SIZE)
        exporter_address = exporter_reservation.map(exporter)
        vmm.copy_to_device(exporter_address, _EXPORTER_PATTERN)
        fd = exporter.export_dma_buf()

        imported = vmm.PhysicalMemory.from_dma_buf(fd, _SIZE)
        imported_reservation = vmm.VirtualReservation(_SIZE)
        imported_address = imported_reservation.map(imported)
        try:
            exporter_reservation.close()
            exporter.close()
            os.close(fd)
            self.assertEqual(
                vmm.copy_from_device(imported_address, len(_EXPORTER_PATTERN)),
                _EXPORTER_PATTERN,
                "the dma-buf reference did not keep the allocation alive",
            )
        finally:
            imported_reservation.close()
            imported.close()

    def test_destroying_a_still_mapped_allocation_is_refused(self):
        """Arc Pro B60 lets zePhysicalMemDestroy through while a mapping is live;
        reading the stale mapping then returns changed data and faults."""
        physical = vmm.PhysicalMemory.create(_SIZE)
        reservation = vmm.VirtualReservation(_SIZE)
        try:
            reservation.map(physical)
            self.assertEqual(physical.mapped_count, 1)
            with self.assertRaises(vmm.ZeVmmError):
                physical.close()
            self.assertFalse(physical.closed, "the allocation was destroyed anyway")

            reservation.close()
            self.assertEqual(physical.mapped_count, 0)
            physical.close()
            self.assertTrue(physical.closed)
        finally:
            reservation.close()
            if not physical.closed:
                physical.close()

    def test_an_allocation_is_remappable_once_its_mapping_is_released(self):
        """Arc Pro B60 permits one live mapping per physical handle, so the
        count a close releases has to be accurate for the next map to work."""
        page = vmm.query_page_size(_SIZE)
        physical = vmm.PhysicalMemory.create(page)
        first = vmm.VirtualReservation(page)
        try:
            first_address = first.map(physical)
            self.assertEqual(physical.mapped_count, 1)
            vmm.copy_to_device(first_address, _EXPORTER_PATTERN)

            first.close()
            self.assertEqual(physical.mapped_count, 0)

            second = vmm.VirtualReservation(page)
            try:
                second_address = second.map(physical)
                self.assertEqual(physical.mapped_count, 1)
                self.assertEqual(
                    vmm.copy_from_device(second_address, len(_EXPORTER_PATTERN)),
                    _EXPORTER_PATTERN,
                    "remapping lost the allocation's contents",
                )
            finally:
                second.close()
            self.assertEqual(physical.mapped_count, 0)
        finally:
            first.close()
            physical.close()

    def test_a_non_zero_physical_offset_maps(self):
        page = vmm.query_page_size(_SIZE)
        physical = vmm.PhysicalMemory.create(2 * page)
        reservation = vmm.VirtualReservation(page)
        try:
            address = reservation.map(physical, size=page, physical_offset=page)
            vmm.copy_to_device(address, _EXPORTER_PATTERN)
            self.assertEqual(
                vmm.copy_from_device(address, len(_EXPORTER_PATTERN)),
                _EXPORTER_PATTERN,
            )
            self.assertEqual(physical.mapped_count, 1)
        finally:
            reservation.close()
            physical.close()

    def test_repeated_local_cycles_leak_no_descriptors(self):
        baseline = _open_fds()
        for _ in range(30):
            physical = vmm.PhysicalMemory.create(_SIZE)
            fd = physical.export_dma_buf()
            imported = vmm.PhysicalMemory.from_dma_buf(fd, _SIZE)
            reservation = vmm.VirtualReservation(_SIZE)
            address = reservation.map(imported)
            vmm.copy_to_device(address, _EXPORTER_PATTERN)
            self.assertEqual(
                vmm.copy_from_device(address, len(_EXPORTER_PATTERN)),
                _EXPORTER_PATTERN,
            )
            reservation.close()
            imported.close()
            os.close(fd)
            physical.close()
        self.assertLessEqual(
            _open_fds() - baseline, 2, "export/import cycles are leaking descriptors"
        )

    def test_a_partially_initialised_reservation_frees_cleanly(self):
        """A failed map must leave close able to free the reservation."""
        physical = vmm.PhysicalMemory.create(_SIZE)
        reservation = vmm.VirtualReservation(_SIZE)
        try:
            with self.assertRaises(ValueError):
                reservation.map(physical, offset=1)
            self.assertEqual(reservation.num_mappings, 0)
        finally:
            reservation.close()
            physical.close()


if __name__ == "__main__":
    unittest.main(verbosity=3)
