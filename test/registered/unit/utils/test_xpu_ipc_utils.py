"""How an XPU IPC handle and its dma_buf fd reach an importing process.

None of these tests needs an Intel GPU: the handle server only moves bytes and
an fd, so /dev/null stands in for the exported dma_buf. The device round trip
lives in test/registered/xpu/test_xpu_ipc_utils_xpu.py.
"""

import asyncio
import os
import resource
import socket
import struct
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import torch

from sglang.srt.utils import xpu_ipc_utils
from sglang.srt.utils.xpu_ipc_utils import (
    XPU_IPC_HANDLE_BYTES,
    AddressRange,
    ExportedSegment,
    XpuIpcError,
    XpuIpcExportedPool,
    XpuIpcHandleServer,
    XpuIpcPeerPools,
    XpuIpcPoolReference,
    ensure_xpu_peer_access,
    fetch_xpu_ipc_handle,
    importable_descriptor,
    is_xpu_ipc_available,
    wait_for_xpu_event,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PEER_TIMEOUT_S = 60
# Well under the server's request timeout, which a silent peer could hold everyone for.
_PROMPT_S = 1.0


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


def _connect_silently(server: XpuIpcHandleServer) -> socket.socket:
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    peer.connect("\0" + server.socket_name)
    return peer


def _serve_one_reply(handle_blob: bytes, descriptor_count: int):
    """A stand-in server that answers one request with the given reply."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    socket_name = f"sglang-xpu-ipc-test-{os.getpid()}-{time.monotonic_ns()}"
    listener.bind("\0" + socket_name)
    listener.listen(1)

    def reply() -> None:
        with listener:
            connection, _ = listener.accept()
            with connection:
                connection.recv(1024)
                descriptors = [
                    os.open("/dev/null", os.O_RDONLY) for _ in range(descriptor_count)
                ]
                try:
                    socket.send_fds(connection, [handle_blob], descriptors)
                finally:
                    for descriptor in descriptors:
                        os.close(descriptor)

    replier = threading.Thread(target=reply, daemon=True)
    replier.start()
    return socket_name, replier


class TestXpuIpcHandleServer(CustomTestCase):
    def setUp(self):
        self.handle_blob = bytes(range(XPU_IPC_HANDLE_BYTES))
        self.exported_fd = os.open("/dev/null", os.O_RDONLY)
        self.addCleanup(os.close, self.exported_fd)
        self.segment = ExportedSegment(
            base_pointer=0,
            segment_bytes=XPU_IPC_HANDLE_BYTES,
            handle_blob=self.handle_blob,
            file_descriptor=self.exported_fd,
        )

    def _start_server(self) -> XpuIpcHandleServer:
        server = XpuIpcHandleServer(self.segment)
        # Closing twice is harmless, so tests that close early still get cleaned up.
        self.addCleanup(server.close)
        return server

    def test_hands_over_the_handle_and_a_working_descriptor(self):
        server = self._start_server()
        handle_blob, received_fd = fetch_xpu_ipc_handle(
            server.socket_name, server.segment_id
        )
        try:
            self.assertEqual(handle_blob, self.handle_blob)
            self.assertNotEqual(received_fd, self.exported_fd)
            self.assertEqual(
                os.fstat(received_fd).st_ino, os.fstat(self.exported_fd).st_ino
            )
        finally:
            os.close(received_fd)

    def test_rejected_requests_keep_no_descriptor_and_do_not_stop_serving(self):
        server = self._start_server()
        before = _open_descriptors()
        with self.assertRaises(XpuIpcError):
            fetch_xpu_ipc_handle(server.socket_name, "another-segment")
        self.assertEqual(_open_descriptors(), before)

        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.settimeout(_PEER_TIMEOUT_S)
            connection.connect("\0" + server.socket_name)
            # Not UTF-8: the server must still log the request and hang up.
            connection.send(b"\xff")
            self.assertEqual(connection.recv(XPU_IPC_HANDLE_BYTES), b"")

        handle_blob, received_fd = fetch_xpu_ipc_handle(
            server.socket_name, server.segment_id
        )
        os.close(received_fd)
        self.assertEqual(handle_blob, self.handle_blob)

    def test_each_server_mints_its_own_unguessable_segment_id(self):
        """The id is what keeps the fd from a same-uid process that can read the
        socket name, so it must not be predictable."""
        first = self._start_server()
        second = self._start_server()
        self.assertNotEqual(first.segment_id, second.segment_id)
        self.assertEqual(len(first.segment_id), 32)
        int(first.segment_id, 16)

    def test_a_failed_accept_does_not_stop_serving(self):
        """Running out of fds is transient; the segment has to stay reachable."""
        server = self._start_server()
        accept = socket.socket.accept
        failures = [OSError(24, "Too many open files")]

        def accept_after_a_failure(listener):
            if failures:
                raise failures.pop()
            return accept(listener)

        with (
            patch.object(socket.socket, "accept", accept_after_a_failure),
            self.assertLogs(xpu_ipc_utils.logger, level="WARNING"),
        ):
            handle_blob, received_fd = fetch_xpu_ipc_handle(
                server.socket_name, server.segment_id
            )
        os.close(received_fd)
        self.assertEqual(handle_blob, self.handle_blob)
        self.assertEqual(failures, [])
        self.assertTrue(server.thread.is_alive())

    def test_closing_waits_for_the_peer_being_served(self):
        """The exporter releases the segment as soon as close returns, so by then
        the server must not be able to send its fd any more."""
        server = self._start_server()
        serving = threading.Event()
        events = []

        def serve_slowly(connection: socket.socket) -> None:
            serving.set()
            time.sleep(0.5)
            events.append("served")

        server._send_handle = serve_slowly
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as peer:
            peer.connect("\0" + server.socket_name)
            peer.send(server.segment_id.encode())
            self.assertTrue(serving.wait(_PEER_TIMEOUT_S))
            server.close()
            events.append("closed")
        self.assertEqual(events, ["served", "closed"])
        self.assertFalse(server.thread.is_alive())
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as late_peer:
            with self.assertRaises(ConnectionRefusedError):
                late_peer.connect("\0" + server.socket_name)

    def test_a_silent_peer_does_not_hold_up_an_importer(self):
        server = self._start_server()
        # More silent peers than any fixed pool of workers could have held.
        silent_peers = [_connect_silently(server) for _ in range(8)]
        for peer in silent_peers:
            self.addCleanup(peer.close)
        start = time.monotonic()
        _, received_fd = fetch_xpu_ipc_handle(server.socket_name, server.segment_id)
        os.close(received_fd)
        self.assertLess(time.monotonic() - start, _PROMPT_S)

    def test_closing_does_not_wait_for_a_silent_peer(self):
        server = self._start_server()
        silent_peer = _connect_silently(server)
        self.addCleanup(silent_peer.close)
        # Let the server accept the peer and start waiting for its request.
        time.sleep(0.3)
        start = time.monotonic()
        server.close()
        self.assertLess(time.monotonic() - start, _PROMPT_S)
        self.assertFalse(server.thread.is_alive())

    def test_a_peer_of_another_user_is_turned_away_at_once(self):
        server = self._start_server()
        with patch.object(
            xpu_ipc_utils.os, "geteuid", side_effect=lambda: os.getuid() + 1
        ):
            start = time.monotonic()
            # The hang-up reads as a reset or an empty reply, depending on timing.
            with self.assertRaises((ConnectionError, XpuIpcError)):
                fetch_xpu_ipc_handle(server.socket_name, server.segment_id)
            self.assertLess(time.monotonic() - start, _PROMPT_S)

    def test_a_uid_past_the_signed_range_is_still_the_own_user(self):
        """uid_t is unsigned, so a uid above 2**31 must not read as negative."""
        uid = 3_000_000_000
        server = XpuIpcHandleServer.__new__(XpuIpcHandleServer)
        server.socket_name = "unused"
        peer = SimpleNamespace(
            getsockopt=lambda level, option, buffer_bytes: struct.pack(
                "iII", 4242, uid, uid
            )
        )
        with patch.object(xpu_ipc_utils.os, "geteuid", return_value=uid):
            self.assertTrue(server._is_own_user(peer))

    def test_a_peer_dropped_while_its_request_is_pending_does_not_stop_serving(
        self,
    ):
        """Admitting a peer into a full waiting set drops the oldest, whose request
        may sit in the same batch of ready sockets; handling that stale entry must
        not end the only thread serving the segment."""
        with patch.object(xpu_ipc_utils, "_SERVER_MAX_WAITING_PEERS", 2):
            server = self._start_server()
            admitting = threading.Event()
            is_own_user = server._is_own_user

            def admit_slowly(connection: socket.socket) -> bool:
                # Lets the next connect and the oldest peer's request share a batch.
                admitting.set()
                time.sleep(0.3)
                server._is_own_user = is_own_user
                return is_own_user(connection)

            oldest = _connect_silently(server)
            self.addCleanup(oldest.close)
            time.sleep(0.2)
            server._is_own_user = admit_slowly
            filling = _connect_silently(server)
            self.addCleanup(filling.close)
            self.assertTrue(admitting.wait(_PEER_TIMEOUT_S))
            newest = _connect_silently(server)
            self.addCleanup(newest.close)
            oldest.send(server.segment_id.encode())
            time.sleep(0.6)

            self.assertTrue(server.thread.is_alive())
            _, received_fd = fetch_xpu_ipc_handle(server.socket_name, server.segment_id)
            os.close(received_fd)


class TestFetchXpuIpcHandle(CustomTestCase):
    def test_a_malformed_reply_is_refused_without_keeping_a_descriptor(self):
        """An extra fd or an oversized handle arrives truncated; whatever did
        arrive with a refused reply is closed, not leaked."""
        cases = {
            "short-handle": (XPU_IPC_HANDLE_BYTES // 2, 1),
            "no-descriptor": (XPU_IPC_HANDLE_BYTES, 0),
            "extra-descriptor": (XPU_IPC_HANDLE_BYTES, 2),
            "oversized-handle": (2 * XPU_IPC_HANDLE_BYTES, 1),
        }
        for name, (handle_bytes, descriptor_count) in cases.items():
            with self.subTest(name):
                # Counted before the stand-in opens its sockets and after it closes them.
                before = _open_descriptors()
                socket_name, replier = _serve_one_reply(
                    bytes(handle_bytes), descriptor_count
                )
                with self.assertRaises(XpuIpcError):
                    fetch_xpu_ipc_handle(socket_name, "0" * 32)
                replier.join(_PEER_TIMEOUT_S)
                self.assertEqual(_open_descriptors(), before)


class TestImportableDescriptor(CustomTestCase):
    def test_a_received_descriptor_is_duplicated_only_when_it_collides(self):
        received = os.open("/dev/null", os.O_RDONLY)
        self.addCleanup(os.close, received)
        self.assertEqual(importable_descriptor(received, received + 1), received)
        # An exporter can hold more fds than this process may open.
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        self.assertEqual(importable_descriptor(received, hard_limit + 1), received)

        duplicate = importable_descriptor(received, received)
        try:
            self.assertNotEqual(duplicate, received)
            self.assertEqual(os.fstat(duplicate).st_ino, os.fstat(received).st_ino)
        finally:
            os.close(duplicate)


class TestIsXpuIpcAvailable(CustomTestCase):
    def setUp(self):
        is_xpu_ipc_available.cache_clear()
        self.addCleanup(is_xpu_ipc_available.cache_clear)

    def test_a_build_without_xpu_is_unavailable(self):
        with patch.object(torch.version, "xpu", None):
            self.assertFalse(is_xpu_ipc_available())

    def test_the_driver_must_see_the_devices_torch_does_and_be_new_enough(self):
        def driver(devices=(10, 11), build=xpu_ipc_utils._MIN_DRIVER_BUILD):
            return xpu_ipc_utils._LevelZeroDriver(
                handle=1, devices=devices, build=build
            )

        cases = {
            "no-driver": (None, 0, True, False),
            "no-devices": (driver(devices=()), 0, True, False),
            "fewer-torch-devices": (driver(), 1, True, False),
            "no-dma-buf-export": (driver(), 2, False, False),
            # 26.14, whose import hands a rewritten fd to pidfd_getfd.
            "driver-before-26.18": (driver(build=37833), 2, True, False),
            "available": (driver(), 2, True, True),
        }
        for name, (found, torch_devices, exports, expected) in cases.items():
            with self.subTest(name):
                is_xpu_ipc_available.cache_clear()
                with (
                    patch.object(torch.version, "xpu", "20260000"),
                    patch.object(
                        xpu_ipc_utils, "_level_zero_driver", return_value=found
                    ),
                    patch.object(
                        xpu_ipc_utils, "_exports_dma_buf", return_value=exports
                    ),
                    patch.object(torch.xpu, "device_count", return_value=torch_devices),
                ):
                    self.assertEqual(is_xpu_ipc_available(), expected)

    def test_a_missing_loader_is_unavailable(self):
        with (
            patch.object(torch.version, "xpu", "20260000"),
            patch.object(
                xpu_ipc_utils,
                "_level_zero_driver",
                side_effect=OSError("libze_loader.so.1: cannot open shared object"),
            ),
        ):
            self.assertFalse(is_xpu_ipc_available())


_POOL_BYTES = 1 << 20


def _host_tensor(pointer: int, num_bytes: int, device_index: int):
    """Stands in for tensor_from_device_pointer, which needs an XPU."""
    return torch.zeros(num_bytes, dtype=torch.uint8)


class _FakeServer:
    def __init__(self, segment):
        self.segment_id = "fake"
        self.socket_name = "fake-socket"


class TestXpuIpcExportedPool(CustomTestCase):
    def setUp(self):
        self.events = []
        self.exported_fd = os.open("/dev/null", os.O_RDONLY)
        self.addCleanup(os.close, self.exported_fd)
        patcher = patch.object(
            xpu_ipc_utils, "tensor_from_device_pointer", side_effect=_host_tensor
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _context(self, **overrides):
        pointers = iter((4096, 8192))

        def export_segment(pointer):
            # The allocation starts 64 bytes before the pointer the driver gave.
            return ExportedSegment(
                base_pointer=pointer - 64,
                segment_bytes=_POOL_BYTES + 64,
                handle_blob=bytes(XPU_IPC_HANDLE_BYTES),
                file_descriptor=self.exported_fd,
            )

        methods = {
            "allocate_exportable": lambda device_index, num_bytes: next(pointers),
            "export_segment": export_segment,
            "release_segment": lambda segment: self.events.append(
                f"released {segment.base_pointer + 64}"
            ),
            "free": lambda pointer: self.events.append(f"freed {pointer}"),
        }
        methods.update(overrides)
        return SimpleNamespace(**methods)

    def test_the_reference_lets_an_importer_reach_the_pool(self):
        pool = XpuIpcExportedPool(self._context(), 3, _POOL_BYTES)
        self.addCleanup(pool.close)
        self.assertEqual(pool.tensor.numel(), _POOL_BYTES)
        self.assertEqual(pool.reference.segment_offset, 64)
        self.assertEqual(pool.reference.pool_bytes, _POOL_BYTES)
        self.assertEqual(pool.reference.device_index, 3)
        handle_blob, received_fd = fetch_xpu_ipc_handle(
            pool.reference.socket_name, pool.reference.segment_id
        )
        os.close(received_fd)
        self.assertEqual(handle_blob, bytes(XPU_IPC_HANDLE_BYTES))

    def test_a_pool_that_fails_to_come_up_gives_back_what_it_made(self):
        """A caller retries after a failure, so every attempt must free its memory."""
        for failing_step in ("wrap", "export", "server"):
            with self.subTest(failing_step):
                self.events.clear()
                failure = RuntimeError(f"{failing_step} failed")
                overrides = {}
                patches = []
                if failing_step == "wrap":
                    patches.append(
                        patch.object(
                            xpu_ipc_utils,
                            "tensor_from_device_pointer",
                            side_effect=failure,
                        )
                    )
                elif failing_step == "export":
                    overrides["export_segment"] = Mock(side_effect=failure)
                else:
                    patches.append(
                        patch.object(
                            xpu_ipc_utils, "XpuIpcHandleServer", side_effect=failure
                        )
                    )
                context = self._context(**overrides)
                with patches[0] if patches else nullcontext():
                    for _ in range(2):
                        with self.assertRaisesRegex(RuntimeError, failure.args[0]):
                            XpuIpcExportedPool(context, 0, _POOL_BYTES)
                exported = failing_step == "server"
                self.assertEqual(
                    self.events,
                    [
                        *(["released 4096"] if exported else []),
                        "freed 4096",
                        *(["released 8192"] if exported else []),
                        "freed 8192",
                    ],
                )

    def test_the_memory_is_freed_even_when_releasing_its_export_fails(self):
        context = self._context(release_segment=Mock(side_effect=RuntimeError("no")))
        with patch.object(
            xpu_ipc_utils, "XpuIpcHandleServer", side_effect=RuntimeError("server")
        ):
            with self.assertRaisesRegex(RuntimeError, "no"):
                XpuIpcExportedPool(context, 0, _POOL_BYTES)
        self.assertEqual(self.events, ["freed 4096"])

    def test_close_attempts_every_step_once(self):
        """A step close skips is never retried, so its handle or memory would leak."""
        context = self._context(release_segment=Mock(side_effect=RuntimeError("no")))
        with patch.object(xpu_ipc_utils, "XpuIpcHandleServer", _FakeServer):
            pool = XpuIpcExportedPool(context, 0, _POOL_BYTES)
        pool._server.close = lambda: self.events.append("server closed")

        with self.assertRaisesRegex(RuntimeError, "no"):
            pool.close()
        pool.close()
        self.assertEqual(self.events, ["server closed", "freed 4096"])
        self.assertIsNone(pool.tensor)


class TestXpuIpcPeerPools(CustomTestCase):
    def setUp(self):
        exported_fd = os.open("/dev/null", os.O_RDONLY)
        self.addCleanup(os.close, exported_fd)
        segment = ExportedSegment(
            base_pointer=0,
            segment_bytes=_POOL_BYTES,
            handle_blob=bytes(XPU_IPC_HANDLE_BYTES),
            file_descriptor=exported_fd,
        )
        server = XpuIpcHandleServer(segment)
        self.addCleanup(server.close)
        self.reference = XpuIpcPoolReference(
            socket_name=server.socket_name,
            segment_id=server.segment_id,
            segment_offset=0,
            pool_bytes=_POOL_BYTES,
            device_index=0,
        )
        self.opened = []
        self.unmapped = []
        self.context = SimpleNamespace(
            open_peer_segment=self._open_peer_segment,
            address_range=lambda pointer: AddressRange(
                base_pointer=pointer, segment_bytes=_POOL_BYTES
            ),
            close_peer_segment=self.unmapped.append,
        )
        patcher = patch.object(
            xpu_ipc_utils, "tensor_from_device_pointer", side_effect=_host_tensor
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _open_peer_segment(self, handle_blob, descriptor, device_index):
        self.opened.append(device_index)
        return 7 + len(self.opened)

    def test_a_pool_is_mapped_once_and_keeps_no_descriptor(self):
        pools = XpuIpcPeerPools(self.context)
        before = _open_descriptors()
        first = pools.open(self.reference, 1)
        again = pools.open(self.reference, 1)
        self.assertIs(first, again)
        self.assertEqual(self.opened, [1])
        self.assertEqual(_open_descriptors(), before)
        pools.close()
        self.assertEqual(self.unmapped, [8])

    def test_a_mapped_pool_is_not_reached_through_another_reference(self):
        pools = XpuIpcPeerPools(self.context)
        pools.open(self.reference, 1)
        wider = msgspec.structs.replace(self.reference, pool_bytes=2 * _POOL_BYTES)
        with self.assertRaises(ValueError):
            pools.open(wider, 1)
        with self.assertRaises(ValueError):
            pools.open(self.reference, 2)
        self.assertEqual(self.opened, [1])

    def test_a_window_outside_the_mapping_is_refused_and_unmapped(self):
        """The window comes from the exporter's message, and a wrong one would let
        torch read or write past the mapping."""
        cases = {
            "negative-offset": (-64, _POOL_BYTES),
            "past-the-end": (0, 2 * _POOL_BYTES),
            "offset-past-the-end": (_POOL_BYTES, 64),
            "empty": (0, 0),
        }
        for name, (segment_offset, pool_bytes) in cases.items():
            with self.subTest(name):
                self.unmapped.clear()
                reference = msgspec.structs.replace(
                    self.reference, segment_offset=segment_offset, pool_bytes=pool_bytes
                )
                pools = XpuIpcPeerPools(self.context)
                with self.assertRaisesRegex(ValueError, "claims bytes"):
                    pools.open(reference, 0)
                self.assertEqual(len(self.unmapped), 1)
                pools.close()
                self.assertEqual(len(self.unmapped), 1)

    def test_a_mapping_torch_cannot_wrap_is_unmapped(self):
        pools = XpuIpcPeerPools(self.context)
        with patch.object(
            xpu_ipc_utils,
            "tensor_from_device_pointer",
            side_effect=RuntimeError("no wrap"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no wrap"):
                pools.open(self.reference, 0)
        self.assertEqual(self.unmapped, [8])

    def test_close_unmaps_every_pool_even_when_one_fails(self):
        other_fd = os.open("/dev/null", os.O_RDONLY)
        self.addCleanup(os.close, other_fd)
        other = XpuIpcHandleServer(
            ExportedSegment(
                base_pointer=0,
                segment_bytes=_POOL_BYTES,
                handle_blob=bytes(XPU_IPC_HANDLE_BYTES),
                file_descriptor=other_fd,
            ),
        )
        self.addCleanup(other.close)
        pools = XpuIpcPeerPools(self.context)
        pools.open(self.reference, 0)
        pools.open(
            msgspec.structs.replace(
                self.reference,
                socket_name=other.socket_name,
                segment_id=other.segment_id,
            ),
            0,
        )
        unmapped = []

        def close_peer_segment(pointer):
            if pointer == 8:
                raise RuntimeError("unmap failed")
            unmapped.append(pointer)

        self.context.close_peer_segment = close_peer_segment
        with self.assertRaisesRegex(RuntimeError, "unmap failed"):
            pools.close()
        self.assertEqual(unmapped, [9])
        pools.close()


class TestEnsureXpuPeerAccess(CustomTestCase):
    def test_only_a_copy_the_device_cannot_do_is_refused(self):
        with (
            patch.object(torch.xpu, "device_count", return_value=4),
            patch.object(
                torch.xpu, "can_device_access_peer", side_effect=lambda a, b: b != 3
            ) as can_access,
        ):
            ensure_xpu_peer_access(2, 2)
            can_access.assert_not_called()
            ensure_xpu_peer_access(1, 0)
            with self.assertRaises(XpuIpcError):
                ensure_xpu_peer_access(3, 0)

    def test_a_source_this_process_cannot_see_is_let_through_with_one_warning(self):
        with (
            patch.object(torch.xpu, "device_count", return_value=1),
            patch.object(torch.xpu, "can_device_access_peer") as can_access,
            patch.object(xpu_ipc_utils, "_PEER_VISIBILITY_WARNED", set()),
            self.assertLogs(xpu_ipc_utils.logger, level="WARNING") as logs,
        ):
            ensure_xpu_peer_access(5, 0)
            ensure_xpu_peer_access(5, 0)
        can_access.assert_not_called()
        self.assertEqual(len(logs.records), 1)


class TestWaitForXpuEvent(CustomTestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)
        patcher = patch.object(
            torch.xpu, "device", side_effect=lambda index: nullcontext()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _wait(self, event, timeout=_PEER_TIMEOUT_S):
        return asyncio.run(
            wait_for_xpu_event(
                event, device_index=0, executor=self.executor, timeout=timeout
            )
        )

    def test_a_finished_event_returns_at_once(self):
        event = SimpleNamespace(query=lambda: True, synchronize=Mock())
        self._wait(event)
        event.synchronize.assert_not_called()

    def test_a_pending_event_is_waited_for_off_the_loop(self):
        loop_thread = threading.get_ident()
        waited_on = []
        event = SimpleNamespace(
            query=lambda: False,
            synchronize=lambda: waited_on.append(threading.get_ident()),
        )
        self._wait(event)
        self.assertEqual(len(waited_on), 1)
        self.assertNotEqual(waited_on[0], loop_thread)

    def test_an_event_that_never_finishes_times_out(self):
        event = SimpleNamespace(query=lambda: False, synchronize=lambda: time.sleep(1))
        with self.assertRaises(asyncio.TimeoutError):
            self._wait(event, timeout=0.05)


if __name__ == "__main__":
    unittest.main()
