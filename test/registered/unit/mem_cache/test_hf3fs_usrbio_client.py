"""CPU tests for the HF3FS USRBIO client with a finite, asynchronous fake ring."""

import errno
import importlib.util
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import timedelta
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None

if torch is None:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(**kwargs):
        pass

else:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ByteTensor:
    """CPU byte tensors used only for direct local runs without PyTorch."""

    itemsize = 1

    def __init__(self, buffer, start=0, stop=None, **kwargs):
        self.buffer = buffer
        self.start = start
        self.stop = len(buffer) if stop is None else stop

    def numel(self):
        return self.stop - self.start

    def __getitem__(self, span):
        start, stop, _ = span.indices(self.numel())
        return _ByteTensor(self.buffer, self.start + start, self.start + stop)

    def copy_(self, source):
        self.buffer[self.start : self.stop] = bytes(source.tolist())

    def tolist(self):
        return list(self.buffer[self.start : self.stop])


def _read_shm(shm, tensors):
    current = 0
    for tensor in tensors:
        end = current + tensor.numel()
        tensor.copy_(shm[current:end])
        current = end


def _write_shm(tensors, shm):
    current = 0
    for tensor in tensors:
        end = current + tensor.numel()
        shm[current:end].copy_(tensor)
        current = end


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_client():
    if torch is not None:
        from sglang.srt.mem_cache.storage.hf3fs import hf3fs_usrbio_client

        return hf3fs_usrbio_client

    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/hf3fs"
    )
    torch_stub = ModuleType("torch")
    torch_stub.Tensor = _ByteTensor
    torch_stub.uint8 = object()
    torch_stub.frombuffer = _ByteTensor
    loader_name = "sglang.srt.utils.cpp_extension_loader"
    loader = ModuleType(loader_name)
    loader.load_extension_with_recovery = lambda **kwargs: SimpleNamespace(
        read_shm=_read_shm, write_shm=_write_shm
    )
    with patch.dict(sys.modules, {"torch": torch_stub, loader_name: loader}):
        _load_source(
            "sglang.srt.mem_cache.storage.hf3fs.hf3fs_client",
            source / "hf3fs_client.py",
        )
        return _load_source("_hf3fs_usrbio_client", source / "hf3fs_usrbio_client.py")


def _tensor(values):
    if torch is None:
        return _ByteTensor(bytearray(values))
    return torch.tensor(values, dtype=torch.uint8)


class _IoVec:
    def __init__(self, shm: SharedMemory, span=slice(None)):
        self.shm = shm
        self.span = span
        self.result = None
        self.userdata = None

    def __getitem__(self, span):
        return _IoVec(self.shm, span)


class _IoRing:
    def __init__(self, entries, storage, for_read):
        self.entries = entries
        self.storage = storage
        self.for_read = for_read
        self.pending = []
        self.completed = []
        self.stalled_offsets = set()
        self.results = {}
        self.prepare_error_offset = None
        self.submit_error = False
        self.wait_error = False
        self.prepare_calls = 0
        self.submit_calls = 0
        self.waits = []

    def prepare(self, iov, read, fd, offset, userdata=None):
        self.prepare_calls += 1
        assert read == self.for_read
        if (
            offset == self.prepare_error_offset
            or len(self.pending) + len(self.completed) == self.entries
        ):
            raise OSError(errno.EAGAIN, "no free USRBIO slots")
        iov.userdata = userdata
        self.pending.append((offset, iov))
        return self

    def finish(self, offset, iov):
        # As in 3FS, the daemon touches SHM at completion, possibly after wait returns.
        with iov.shm.buf[iov.span] as data:
            iov.result = self.results.get(offset, len(data))
            if iov.result > 0:
                end = offset + iov.result
                if self.for_read:
                    data[: iov.result] = self.storage[offset:end]
                else:
                    self.storage[offset:end] = data[: iov.result]
        self.pending.remove((offset, iov))
        self.completed.append(iov)

    def submit(self):
        self.submit_calls += 1
        if self.submit_error:
            self.submit_error = False
            raise OSError(errno.EINTR, "submission notification failed")
        for offset, iov in self.pending[::-1]:
            if offset not in self.stalled_offsets:
                self.finish(offset, iov)
        return self

    def wait(self, *, min_results=0, max_results=0, timeout=None):
        self.waits.append(timeout)
        if self.wait_error:
            self.wait_error = False
            raise OSError(errno.EINTR, "wait interrupted before reaping")
        # A short/empty return models deadline expiration. Completed slots stay
        # occupied until this method removes their CQEs.
        count = max(min_results, max_results, 1)
        results = self.completed[:count]
        del self.completed[:count]
        return results


class TestHf3fsUsrBioClient(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client_module = _load_client()

    def setUp(self):
        super().setUp()
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.storage = bytearray(range(64))
        self.read_ring = _IoRing(3, self.storage, for_read=True)
        self.write_ring = _IoRing(3, self.storage, for_read=False)
        self.registered_fds = set()
        mocks = patch.multiple(
            self.client_module,
            create=True,
            HF3FS_AVAILABLE=True,
            register_fd=self.registered_fds.add,
            deregister_fd=self.registered_fds.remove,
            extract_mount_point=lambda path: tmp_dir.name,
            make_ioring=lambda *args, for_read, **kwargs: (
                self.read_ring if for_read else self.write_ring
            ),
            make_iovec=lambda shm, mount_point: _IoVec(shm),
        )
        mocks.start()
        self.addCleanup(mocks.stop)
        self.client = self.client_module.Hf3fsUsrBioClient(
            str(Path(tmp_dir.name) / "cache"),
            size=len(self.storage),
            bytes_per_page=8,
            entries=3,
            client_timeout=1,
        )
        self.addCleanup(self._close_client)

    def _close_client(self):
        if self.registered_fds:
            self.client.close()

    def assert_ring_drained(self, ring):
        self.assertEqual(ring.pending, [])
        self.assertEqual(ring.completed, [])

    def test_out_of_order_reads_and_writes(self):
        offsets = [0, 12, 30]
        tensors = [_tensor([255]), _tensor([255] * 3), _tensor([255] * 2)]
        self.assertEqual(self.client.batch_read(offsets, tensors), [1, 3, 2])
        self.assertEqual(
            [tensor.tolist() for tensor in tensors], [[0], [12, 13, 14], [30, 31]]
        )

        inputs = [_tensor([50]), _tensor([60, 61, 62]), _tensor([70, 71])]
        self.assertEqual(self.client.batch_write(offsets, inputs), [1, 3, 2])
        self.assertEqual(list(self.storage[0:1]), [50])
        self.assertEqual(list(self.storage[12:15]), [60, 61, 62])
        self.assertEqual(list(self.storage[30:32]), [70, 71])
        for ring in (self.read_ring, self.write_ring):
            self.assertEqual(ring.submit_calls, 1)
            self.assertEqual(ring.waits, [timedelta(seconds=1)])
            self.assert_ring_drained(ring)

    def test_short_and_failed_reads_do_not_overwrite_destinations(self):
        self.read_ring.results = {0: 1, 12: -errno.EIO}
        tensors = [_tensor([255] * 2), _tensor([255] * 3), _tensor([255] * 2)]
        self.assertEqual(
            self.client.batch_read([0, 12, 30], tensors), [1, -errno.EIO, 2]
        )
        self.assertEqual(
            [tensor.tolist() for tensor in tensors], [[255, 255], [255] * 3, [30, 31]]
        )

    def test_read_timeout_preserves_indices_and_recovers(self):
        ring = self.read_ring
        ring.stalled_offsets.add(0)
        first = [_tensor([255]), _tensor([255] * 3), _tensor([255] * 2)]
        self.assertEqual(self.client.batch_read([0, 12, 30], first), [0, 3, 2])
        self.assertEqual(
            [tensor.tolist() for tensor in first], [[255], [12, 13, 14], [30, 31]]
        )
        self.assertEqual(len(ring.pending), 1)

        later = [_tensor([255] * 2) for _ in range(3)]
        self.assertEqual(self.client.batch_read([5, 20, 40], later), [0, 0, 0])
        self.assertEqual(ring.prepare_calls, 3)
        self.assertEqual([tensor.tolist() for tensor in later], [[255, 255]] * 3)
        self.assertEqual(self.client.batch_write([50], [_tensor([123])]), [1])

        ring.stalled_offsets.clear()
        ring.finish(*ring.pending[0])
        self.assertEqual(self.client.batch_read([5, 20, 40], later), [2, 2, 2])
        self.assertEqual(
            [tensor.tolist() for tensor in later], [[5, 6], [20, 21], [40, 41]]
        )
        self.assertEqual(first[0].tolist(), [255])
        self.assert_ring_drained(ring)
        self.assertTrue(
            all(
                t is not None and timedelta(0) <= t <= timedelta(seconds=1)
                for t in ring.waits
            )
        )

    def test_write_timeout_does_not_overwrite_pending_data(self):
        ring = self.write_ring
        ring.stalled_offsets.add(0)
        first = [_tensor([100]), _tensor([101] * 3), _tensor([102] * 2)]
        self.assertEqual(self.client.batch_write([0, 12, 30], first), [0, 3, 2])
        pending = bytes(self.client.shm_w.buf)
        later = [_tensor([200] * 2) for _ in range(3)]
        self.assertEqual(self.client.batch_write([5, 20, 40], later), [0, 0, 0])
        self.assertEqual(ring.prepare_calls, 3)
        self.assertEqual(bytes(self.client.shm_w.buf), pending)

        ring.stalled_offsets.clear()
        ring.finish(*ring.pending[0])
        self.assertEqual(self.storage[0], 100)
        self.assertEqual(self.client.batch_write([5, 20, 40], later), [2, 2, 2])
        self.assertEqual(list(self.storage[5:7]), [200, 200])
        self.assert_ring_drained(ring)

    def test_prepare_error_reaps_the_accepted_prefix(self):
        for method, ring in (
            (self.client.batch_read, self.read_ring),
            (self.client.batch_write, self.write_ring),
        ):
            with self.subTest(for_read=ring.for_read):
                offsets = [0, 12, 30]
                tensors = [_tensor([255]), _tensor([255] * 3), _tensor([255] * 2)]
                ring.prepare_error_offset = 12
                self.assertEqual(method(offsets, tensors), [1, 0, 0])
                self.assertEqual(ring.submit_calls, 1)
                self.assert_ring_drained(ring)
                ring.prepare_error_offset = None
                self.assertEqual(method(offsets, tensors), [1, 3, 2])
                self.assert_ring_drained(ring)

    def test_ring_errors_preserve_pending_requests_for_recovery(self):
        for attribute in ("wait_error", "submit_error"):
            with self.subTest(error=attribute):
                ring = self.read_ring
                setattr(ring, attribute, True)
                old = [_tensor([255]) for _ in range(3)]
                self.assertEqual(self.client.batch_read([0, 12, 30], old), [0, 0, 0])
                self.assertEqual([tensor.tolist() for tensor in old], [[255]] * 3)
                new = [_tensor([255] * 2) for _ in range(3)]
                self.assertEqual(self.client.batch_read([5, 20, 40], new), [2, 2, 2])
                self.assertEqual(
                    [tensor.tolist() for tensor in new], [[5, 6], [20, 21], [40, 41]]
                )
                self.assert_ring_drained(ring)

    def test_empty_batch_does_not_access_rings(self):
        self.assertEqual(self.client.batch_read([], []), [])
        self.assertEqual(self.client.batch_write([], []), [])
        self.assertEqual(self.read_ring.prepare_calls, 0)
        self.assertEqual(self.write_ring.prepare_calls, 0)
        self.assertEqual(self.read_ring.waits, [])
        self.assertEqual(self.write_ring.waits, [])
        self.assertEqual(self.client.batch_read([0], [_tensor([255])]), [1])

    def test_invalid_batches_do_not_close_client(self):
        for offsets, tensors in (
            ([0, 64], [_tensor([255]), _tensor([255])]),
            ([0, -1], [_tensor([255]), _tensor([255])]),
            ([0, 12], [_tensor([255]), _tensor([255] * 9)]),
            ([], [_tensor([255])]),
            ([0], []),
        ):
            with self.subTest(offsets=offsets, lengths=[t.numel() for t in tensors]):
                prepare_calls = self.write_ring.prepare_calls
                with self.assertRaises(ValueError):
                    self.client.batch_write(offsets, tensors)
                self.assertEqual(len(self.registered_fds), 1)
                self.assertEqual(self.write_ring.prepare_calls, prepare_calls)
        self.assertEqual(self.client.batch_write([0], [_tensor([255])]), [1])

    def test_close_is_idempotent(self):
        self.client.close()
        self.client.close()
        self.assertFalse(self.registered_fds)

    def test_close_releases_both_buffers_after_unlink_error(self):
        shm_r, shm_w = self.client.shm_r, self.client.shm_w
        unlink = SharedMemory.unlink

        def fail_read_unlink(shm):
            if shm is shm_r:
                raise OSError("unlink failed")
            unlink(shm)

        with (
            patch.object(SharedMemory, "unlink", fail_read_unlink),
            self.assertRaisesRegex(OSError, "unlink failed"),
        ):
            self.client.close()

        self.assertIsNone(shm_r.buf)
        self.assertIsNone(shm_w.buf)
        self.assertFalse(self.registered_fds)
        self.client.close()

    def test_close_after_registration_failure(self):
        client_type = type(self.client)
        failed_client = client_type.__new__(client_type)
        path = str(Path(self.client.path).with_name("failed-cache"))
        with (
            patch.object(
                self.client_module,
                "register_fd",
                side_effect=OSError("register failed"),
            ) as register,
            self.assertRaisesRegex(OSError, "register failed"),
        ):
            failed_client.__init__(path, 64, 8, 3, 1)

        fd = register.call_args.args[0]
        try:
            failed_client.close()
            with self.assertRaises(OSError):
                os.fstat(fd)
        finally:
            with suppress(OSError):
                os.close(fd)

    def test_same_ring_serializes_batches(self):
        first_waiting = Event()
        release_wait = Event()
        second_started = Event()
        second_finished = Event()
        ring = self.read_ring
        wait = ring.wait

        def pause_first_wait(**kwargs):
            if not first_waiting.is_set():
                first_waiting.set()
                if not release_wait.wait(timeout=5):
                    raise TimeoutError("test did not resume first wait")
            return wait(**kwargs)

        def read_second_batch(tensor):
            second_started.set()
            result = self.client.batch_read([12], [tensor])
            second_finished.set()
            return result

        first, second = _tensor([255]), _tensor([255])
        with (
            patch.object(ring, "wait", pause_first_wait),
            ThreadPoolExecutor(2) as pool,
        ):
            first_result = pool.submit(self.client.batch_read, [0], [first])
            try:
                self.assertTrue(first_waiting.wait(timeout=5))
                second_result = pool.submit(read_second_batch, second)
                self.assertTrue(second_started.wait(timeout=5))
                self.assertFalse(second_finished.wait(timeout=0.05))
            finally:
                release_wait.set()
            self.assertEqual(first_result.result(timeout=5), [1])
            self.assertEqual(second_result.result(timeout=5), [1])
        self.assertEqual(first.tolist(), [0])
        self.assertEqual(second.tolist(), [12])
        self.assert_ring_drained(ring)


if __name__ == "__main__":
    unittest.main()
