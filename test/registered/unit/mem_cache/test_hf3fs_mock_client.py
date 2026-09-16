"""CPU regressions for concurrent operations on one mock 3FS client."""

import concurrent.futures
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.mem_cache.storage.hf3fs import hf3fs_client
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHf3fsMockClient(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.page_size = 64
        self.client = hf3fs_client.Hf3fsMockClient(
            str(Path(directory.name) / "pages"), 3 * self.page_size, self.page_size, 2
        )
        self.addCleanup(self.client.close)
        self.pages = [torch.full((4, 4), value) for value in (17.0, 34.0)]

    def _overlap_operations(self, operation, tensors):
        """Force seek(0), seek(page), IO(0), IO(page) on the legacy path.

        Positional IO uses the same operation ordering, but each call retains
        its own offset. The assertions compare real file/tensor contents.
        """
        first_started = threading.Event()
        second_started = threading.Event()
        first_finished = threading.Event()
        thread_state = threading.local()

        def begin(offset, seek=False):
            thread_state.offset = offset
            if offset == 0:
                if seek:
                    os.lseek(self.client.file, offset, os.SEEK_SET)
                first_started.set()
                self.assertTrue(second_started.wait(5))
            else:
                self.assertTrue(first_started.wait(5))
                if seek:
                    os.lseek(self.client.file, offset, os.SEEK_SET)
                second_started.set()
                self.assertTrue(first_finished.wait(5))

        def finish():
            if thread_state.offset == 0:
                first_finished.set()

        def seek(fd, offset, whence):
            begin(offset, seek=True)
            return offset

        def sequential_io(fd, data):
            try:
                return getattr(os, operation)(fd, data)
            finally:
                finish()

        def positional_io(fd, data, offset):
            begin(offset)
            try:
                return getattr(os, "p" + operation)(fd, data, offset)
            finally:
                finish()

        with mock.patch.object(hf3fs_client, "os", wraps=os) as file_io:
            file_io.lseek.side_effect = seek
            getattr(file_io, operation).side_effect = sequential_io
            getattr(file_io, "p" + operation).side_effect = positional_io
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        getattr(self.client, "batch_" + operation),
                        [index * self.page_size],
                        [tensor],
                    )
                    for index, tensor in enumerate(tensors)
                ]
                results = [future.result(timeout=10) for future in futures]
        self.assertEqual(results, [[self.page_size], [self.page_size]])
        self.assertTrue(first_finished.is_set())

    def test_concurrent_reads_keep_each_requested_page(self):
        for index, page in enumerate(self.pages):
            os.pwrite(self.client.file, page.numpy().tobytes(), index * self.page_size)
        targets = [torch.empty_like(page) for page in self.pages]
        self._overlap_operations("read", targets)
        for target, expected in zip(targets, self.pages):
            torch.testing.assert_close(target, expected)

    def test_concurrent_writes_keep_each_requested_page(self):
        self._overlap_operations("write", self.pages)
        for index, page in enumerate(self.pages):
            self.assertEqual(
                os.pread(self.client.file, self.page_size, index * self.page_size),
                page.numpy().tobytes(),
            )

    def test_short_read_reports_length_without_mutating_target(self):
        os.ftruncate(self.client.file, 7)
        target = torch.full((self.page_size,), 9, dtype=torch.uint8)
        self.assertEqual(self.client.batch_read([0], [target]), [7])
        self.assertTrue(torch.all(target == 9).item())

    def test_short_write_reports_written_length(self):
        def short_write(fd, data, offset):
            return os.pwrite(fd, data[:7], offset)

        with mock.patch.object(hf3fs_client, "os", wraps=os) as file_io:
            file_io.pwrite.side_effect = short_write
            self.assertEqual(self.client.batch_write([0], [self.pages[0]]), [7])
        self.assertEqual(
            os.pread(self.client.file, self.page_size, 0),
            self.pages[0].numpy().tobytes()[:7] + bytes(self.page_size - 7),
        )

    def test_io_errors_report_zero(self):
        self.client.close()
        self.assertEqual(self.client.batch_read([0], [self.pages[0]]), [0])
        self.assertEqual(self.client.batch_write([0], [self.pages[0]]), [0])


if __name__ == "__main__":
    unittest.main()
