"""The file-backed Engram table store: a regular file stands in for a dev-dax
device, so this runs on any Linux host with tmpfs and needs no CXL hardware.

The CUDA pin is stubbed out: what is under test is the placement (disjoint,
aligned regions over one shared file) and the fact that a table's bytes reach the
mapped file at its own offset.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.layers import engram as engram_module
from sglang.srt.layers.engram import _HostTable
from sglang.srt.layers.engram_region import (
    STORE_ALIGN,
    engram_region_offsets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DIM, BLOCK = 256, 32
ROW_BYTES = DIM + DIM // BLOCK


class _Group:
    """The slice of the TP group ``_HostTable`` uses, for one rank."""

    rank_in_group = 0

    def barrier(self):
        pass


class _NoPin:
    """``cudaHostRegister`` succeeds and the device pointer is the host one."""

    def __enter__(self):
        self._patches = [
            mock.patch.object(
                engram_module.torch.cuda,
                "cudart",
                lambda: mock.Mock(cudaHostRegister=lambda *a: 0),
            ),
            mock.patch.object(engram_module, "_registered_device_ptr", lambda p: p),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


class TestFileBackedStore(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "engram_store")
        self.rows = [4_096, 8_192]
        self.regions = engram_region_offsets(self.rows, DIM, BLOCK)

    def tearDown(self):
        self._dir.cleanup()

    def _tables(self):
        return [
            _HostTable(
                "none",
                rows * ROW_BYTES,
                f"table{i}",
                _Group(),
                store_path=self.path,
                store_offset=offset,
            )
            for i, ((offset, _), rows) in enumerate(zip(self.regions, self.rows))
        ]

    def test_tables_map_disjoint_regions_of_one_file(self):
        with _NoPin():
            tables = self._tables()
            for table, (offset, _) in zip(tables, self.regions):
                self.assertEqual(table.store_path, self.path)
                self.assertEqual(table.store_offset, offset)
            # A write to one table must not be visible through the next one.
            tables[0].bytes[:8] = torch.tensor(
                [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.uint8
            )
            head = torch.frombuffer(memoryview(tables[1].mm), dtype=torch.uint8)[:8]
            self.assertEqual(head.tolist(), [0] * 8)

    def test_a_table_reaches_the_file_at_its_own_offset(self):
        with _NoPin():
            table = self._tables()[0]
            table.bytes[:8] = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2], dtype=torch.uint8)
            offset = self.regions[0][0]
            with open(self.path, "rb") as f:
                f.seek(offset)
                self.assertEqual(list(f.read(8)), [9, 8, 7, 6, 5, 4, 3, 2])

    def test_the_file_grows_to_the_last_table(self):
        with _NoPin():
            self._tables()
        want = self.regions[-1][0] + self.rows[-1] * ROW_BYTES
        self.assertEqual(os.stat(self.path).st_size, want)

    def test_offsets_are_page_aligned(self):
        for offset, length in self.regions:
            self.assertEqual(offset % STORE_ALIGN, 0)
            self.assertEqual(length % STORE_ALIGN, 0)

    def test_a_missing_store_is_created(self):
        # A regular file must not have to be prepared by hand: the store is created
        # and grown on first use, which is what lets a fresh tmpfs path be named
        # directly in --engram-config.
        missing = os.path.join(self._dir.name, "not-yet-there")
        self.assertFalse(os.path.exists(missing))
        with _NoPin():
            table = _HostTable(
                "none",
                4_096,
                "fresh",
                _Group(),
                store_path=missing,
                store_offset=0,
            )
            self.assertEqual(table.store_offset, 0)
        self.assertTrue(os.path.exists(missing))
        self.assertGreaterEqual(os.path.getsize(missing), 4_096)

    def test_a_misaligned_offset_is_rejected(self):
        with _NoPin(), self.assertRaisesRegex(ValueError, "not page aligned"):
            _HostTable(
                "none",
                4_096,
                "bad",
                _Group(),
                store_path=self.path,
                store_offset=1,
            )

    def test_finish_load_reports_the_store(self):
        with _NoPin():
            table = self._tables()[0]
            table.dirty = True
            with self.assertLogs(engram_module.logger, level="INFO") as logs:
                table.finish_load("layer-test")
            joined = "\n".join(logs.output)
            self.assertIn(self.path, joined)
            self.assertIn("offset=0", joined)


if __name__ == "__main__":
    unittest.main()
