"""
Unit tests for HiCacheHF3FS._batch_get partial-hit alignment (3FS backend).

``_batch_get`` builds ``file_offsets`` only from keys whose page index is not
``None`` (partial L3 hits). The target tensors passed to ``batch_read`` must be
filtered with the same ``batch_indices``; otherwise the read result of one key
is written into another key's tensor.

These tests drive the real ``HiCacheHF3FS`` through the mock file client and
the in-memory local metadata client: each hit key is written with a distinct
payload via ``_batch_set``, then ``_batch_get`` runs over a superset of keys so
``page_indices`` contains ``None`` at controlled positions. The per-key payload
assertion fails if any read lands in the wrong tensor.

Pure CPU tests: no server, no CUDA, no 3FS deployment.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_hf3fs_batch_get.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import shutil
import tempfile

import torch

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.test_utils import CustomTestCase


class HiCacheHF3FSBatchGetAlignmentTest(CustomTestCase):
    """_batch_get must align file_results with batch_indices on partial hits."""

    PAGE_BYTES = 16
    NUM_PAGES = 128
    MISS_FILL = 0xFF

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="hf3fs_batch_get_unit_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _make_storage(self, entries):
        storage = HiCacheHF3FS(
            rank=0,
            file_path=os.path.join(self.tmp_dir, f"hf3fs-entries{entries}.bin"),
            file_size=self.PAGE_BYTES * self.NUM_PAGES,
            numjobs=1,
            bytes_per_page=self.PAGE_BYTES,
            entries=entries,
            client_timeout=5,
            dtype=torch.uint8,
            metadata_client=Hf3fsLocalMetadataClient(),
            use_mock_client=True,
        )
        self.addCleanup(storage.close)
        return storage

    def _run_partial_hit(self, hit_positions, num_keys, entries):
        """Write a distinct payload per hit key, then batch-get all keys.

        Returns (results, targets, payloads) for the caller to assert on.
        """
        storage = self._make_storage(entries=entries)
        keys = [f"key-{i}" for i in range(num_keys)]
        hit_keys = [keys[i] for i in hit_positions]
        payloads = [
            torch.full((self.PAGE_BYTES,), 0xA0 + pos, dtype=torch.uint8)
            for pos in range(len(hit_positions))
        ]
        self.assertTrue(all(storage._batch_set(hit_keys, payloads)))

        targets = [
            torch.full((self.PAGE_BYTES,), self.MISS_FILL, dtype=torch.uint8)
            for _ in range(num_keys)
        ]
        results = storage._batch_get(keys, targets)
        return results, targets, payloads

    def _assert_alignment(self, hit_positions, num_keys, results, targets, payloads):
        expected = [i in hit_positions for i in range(num_keys)]
        self.assertEqual(results, expected)
        for pos, i in enumerate(hit_positions):
            self.assertTrue(
                torch.equal(targets[i], payloads[pos]),
                f"key {i} does not contain its own payload (misaligned read)",
            )
        for i in range(num_keys):
            if i not in hit_positions:
                self.assertTrue(
                    torch.equal(
                        targets[i], torch.full_like(targets[i], self.MISS_FILL)
                    ),
                    f"miss key {i} was written into",
                )

    def test_partial_hit_scattered(self):
        # Misses before/after/between hits; ragged last batch (5 hits, entries=2).
        hits = (0, 2, 4, 5, 7)
        results, targets, payloads = self._run_partial_hit(hits, num_keys=8, entries=2)
        self._assert_alignment(hits, 8, results, targets, payloads)

    def test_partial_hit_leading_miss(self):
        # A miss at position 0 shifts every subsequent offset/tensor pairing.
        hits = (1, 2, 4)
        results, targets, payloads = self._run_partial_hit(hits, num_keys=6, entries=2)
        self._assert_alignment(hits, 6, results, targets, payloads)

    def test_partial_hit_consecutive_misses(self):
        hits = (0, 3, 4)
        results, targets, payloads = self._run_partial_hit(hits, num_keys=6, entries=2)
        self._assert_alignment(hits, 6, results, targets, payloads)

    def test_all_hits(self):
        # No None entries: file_results equals the full values list.
        hits = tuple(range(6))
        results, targets, payloads = self._run_partial_hit(hits, num_keys=6, entries=2)
        self._assert_alignment(hits, 6, results, targets, payloads)

    def test_all_misses(self):
        hits = ()
        results, targets, payloads = self._run_partial_hit(hits, num_keys=4, entries=2)
        self._assert_alignment(hits, 4, results, targets, payloads)

    def test_single_entry_batches(self):
        # One page per batch: misalignment shows up as off-by-one pairings.
        hits = (1, 3, 5)
        results, targets, payloads = self._run_partial_hit(hits, num_keys=6, entries=1)
        self._assert_alignment(hits, 6, results, targets, payloads)

    def test_entries_larger_than_hits(self):
        # Single batch shorter than entries (ragged slice).
        hits = (2, 5)
        results, targets, payloads = self._run_partial_hit(hits, num_keys=7, entries=16)
        self._assert_alignment(hits, 7, results, targets, payloads)


if __name__ == "__main__":
    import unittest

    unittest.main()
