"""CPU contracts for the Mooncake Engram adapter; no Mooncake install required."""

import sys
import unittest
from types import ModuleType
from unittest import mock

import numpy as np
import torch

from sglang.srt.mem_cache.engram import create_storage
from sglang.srt.mem_cache.storage.mooncake_engram import MooncakeEngramStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _CpuEngramStore:
    """Replace only the external byte-table API, including its -1 sentinel."""

    def __init__(self, *, tables, namespace, local_dir, persistent):
        self.tables = tables
        self.rows = None
        self.close_count = 0
        self.lookup_count = 0

    def populate(self, table_id, rows):
        spec = self.tables[table_id]
        if rows.dtype != np.uint8 or rows.shape != (
            spec["num_rows"],
            spec["row_bytes"],
        ):
            raise ValueError("table metadata does not match the published rows")
        if not rows.flags.c_contiguous:
            raise ValueError("populate requires contiguous rows")
        self.rows = rows.copy()

    def lookup(self, table_ids, indices, output):
        if table_ids != ["rows"] or indices.shape[1:] != (1,):
            raise ValueError("expected one table column")
        if not indices.flags.c_contiguous:
            raise ValueError("lookup requires contiguous IDs")
        self.lookup_count += 1
        valid = indices[:, 0] >= 0
        output[valid, 0] = self.rows[indices[valid, 0]]

    def close(self):
        self.close_count += 1


class TestEngramStorageConfig(CustomTestCase):
    def test_local_backend_defaults_and_explicit_directory(self):
        for config, directory in (
            (None, "/dev/shm/sglang-engram"),
            ('{"backend":"local","local_dir":"/tmp/engram-test"}', "/tmp/engram-test"),
        ):
            with self.subTest(config=config):
                storage = create_storage(config)
                self.addCleanup(storage.close)
                self.assertIsInstance(storage, MooncakeEngramStorage)
                self.assertEqual(storage._local_dir, directory)

    def test_invalid_config_does_not_silently_fall_back_to_local(self):
        for config, message in (
            ("{", "JSON object"),
            ("[]", "JSON object"),
            ('{"local_dir":""}', "nonempty string"),
            ('{"local_dir":42}', "nonempty string"),
            ('{"unknown_option":true}', "Unknown.*options"),
            ('{"backend":"typo"}', "Unknown Engram storage backend"),
        ):
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, message),
            ):
                create_storage(config)
        with self.assertRaisesRegex(NotImplementedError, "Remote"):
            create_storage('{"backend":"remote"}')


class TestMooncakeEngramStorage(CustomTestCase):
    def setUp(self):
        self.stores = []
        module = ModuleType("mooncake.engram")
        module.EngramStore = self._new_store
        self.enterContext(
            mock.patch.dict(
                sys.modules,
                {"mooncake": ModuleType("mooncake"), "mooncake.engram": module},
            )
        )
        # These tests must not initialize a CUDA runtime, even on GPU hosts.
        self.enterContext(mock.patch("torch.cuda.is_initialized", return_value=False))
        self.storage = MooncakeEngramStorage(local_dir="/unused/engram-test")
        self.addCleanup(self.storage.close)

    def _new_store(self, **kwargs):
        store = _CpuEngramStore(**kwargs)
        self.stores.append(store)
        return store

    def test_publish_copies_noncontiguous_rows_and_lookup_preserves_shape(self):
        rows = torch.arange(48, dtype=torch.uint8).reshape(6, 8)[:, ::2]
        expected_rows = rows.clone()
        self.storage.initialize(rows)
        rows.zero_()  # Storage must own the data after the loader releases it.
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                ids = torch.tensor([[5, -1, 2], [0, 4, 1]], dtype=dtype).t()
                actual = self.storage.lookup(ids)
                expected = torch.zeros((*ids.shape, 4), dtype=torch.uint8)
                expected[ids >= 0] = expected_rows[ids[ids >= 0].long()]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_batch_never_calls_backend(self):
        self.storage.initialize(torch.ones((6, 4), dtype=torch.uint8))
        actual = self.storage.lookup(torch.empty((0, 3), dtype=torch.int64))
        self.assertEqual(actual.shape, (0, 3, 4))
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(self.stores[0].lookup_count, 0)

    def test_rejected_initialization_can_be_retried(self):
        for rows in (
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones(8, dtype=torch.uint8),
            torch.empty((2, 4), dtype=torch.uint8, device="meta"),
        ):
            with self.subTest(shape=rows.shape, dtype=rows.dtype, device=rows.device):
                with self.assertRaisesRegex(ValueError, "CPU byte matrix"):
                    self.storage.initialize(rows)
        self.assertEqual(self.stores, [])
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        self.storage.initialize(rows)
        torch.testing.assert_close(self.storage.lookup(torch.tensor([1])), rows[1:])

    def test_publication_failure_closes_store_and_allows_retry(self):
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        with mock.patch.object(
            _CpuEngramStore, "populate", side_effect=OSError("publish failed")
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.storage.initialize(rows)
        self.assertEqual(self.stores[0].close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))
        self.storage.initialize(rows)
        torch.testing.assert_close(self.storage.lookup(torch.tensor([0])), rows[:1])

    def test_duplicate_initialization_preserves_original_table(self):
        rows = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        self.storage.initialize(rows)
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            self.storage.initialize(torch.zeros_like(rows))
        torch.testing.assert_close(self.storage.lookup(torch.tensor([1])), rows[1:])
        self.assertEqual(len(self.stores), 1)

    def test_graph_capture_rejects_cold_publication(self):
        with (
            mock.patch("torch.cuda.is_initialized", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"),
        ):
            self.storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        self.assertEqual(self.stores, [])

    def test_lookup_lifecycle_and_invalid_id_dtype(self):
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))
        self.storage.initialize(torch.ones((2, 4), dtype=torch.uint8))
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            self.storage.lookup(torch.tensor([0.0]))
        self.assertEqual(self.stores[0].lookup_count, 0)
        self.storage.close()
        self.storage.close()
        self.assertEqual(self.stores[0].close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Initialize"):
            self.storage.lookup(torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
