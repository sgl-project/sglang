"""Unit tests for EmbeddingStore, FileEmbeddingStore, and EmbeddingStoreFactory."""

import ctypes
import os
import shutil
import tempfile
import unittest

import torch

from sglang.srt.mem_cache.embedding_store import (
    EmbeddingStore,
    EmbeddingStoreFactory,
    FileEmbeddingStore,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFileEmbeddingStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = FileEmbeddingStore(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_buffer(self, data: bytes) -> torch.Tensor:
        buf = torch.empty(len(data), dtype=torch.uint8)
        ctypes.memmove(buf.data_ptr(), data, len(data))
        return buf

    def test_round_trip(self):
        data = b"hello world embedding data" * 100
        buf = self._make_buffer(data)
        size = len(data)

        results = self.store.batch_put(["hash1"], [buf.data_ptr()], [size])
        self.assertEqual(results, [True])

        read_buf = torch.empty(size, dtype=torch.uint8)
        results = self.store.batch_get(["hash1"], [read_buf.data_ptr()], [size])
        self.assertEqual(results, [True])

        original = (ctypes.c_char * size).from_address(buf.data_ptr())
        retrieved = (ctypes.c_char * size).from_address(read_buf.data_ptr())
        self.assertEqual(bytes(original), bytes(retrieved))

    def test_batch_operations(self):
        hashes = ["a", "b", "c"]
        bufs = []
        ptrs = []
        sizes = []
        for i in range(len(hashes)):
            data = bytes([i] * 256)
            buf = self._make_buffer(data)
            bufs.append(buf)
            ptrs.append(buf.data_ptr())
            sizes.append(256)

        results = self.store.batch_put(hashes, ptrs, sizes)
        self.assertEqual(results, [True, True, True])

        read_bufs = [torch.empty(256, dtype=torch.uint8) for _ in hashes]
        results = self.store.batch_get(hashes, [b.data_ptr() for b in read_bufs], sizes)
        self.assertEqual(results, [True, True, True])

        for i, rb in enumerate(read_bufs):
            retrieved = (ctypes.c_char * 256).from_address(rb.data_ptr())
            self.assertEqual(bytes(retrieved), bytes([i] * 256))

    def test_multi_buffer_round_trip(self):
        chunk1 = b"AAAA" * 64
        chunk2 = b"BBBB" * 64
        buf1 = self._make_buffer(chunk1)
        buf2 = self._make_buffer(chunk2)

        results = self.store.batch_put_from_multi_buffers(
            ["multi"],
            [[buf1.data_ptr(), buf2.data_ptr()]],
            [[len(chunk1), len(chunk2)]],
        )
        self.assertEqual(results, [True])

        read1 = torch.empty(len(chunk1), dtype=torch.uint8)
        read2 = torch.empty(len(chunk2), dtype=torch.uint8)
        results = self.store.batch_get_into_multi_buffers(
            ["multi"],
            [[read1.data_ptr(), read2.data_ptr()]],
            [[len(chunk1), len(chunk2)]],
        )
        self.assertEqual(results, [True])

        got1 = (ctypes.c_char * len(chunk1)).from_address(read1.data_ptr())
        got2 = (ctypes.c_char * len(chunk2)).from_address(read2.data_ptr())
        self.assertEqual(bytes(got1), chunk1)
        self.assertEqual(bytes(got2), chunk2)

    def test_batch_is_exist(self):
        data = b"test data"
        buf = self._make_buffer(data)
        self.store.batch_put(["exists"], [buf.data_ptr()], [len(data)])

        results = self.store.batch_is_exist(["exists", "missing"])
        self.assertEqual(results, [True, False])

    def test_batch_is_exist_ignores_directory(self):
        os.makedirs(self.store._path_for_key(self.store.get_key("dirkey")))

        results = self.store.batch_is_exist(["dirkey"])

        self.assertEqual(results, [False])

    def test_get_into_multi_buffers_returns_false_for_bad_path(self):
        os.makedirs(self.store._path_for_key(self.store.get_key("badpath")))
        read_buf = torch.empty(8, dtype=torch.uint8)

        results = self.store.batch_get_into_multi_buffers(
            ["badpath"], [[read_buf.data_ptr()]], [[8]]
        )

        self.assertEqual(results, [False])

    def test_max_size_skips_put(self):
        old_value = os.environ.get("SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE")
        os.environ["SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE"] = "4"
        try:
            store = FileEmbeddingStore(storage_dir=self.test_dir)
            data = b"too large"
            buf = self._make_buffer(data)

            results = store.batch_put(["large"], [buf.data_ptr()], [len(data)])

            self.assertEqual(results, [False])
            self.assertFalse(
                os.path.exists(store._path_for_key(store.get_key("large")))
            )
        finally:
            if old_value is None:
                os.environ.pop("SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE", None)
            else:
                os.environ["SGLANG_MM_EMBEDDING_CACHE_MAX_SIZE"] = old_value

    def test_clear_on_start_removes_embedding_cache_files(self):
        keep_path = os.path.join(self.test_dir, "keep.txt")
        cache_path = self.store._path_for_key(self.store.get_key("old"))
        tmp_path = cache_path + ".tmp.123"
        with open(keep_path, "wb") as f:
            f.write(b"keep")
        with open(cache_path, "wb") as f:
            f.write(b"old")
        with open(tmp_path, "wb") as f:
            f.write(b"tmp")

        old_value = os.environ.get("SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START")
        os.environ["SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START"] = "1"
        try:
            FileEmbeddingStore(storage_dir=self.test_dir)

            self.assertTrue(os.path.exists(keep_path))
            self.assertFalse(os.path.exists(cache_path))
            self.assertFalse(os.path.exists(tmp_path))
        finally:
            if old_value is None:
                os.environ.pop("SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START", None)
            else:
                os.environ["SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START"] = old_value

    def test_clear_on_start_can_be_disabled(self):
        cache_path = self.store._path_for_key(self.store.get_key("persist"))
        with open(cache_path, "wb") as f:
            f.write(b"persist")

        old_value = os.environ.get("SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START")
        os.environ["SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START"] = "0"
        try:
            FileEmbeddingStore(storage_dir=self.test_dir)

            self.assertTrue(os.path.exists(cache_path))
        finally:
            if old_value is None:
                os.environ.pop("SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START", None)
            else:
                os.environ["SGLANG_MM_EMBEDDING_CACHE_CLEAR_ON_START"] = old_value

    def test_dedup_on_put(self):
        data1 = b"original"
        buf1 = self._make_buffer(data1)
        self.store.batch_put(["dup"], [buf1.data_ptr()], [len(data1)])

        data2 = b"replaced"
        buf2 = self._make_buffer(data2)
        results = self.store.batch_put(["dup"], [buf2.data_ptr()], [len(data2)])
        self.assertEqual(results, [True])

        read_buf = torch.empty(len(data1), dtype=torch.uint8)
        self.store.batch_get(["dup"], [read_buf.data_ptr()], [len(data1)])
        retrieved = (ctypes.c_char * len(data1)).from_address(read_buf.data_ptr())
        self.assertEqual(bytes(retrieved), data1)

    def test_get_missing_key(self):
        read_buf = torch.empty(10, dtype=torch.uint8)
        results = self.store.batch_get(["missing"], [read_buf.data_ptr()], [10])
        self.assertEqual(results, [False])

    def test_register_buffer_is_noop(self):
        tensor = torch.empty(100, dtype=torch.uint8)
        self.store.register_buffer(tensor)

    def test_get_key(self):
        self.assertEqual(self.store.get_key("abc"), "emb_abc")

    def test_isinstance_embedding_store(self):
        self.assertIsInstance(self.store, EmbeddingStore)


class TestEmbeddingStoreFactory(unittest.TestCase):
    def test_create_file_backend(self):
        test_dir = tempfile.mkdtemp()
        try:
            store = EmbeddingStoreFactory.create_backend("file", storage_dir=test_dir)
            self.assertIsInstance(store, FileEmbeddingStore)
            self.assertEqual(store.storage_dir, test_dir)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingStoreFactory.create_backend("unknown_backend")

    def test_mooncake_registered(self):
        self.assertIn("mooncake", EmbeddingStoreFactory._registry)

    def test_file_registered(self):
        self.assertIn("file", EmbeddingStoreFactory._registry)


if __name__ == "__main__":
    unittest.main()
