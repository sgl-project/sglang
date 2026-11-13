"""Retained-buffer accounting for multimodal preprocessing cache values."""

import array
import ctypes
import gc
import unittest
import weakref

import numpy as np
import torch

from sglang.srt.multimodal.cache.preprocess_cache import (
    MultimodalPreprocessCache,
    estimate_cache_size_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Artifact:
    def __init__(self, *items):
        self.items = items

    def cache_size_items(self):
        return self.items


class _OpaqueArrayOwner:
    def __init__(self, data):
        self.data = data
        self.__array_interface__ = data.__array_interface__


class TestPreprocessRetainedBuffers(unittest.TestCase):
    def test_numpy_views_count_the_retained_allocation(self):
        owner = np.arange(8192, dtype=np.float32)
        original = owner.copy()
        views = [
            owner[:1],
            owner[17:43:3],
            owner[::-4],
            owner.reshape(64, 128).T[1:3, ::2],
            owner[:0],
        ]
        for value in views:
            with self.subTest(shape=value.shape, strides=value.strides):
                self.assertEqual(estimate_cache_size_bytes(value), original.nbytes)
                self.assertTrue(np.shares_memory(value, owner) or value.size == 0)
        np.testing.assert_array_equal(owner, original)

    def test_broadcast_and_stride_trick_wrappers_reach_the_owner(self):
        owner = np.arange(32, dtype=np.int16)
        values = [
            np.broadcast_to(owner[:1], (10000,)),
            np.lib.stride_tricks.as_strided(
                owner, shape=(3, 2), strides=(8, 2), writeable=False
            ),
            np.lib.stride_tricks.sliding_window_view(owner, 3)[1:2],
        ]
        for value in values:
            with self.subTest(shape=value.shape):
                self.assertEqual(estimate_cache_size_bytes(value), owner.nbytes)
                self.assertFalse(value.flags.writeable)

    def test_memoryviews_count_owner_bytes_instead_of_elements(self):
        owner = bytearray(4096)
        values = [
            memoryview(owner)[3:7],
            memoryview(owner).cast("I")[3:4],
            memoryview(owner).cast("I", shape=(16, 64)),
            memoryview(owner)[::-3],
            memoryview(owner)[0:0],
        ]
        for value in values:
            with self.subTest(format=value.format, shape=value.shape):
                self.assertEqual(estimate_cache_size_bytes(value), len(owner))
        self.assertEqual(owner, bytearray(4096))

    def test_array_views_of_external_buffers_count_the_full_buffer(self):
        owners = [
            bytearray(4096),
            array.array("I", range(1024)),
            (ctypes.c_uint32 * 1024)(*range(1024)),
        ]
        for owner in owners:
            value = np.frombuffer(owner, dtype=np.uint8, count=4, offset=16)
            with self.subTest(owner=type(owner).__name__):
                self.assertEqual(
                    estimate_cache_size_bytes(value), memoryview(owner).nbytes
                )
                self.assertEqual(
                    estimate_cache_size_bytes(memoryview(value)),
                    memoryview(owner).nbytes,
                )

    def test_shared_numpy_and_buffer_owners_are_counted_once(self):
        owner = np.arange(4096, dtype=np.float32)
        values = _Artifact(owner[:1], owner[31:37], memoryview(owner), owner, owner[:0])
        self.assertEqual(estimate_cache_size_bytes(values), owner.nbytes)
        independent = np.arange(4096, dtype=np.float32)
        self.assertEqual(
            estimate_cache_size_bytes([values, independent[:1]]),
            owner.nbytes + independent.nbytes,
        )

    def test_small_views_cannot_bypass_cache_admission(self):
        for use_memoryview in (False, True):
            with self.subTest(use_memoryview=use_memoryview):
                owner = np.arange(16384, dtype=np.float32)
                reference = weakref.ref(owner)
                value = owner[:1]
                if use_memoryview:
                    value = memoryview(value)
                cache = MultimodalPreprocessCache(max_size_bytes=1024)
                self.assertFalse(cache.put("image", value))
                self.assertEqual(cache.stats()["size_bytes"], 0)
                self.assertEqual(len(cache), 0)
                del owner, value
                gc.collect()
                self.assertIsNone(reference())

    def test_byte_budget_eviction_releases_retained_numpy_owner(self):
        owner = np.arange(4096, dtype=np.float32)
        reference = weakref.ref(owner)
        cache = MultimodalPreprocessCache(max_size_bytes=owner.nbytes)
        self.assertTrue(cache.put("first", owner[:1]))
        del owner
        gc.collect()
        self.assertIsNotNone(reference())

        other = np.arange(4096, dtype=np.float32)
        self.assertTrue(cache.put("second", other[-1:]))
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn("first", cache)
        self.assertIn("second", cache)
        self.assertEqual(cache.stats()["size_bytes"], other.nbytes)
        self.assertEqual(cache.stats()["evictions"], 1)
        np.testing.assert_array_equal(cache.get("second"), other[-1:])

    def test_singleflight_completion_uses_retained_buffer_admission(self):
        owner = np.arange(2048, dtype=np.float32)
        cache = MultimodalPreprocessCache(max_size_bytes=128)
        (miss,) = cache.lookup_or_claim_many(["image"])
        value = owner[:2]
        cache.complete_miss(miss, value)
        self.assertIs(miss.future.result(), value)
        self.assertEqual(cache.stats()["inflight"], 0)
        self.assertEqual(cache.stats()["size_bytes"], 0)
        self.assertNotIn("image", cache)

    def test_unmeasurable_array_owner_is_not_admitted(self):
        owner = _OpaqueArrayOwner(np.arange(4096, dtype=np.float32))
        value = np.asarray(owner)[:1]
        self.assertIsNone(estimate_cache_size_bytes(value))
        cache = MultimodalPreprocessCache(max_size_bytes=1024)
        self.assertFalse(cache.put("image", value))

    def test_ordinary_values_and_explicit_size_keep_their_contract(self):
        owner = np.arange(16, dtype=np.float32)
        tensor = torch.arange(32, dtype=torch.float32)
        self.assertEqual(estimate_cache_size_bytes(owner), owner.nbytes)
        self.assertEqual(estimate_cache_size_bytes(b"abc"), 3)
        self.assertEqual(estimate_cache_size_bytes(bytearray(b"abcd")), 4)
        self.assertEqual(
            estimate_cache_size_bytes(tensor), tensor.untyped_storage().nbytes()
        )
        self.assertIsNone(estimate_cache_size_bytes(torch.empty(1, device="meta")))
        cache = MultimodalPreprocessCache(max_size_bytes=4)
        value = owner[:1]
        self.assertTrue(cache.put("explicit", value, size_bytes=4))
        self.assertIs(cache.get("explicit"), value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
