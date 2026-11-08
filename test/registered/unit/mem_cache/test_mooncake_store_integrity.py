"""Regression tests for Mooncake zero-copy receipt validation.

A short zero-copy read returns the stored object's byte count as a positive
value.  The store must reject it when the destination is larger, and must
never propagate a short/long result vector into the per-pool hit results.
"""

import unittest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store import (
    MooncakeEmbeddingStore,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeStore:
    def __init__(self, get_results=None, put_results=None, exist_results=None):
        self.get_results = get_results
        self.put_results = put_results
        self.exist_results = exist_results
        self.get_calls = 0
        self.put_calls = 0

    def batch_get_into(self, keys, ptrs, sizes):
        self.get_calls += 1
        return list(self.get_results)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        self.get_calls += 1
        return list(self.get_results)

    def batch_put_from(self, keys, ptrs, sizes):
        self.put_calls += 1
        return list(self.put_results)

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config=None):
        self.put_calls += 1
        return list(self.put_results)

    def batch_is_exist(self, keys):
        return list(self.exist_results)


def _store(fake):
    obj = object.__new__(MooncakeStore)
    obj.store = fake
    obj._can_use_group_semantics = lambda: False
    return obj


class TestMooncakeStoreIntegrity(unittest.TestCase):
    def test_short_read_is_not_a_hit(self):
        checked = MooncakeStore._checked_io_results([16], [32], is_set=False)
        self.assertEqual(checked, [-1])

    def test_exact_read_bytes_pass(self):
        checked = MooncakeStore._checked_io_results([32], [32], is_set=False)
        self.assertEqual(checked, [32])

    def test_multi_buffer_bytes_are_summed(self):
        checked = MooncakeStore._checked_io_results([30], [[10, 20]], is_set=False)
        self.assertEqual(checked, [30])
        self.assertEqual(
            MooncakeStore._checked_io_results([31], [[10, 20]], is_set=False),
            [-1],
        )

    def test_result_count_mismatch_fails_closed(self):
        self.assertEqual(
            MooncakeStore._checked_io_results([0], [16, 16], is_set=True),
            [-1, -1],
        )
        self.assertEqual(
            MooncakeStore._checked_io_results([16], [16, 16], is_set=False),
            [-1, -1],
        )

    def test_zero_or_null_buffers_are_rejected(self):
        self.assertFalse(MooncakeStore._valid_object_buffers([0], [16]))
        self.assertFalse(MooncakeStore._valid_object_buffers([123], [0]))
        self.assertTrue(MooncakeStore._valid_object_buffers([123], [16]))
        self.assertTrue(MooncakeStore._valid_object_buffers([1, 2], [8, 8]))

    def test_short_native_result_vector_marks_all_rows(self):
        fake = _FakeStore(get_results=[16, 15])
        store = _store(fake)
        got = store._get_batch_zero_copy_impl(["a", "b"], [100, 200], [16, 16])
        self.assertEqual(got, [16, -1])
        self.assertEqual(fake.get_calls, 1)

    def test_null_pointer_never_reaches_native_store(self):
        fake = _FakeStore(get_results=[16])
        store = _store(fake)
        got = store._get_batch_zero_copy_impl(["a"], [0], [16])
        self.assertEqual(got, [-1])
        self.assertEqual(fake.get_calls, 0)

    def test_short_put_result_vector_fails_closed(self):
        fake = _FakeStore(put_results=[0])
        store = _store(fake)
        got = store._put_batch_zero_copy_impl(["a", "b"], [100, 200], [16, 16])
        self.assertEqual(got, [-1, -1])
        self.assertEqual(fake.put_calls, 1)

    def test_short_exist_vector_fails_closed(self):
        fake = _FakeStore(exist_results=[1])
        store = _store(fake)
        self.assertEqual(store._batch_exist(["a", "b"]), [-1, -1])

    def test_batch_set_does_not_acknowledge_failed_puts(self):
        # A short existence reply becomes [-1, -1]. Before the fix the public
        # batch_set only stopped on 0, so two failed puts still returned True.
        fake = _FakeStore(exist_results=[0], put_results=[16, 15])
        store = _store(fake)
        store.enable_storage_metrics = False
        got = store.batch_set(
            ["a", "b"], target_locations=[100, 200], target_sizes=[16, 16]
        )
        self.assertFalse(got)
        self.assertEqual(fake.put_calls, 1)


class TestMooncakeEmbeddingStoreIntegrity(unittest.TestCase):
    def test_embedding_reads_require_exact_bytes_and_receipt_count(self):
        for method, ptrs, sizes in (
            ("batch_get", [100], [16]),
            ("batch_get_into_multi_buffers", [[100, 200]], [[8, 8]]),
        ):
            for receipts, expected in (
                ([16], [True]),
                ([15], [False]),
                ([17], [False]),
                ([], [False]),
                ([16, 16], [False]),
            ):
                with self.subTest(method=method, receipts=receipts):
                    store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
                    store.store = _FakeStore(get_results=receipts)
                    self.assertEqual(
                        getattr(store, method)(["a"], ptrs, sizes), expected
                    )

    def test_embedding_puts_do_not_acknowledge_missing_receipts(self):
        for method, ptrs, sizes in (
            ("batch_put", [100, 200], [16, 16]),
            ("batch_put_from_multi_buffers", [[100], [200]], [[16], [16]]),
        ):
            for receipts in ([], [0, 0], [16]):
                with self.subTest(method=method, receipts=receipts):
                    store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
                    store.store = _FakeStore(put_results=receipts, exist_results=[1, 0])
                    self.assertEqual(
                        getattr(store, method)(["a", "b"], ptrs, sizes),
                        [True, False],
                    )
                    self.assertEqual(store.store.put_calls, 1)

    def test_embedding_short_existence_receipts_fail_closed(self):
        store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
        store.store = _FakeStore(put_results=[0, 0], exist_results=[1])
        self.assertEqual(store.batch_is_exist(["a", "b"]), [False, False])
        self.assertEqual(
            store.batch_put(["a", "b"], [100, 200], [16, 16]), [False, False]
        )
        self.assertEqual(store.store.put_calls, 0)

    def test_embedding_invalid_buffers_do_not_reach_sdk(self):
        store = MooncakeEmbeddingStore.__new__(MooncakeEmbeddingStore)
        store.store = _FakeStore(get_results=[16])
        with self.assertRaisesRegex(ValueError, "counts differ"):
            store.batch_get(["a", "b"], [100], [16])
        self.assertEqual(store.batch_get(["a"], [0], [16]), [False])
        self.assertEqual(store.store.get_calls, 0)


if __name__ == "__main__":
    unittest.main()
