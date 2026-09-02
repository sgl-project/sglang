"""Unit tests for LimitedCapacityDict in detokenizer_manager.

Covers __setitem__ semantics:
- Updating an existing key must NOT evict any other entry.
- Inserting a new key at capacity MUST evict the oldest entry.
- capacity=0 must not raise during key insertion.
- Dict length is stable across updates (no silent shrinkage).
- Insertion order is preserved when an existing key is updated.
"""

import unittest

from sglang.srt.managers.detokenizer_manager import LimitedCapacityDict
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLimitedCapacityDictSetItem(unittest.TestCase):
    def test_new_key_at_capacity_evicts_oldest(self):
        d = LimitedCapacityDict(capacity=2)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3  # "a" is oldest and must be evicted
        self.assertNotIn("a", d)
        self.assertIn("b", d)
        self.assertIn("c", d)
        self.assertEqual(len(d), 2)

    def test_update_existing_key_does_not_evict_other_entry(self):
        d = LimitedCapacityDict(capacity=2)
        d["a"] = 1
        d["b"] = 2
        d["a"] = 99  # update, not insert — "b" must survive
        self.assertIn("a", d)
        self.assertEqual(d["a"], 99)
        self.assertIn("b", d)
        self.assertEqual(len(d), 2)

    def test_update_existing_key_does_not_grow_dict(self):
        d = LimitedCapacityDict(capacity=3)
        d["x"] = 10
        d["y"] = 20
        d["x"] = 30  # update, not insert
        self.assertEqual(len(d), 2)

    def test_capacity_zero_does_not_raise(self):
        d = LimitedCapacityDict(capacity=0)
        try:
            d["k"] = 1
        except Exception as exc:
            self.fail(f"capacity=0 raised unexpectedly: {exc}")

    def test_insertion_order_preserved_after_update(self):
        d = LimitedCapacityDict(capacity=3)
        d["a"] = 1
        d["b"] = 2
        d["c"] = 3
        d["b"] = 99  # update middle key — must not reorder
        self.assertEqual(list(d.keys()), ["a", "b", "c"])
        self.assertEqual(d["b"], 99)

    def test_sequential_eviction_order_is_fifo(self):
        d = LimitedCapacityDict(capacity=2)
        d["first"] = 1
        d["second"] = 2
        d["third"] = 3   # evicts "first"
        d["fourth"] = 4  # evicts "second"
        self.assertEqual(list(d.keys()), ["third", "fourth"])


if __name__ == "__main__":
    unittest.main()
