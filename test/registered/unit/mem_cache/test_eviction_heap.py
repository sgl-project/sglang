import unittest

from sglang.srt.mem_cache.eviction_heap import LazyEvictionHeap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class LazyEvictionHeapTest(unittest.TestCase):
    def test_pop_refreshes_priority_and_skips_removed_nodes(self):
        active = {"a", "b", "c"}
        priorities = {"a": 1, "b": 2, "c": 3}
        heap = LazyEvictionHeap(
            priority=priorities.__getitem__,
            candidate=active.__contains__,
            active_count=lambda: len(active),
        )
        for node in active:
            heap.add(node)

        priorities["a"] = 10
        active.remove("b")

        self.assertEqual(heap.pop(), "c")
        self.assertEqual(heap.pop(), "a")
        self.assertIsNone(heap.pop())
        self.assertEqual(len(heap), 0)

    def test_refresh_invalidates_older_versions_after_priority_decrease(self):
        active = {"a", "b", "c"}
        priorities = {"a": 1, "b": 2, "c": 3}
        heap = LazyEvictionHeap(
            priority=priorities.__getitem__,
            candidate=active.__contains__,
            active_count=lambda: len(active),
        )
        for node in active:
            heap.add(node)

        priorities.update({"a": 100, "b": 1, "c": 2})
        heap.refresh("b")
        heap.refresh("c")

        self.assertEqual(heap.pop(), "b")
        self.assertEqual(heap.pop(), "c")
        self.assertEqual(heap.pop(), "a")
        self.assertIsNone(heap.pop())

    def test_readd_after_lazy_delete_keeps_one_live_entry(self):
        active = {"a", "b"}
        priorities = {"a": 1, "b": 2}
        heap = LazyEvictionHeap(
            priority=priorities.__getitem__,
            candidate=active.__contains__,
            active_count=lambda: len(active),
        )
        heap.add("a")
        active.remove("a")
        priorities["a"] = 0
        active.add("a")
        heap.add("a")
        heap.add("b")

        self.assertEqual(heap.pop(), "a")
        self.assertEqual(heap.pop(), "b")
        self.assertIsNone(heap.pop())

    def test_compaction_drops_invalid_entries_and_refreshes_priorities(self):
        active = {f"n{index}" for index in range(10)}
        priorities = {node: index for index, node in enumerate(sorted(active))}
        heap = LazyEvictionHeap(
            priority=priorities.__getitem__,
            candidate=active.__contains__,
            active_count=lambda: len(active),
            minimum_compaction_size=4,
        )
        for node in active:
            heap.add(node)

        active.intersection_update({"n5"})
        priorities["n5"] = 100

        self.assertEqual(heap.pop(), "n5")
        self.assertEqual(len(heap), 0)

    def test_clear_resets_heap_and_membership(self):
        active = {"a"}
        heap = LazyEvictionHeap(
            priority=lambda _: 1,
            candidate=active.__contains__,
            active_count=lambda: len(active),
        )
        heap.add("a")
        heap.clear()
        self.assertEqual(len(heap), 0)
        heap.add("a")
        self.assertEqual(heap.pop(), "a")


if __name__ == "__main__":
    unittest.main()
