"""Unit tests for srt/disaggregation/pp_consensus_store."""

import time
import unittest
from collections.abc import Callable

from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _MockPPGroup:
    def __init__(self):
        self._obj = None

    def broadcast_object(self, obj, src=0):
        if obj is not None:
            self._obj = obj
        return self._obj


class TestPPConsensusStore(CustomTestCase):
    def _wait_until(self, condition: Callable[[], bool]) -> None:
        now = time.time()
        start = now
        deadline = now + 1
        delay_time = 0.01
        while now < deadline and not condition():
            time.sleep(delay_time)
            now = time.time()
            delay_time *= 2
        self.assertTrue(
            now < deadline, f"Condition not met after {now - start} seconds"
        )

    def test_setitem(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        store0["key"] = 10
        self.assertEqual(store0["key"], 10)
        store0.close()

    def test_setitme_replicate_to_rank0(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(2, 0, group)
        store1 = PPConsensusStore(2, 1, group)
        store0["key"] = 10
        store1["key"] = 11
        self._wait_until(lambda: store0.collect("key") == [10, 11])
        store0.close()
        store1.close()

    def test_contains(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        store0["key"] = 10
        self.assertTrue("key" in store0)
        self.assertFalse("no_such_key" in store0)
        store0.close()

    def test_get(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        store0["key"] = 10
        self.assertEqual(store0.get("key"), 10)
        store0.close()

    def test_get_not_exist(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        self.assertIsNone(store0.get("no_such_key"))
        self.assertEqual(store0.get("no_such_key", "default"), "default")
        store0.close()

    def test_get_default(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        self.assertEqual(store0.get("no_such_key", "default"), "default")
        store0.close()

    def test_getitem_not_exist(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        with self.assertRaises(KeyError):
            store0["no_such_key"]
        store0.close()

    def test_delitem(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        store0["key"] = 10
        del store0["key"]
        self.assertFalse("key" in store0)
        store0.close()

    def test_delitem_not_exist(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        with self.assertRaises(KeyError):
            del store0["no_such_key"]
        store0.close()

    def test_delitem_replicate_to_rank0(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(2, 0, group)
        store1 = PPConsensusStore(2, 1, group)
        store0["key"] = 10
        store1["key"] = 11
        self._wait_until(lambda: store0.collect("key") == [10, 11])
        del store1["key"]
        self._wait_until(lambda: store0.collect("key") == [10, None])
        store0.close()
        store1.close()

    def test_pop(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        store0["key"] = 10
        self.assertEqual(store0.pop("key"), 10)
        self.assertFalse("key" in store0)
        store0.close()

    def test_pop_not_exist(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        self.assertIsNone(store0.pop("no_such_key"))
        store0.close()

    def test_pop_default(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        self.assertEqual(store0.pop("no_such_key", "default"), "default")
        store0.close()

    def test_pop_replicate_to_rank0(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(2, 0, group)
        store1 = PPConsensusStore(2, 1, group)
        store0["key"] = 10
        store1["key"] = 11
        self._wait_until(lambda: store0.collect("key") == [10, 11])
        self.assertEqual(store1.pop("key"), 11)
        self.assertFalse("key" in store1)
        self._wait_until(lambda: store0.collect("key") == [10, None])
        store0.close()
        store1.close()

    def test_collect(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(3, 0, group)
        store1 = PPConsensusStore(3, 1, group)
        store2 = PPConsensusStore(3, 2, group)
        store0["key"] = 10
        store1["key"] = 11
        store2["key"] = 12
        self._wait_until(lambda: store0.collect("key") == [10, 11, 12])
        store0.close()
        store1.close()
        store2.close()

    def test_collect_missing(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(3, 0, group)
        store1 = PPConsensusStore(3, 1, group)
        store2 = PPConsensusStore(3, 2, group)
        store0["key"] = 10
        store2["key"] = 12
        self._wait_until(lambda: store0.collect("key") == [10, None, 12])
        store0.close()
        store1.close()
        store2.close()

    def test_collect_not_exist(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(1, 0, group)
        self.assertEqual(store0.collect("no_such_key"), [None])
        store0.close()

    def test_collect_requires_pp0(self):
        group = _MockPPGroup()
        store0 = PPConsensusStore(2, 0, group)
        store1 = PPConsensusStore(2, 1, group)
        with self.assertRaises(AssertionError):
            store1.collect("key")
        store0.close()
        store1.close()


if __name__ == "__main__":
    unittest.main()
