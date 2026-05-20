"""Unit tests for ``MirroredCpAvailability`` (DESIGN_kv_reshard.md §7).

Pure-Python state machine — no GPU, no NCCL.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest

from sglang.srt.mem_cache.mirrored_cp_availability import MirroredCpAvailability
from sglang.test.test_utils import CustomTestCase


class TestMirroredCpAvailability(CustomTestCase):
    def test_init_mirrors_size_local_to_all_ranks(self):
        m = MirroredCpAvailability(cp_size=4, size_local=1024)
        self.assertEqual(m.cp_size, 4)
        self.assertEqual(m.size_local, 1024)
        self.assertEqual(m.local_available, [1024, 1024, 1024, 1024])

    def test_can_admit_true_when_every_rank_has_room(self):
        m = MirroredCpAvailability(cp_size=2, size_local=10)
        self.assertTrue(m.can_admit([5, 7]))
        self.assertTrue(m.can_admit([10, 10]))
        self.assertTrue(m.can_admit([0, 0]))

    def test_can_admit_false_when_any_rank_lacks_room(self):
        m = MirroredCpAvailability(cp_size=2, size_local=10)
        self.assertFalse(m.can_admit([11, 0]))
        self.assertFalse(m.can_admit([0, 11]))
        self.assertFalse(m.can_admit([10, 11]))

    def test_alloc_decrements_per_rank_independently(self):
        m = MirroredCpAvailability(cp_size=3, size_local=100)
        m.alloc([10, 0, 30])
        self.assertEqual(m.local_available, [90, 100, 70])
        m.alloc([5, 5, 5])
        self.assertEqual(m.local_available, [85, 95, 65])

    def test_free_restores_per_rank_independently(self):
        m = MirroredCpAvailability(cp_size=3, size_local=100)
        m.alloc([10, 20, 30])
        # after alloc: [90, 80, 70]; free [4, 8, 10] -> [94, 88, 80]
        m.free([4, 8, 10])
        self.assertEqual(m.local_available, [94, 88, 80])

    def test_alloc_free_round_trip_preserves_state(self):
        m = MirroredCpAvailability(cp_size=4, size_local=1000)
        ops = [
            [100, 200, 50, 25],
            [0, 0, 300, 75],
            [10, 10, 10, 10],
        ]
        for op in ops:
            m.alloc(op)
        for op in ops:
            m.free(op)
        self.assertEqual(m.local_available, [1000] * 4)

    def test_min_available_returns_most_loaded_rank_budget(self):
        m = MirroredCpAvailability(cp_size=4, size_local=100)
        m.alloc([10, 50, 5, 30])
        # remaining: [90, 50, 95, 70]
        self.assertEqual(m.min_available(), 50)

    def test_can_admit_exact_match_is_true(self):
        m = MirroredCpAvailability(cp_size=2, size_local=8)
        m.alloc([3, 5])
        # remaining: [5, 3]; ask for exactly what's left
        self.assertTrue(m.can_admit([5, 3]))

    def test_wrong_length_owned_counts_raises(self):
        m = MirroredCpAvailability(cp_size=3, size_local=10)
        with self.assertRaises(ValueError):
            m.can_admit([1, 2])
        with self.assertRaises(ValueError):
            m.alloc([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            m.free([1])

    def test_invalid_init_args_raise(self):
        with self.assertRaises(ValueError):
            MirroredCpAvailability(cp_size=0, size_local=10)
        with self.assertRaises(ValueError):
            MirroredCpAvailability(cp_size=-1, size_local=10)
        with self.assertRaises(ValueError):
            MirroredCpAvailability(cp_size=2, size_local=-1)

    def test_spmd_determinism_two_instances_match(self):
        """Two ranks running identical operations on identical inputs end
        up with identical local_available — the SPMD invariant the class
        relies on (no IPC needed)."""
        rank0 = MirroredCpAvailability(cp_size=4, size_local=500)
        rank1 = MirroredCpAvailability(cp_size=4, size_local=500)
        operations = [
            ("alloc", [10, 20, 30, 40]),
            ("alloc", [5, 0, 0, 5]),
            ("free", [7, 7, 7, 7]),
            ("alloc", [100, 50, 25, 0]),
            ("free", [10, 10, 10, 10]),
        ]
        for op, counts in operations:
            getattr(rank0, op)(counts)
            getattr(rank1, op)(counts)
        self.assertEqual(rank0.local_available, rank1.local_available)


if __name__ == "__main__":
    unittest.main()
