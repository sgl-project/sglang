import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = next(
    parent
    for parent in _THIS_FILE.parents
    if (parent / "benchmark/hicache/hicache_policy_simulator.py").exists()
)
_SIM_PATH = _REPO_ROOT / "benchmark/hicache/hicache_policy_simulator.py"
_SPEC = importlib.util.spec_from_file_location("hicache_policy_simulator", _SIM_PATH)
sim_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = sim_mod
_SPEC.loader.exec_module(sim_mod)

HiCachePolicySimulator = sim_mod.HiCachePolicySimulator
TraceRecord = sim_mod.TraceRecord
load_trace_records = sim_mod.load_trace_records
simulate_policies = sim_mod.simulate_policies


class TestHiCachePolicySimulator(unittest.TestCase):
    def test_load_trace_records_and_sort_during_simulation(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.write(json.dumps({"timestamp": 2, "hash_ids": [2]}) + "\n")
            f.write(json.dumps({"timestamp": 1, "hash_ids": [1]}) + "\n")
            f.flush()

            records = load_trace_records(f.name)

        self.assertEqual([r.timestamp for r in records], [2.0, 1.0])
        metrics = simulate_policies(
            records,
            policies=["main_write_through"],
            l1_pages=10,
            l2_pages=10,
        )["main_write_through"]
        self.assertEqual(metrics.requests, 2)
        self.assertEqual(metrics.total_input_pages, 2)

    def test_repeated_prefix_hits_l1(self):
        records = [
            TraceRecord(timestamp=1, hash_ids=(1, 2, 3)),
            TraceRecord(timestamp=2, hash_ids=(1, 2, 3)),
        ]
        metrics = simulate_policies(
            records,
            policies=["main_write_through"],
            l1_pages=10,
            l2_pages=10,
        )["main_write_through"]

        self.assertEqual(metrics.total_input_pages, 6)
        self.assertEqual(metrics.l1_hit_pages, 3)
        self.assertEqual(metrics.l2_hit_pages, 0)
        self.assertEqual(metrics.miss_pages, 3)

    def test_main_write_through_small_l2_keeps_duplicate_and_fails_new_backup(self):
        sim = HiCachePolicySimulator(
            policy="main_write_through",
            l1_pages=10,
            l2_pages=1,
        )

        sim.process_request((1,), timestamp=1)
        sim.process_request((1, 2), timestamp=2)

        sim.sanity_check()
        self.assertEqual(sim.metrics.failed_h_allocations, 1)
        self.assertEqual(sim.d_pages, 2)
        self.assertEqual(sim.h_pages, 1)
        node_a = sim.root.children[1]
        node_b = node_a.children[2]
        self.assertTrue(node_a.has_d and node_a.has_h)
        self.assertTrue(node_b.has_d)
        self.assertFalse(node_b.has_h)

    def test_l1_heap_ignores_stale_last_access_entries(self):
        sim = HiCachePolicySimulator(
            policy="main_write_through",
            l1_pages=2,
            l2_pages=0,
        )

        sim.process_request((1,), timestamp=1)
        sim.process_request((2,), timestamp=2)
        sim.process_request((1,), timestamp=3)
        sim.process_request((3,), timestamp=4)

        self.assertIn(1, sim.root.children)
        self.assertNotIn(2, sim.root.children)
        self.assertIn(3, sim.root.children)
        sim.sanity_check()

    def test_boundary_sanity_rejects_d_only_parent_to_h_only_child(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=10,
            l2_pages=10,
        )
        node_a, node_b = sim._get_or_create_path((1, 2))
        node_a.has_d = True
        node_b.has_h = True
        sim.d_pages = 1
        sim.h_pages = 1

        with self.assertRaisesRegex(AssertionError, "boundary invariant violation"):
            sim.sanity_check()

    def test_boundary_l1_evict_d_leaf_creates_parent_boundary(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=2,
            l2_pages=2,
        )

        sim.process_request((1, 2, 3), timestamp=1)

        node_a = sim.root.children[1]
        node_b = node_a.children[2]
        node_c = node_b.children[3]
        self.assertTrue(node_a.has_d)
        self.assertFalse(node_a.has_h)
        self.assertTrue(node_b.has_d and node_b.has_h)
        self.assertFalse(node_c.has_d)
        self.assertTrue(node_c.has_h)
        sim.sanity_check()

    def test_boundary_writes_request_path_to_l1_and_l2(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=10,
            l2_pages=10,
        )

        sim.process_request((1, 2, 3), timestamp=1)

        node_a = sim.root.children[1]
        node_b = node_a.children[2]
        node_c = node_b.children[3]
        self.assertTrue(node_a.has_d and node_a.has_h)
        self.assertTrue(node_b.has_d and node_b.has_h)
        self.assertTrue(node_c.has_d and node_c.has_h)
        sim.sanity_check()

    def test_boundary_l1_evict_deletes_d_leaf_when_boundary_cannot_fit(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=2,
            l2_pages=1,
        )

        sim.process_request((1, 2, 3), timestamp=1)

        node_a = sim.root.children[1]
        node_b = node_a.children[2]
        self.assertTrue(node_a.has_d)
        self.assertTrue(node_b.has_d)
        self.assertNotIn(3, node_b.children)
        self.assertEqual(sim.d_pages, 2)
        self.assertEqual(sim.h_pages, 0)
        self.assertEqual(sim.metrics.failed_h_allocations, 2)
        sim.sanity_check()

    def test_boundary_l2_evicts_duplicate_leaf_before_h_only_leaf(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=10,
            l2_pages=1,
        )
        node_a = sim._get_or_create_path((1,))[0]
        node_b = sim._get_or_create_path((2,))[0]
        sim._ensure_d(node_a)
        self.assertTrue(sim._try_ensure_h(node_a))
        sim._ensure_d(node_b)

        self.assertTrue(sim._try_ensure_h(node_b))

        self.assertTrue(node_a.has_d)
        self.assertFalse(node_a.has_h)
        self.assertTrue(node_b.has_d and node_b.has_h)
        self.assertEqual(sim.metrics.dh_to_d_l2_dedup_evictions, 1)
        sim.sanity_check()

    def test_l2_heap_keeps_protected_candidate_for_later(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=10,
            l2_pages=10,
        )
        protected = sim._get_or_create_path((1,))[0]
        victim = sim._get_or_create_path((2,))[0]
        protected.last_access = 1
        victim.last_access = 2
        sim._ensure_d(protected)
        sim._ensure_d(victim)
        self.assertTrue(sim._try_ensure_h(protected))
        self.assertTrue(sim._try_ensure_h(victim))

        self.assertIs(sim._select_duplicate_h_leaf({protected}), victim)
        self.assertIs(sim._select_duplicate_h_leaf(), protected)

    def test_boundary_l2_does_not_clear_boundary_with_h_descendant(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=2,
            l2_pages=2,
        )
        sim.process_request((1, 2, 3), timestamp=1)

        node_a = sim.root.children[1]
        node_b = node_a.children[2]
        node_c = node_b.children[3]
        self.assertTrue(node_b.has_d and node_b.has_h)
        self.assertTrue(node_c.has_h and not node_c.has_d)

        sim.process_request((4,), timestamp=2)

        self.assertTrue(node_a.has_d and node_a.has_h)
        self.assertFalse(node_b.has_d)
        self.assertTrue(node_b.has_h)
        self.assertFalse(3 in node_b.children)
        self.assertEqual(sim.metrics.h_to_deleted_evictions, 1)
        sim.sanity_check()

    def test_boundary_can_clear_boundary_after_h_descendant_deleted(self):
        sim = HiCachePolicySimulator(
            policy="boundary_l1_l2",
            l1_pages=2,
            l2_pages=2,
        )
        sim.process_request((1, 2, 3), timestamp=1)
        sim.process_request((4,), timestamp=2)

        node_a = sim.root.children[1]
        self.assertTrue(node_a.has_d and node_a.has_h)
        self.assertTrue(sim._evict_l2_one())
        self.assertTrue(node_a.has_d and node_a.has_h)
        self.assertTrue(sim._evict_l2_one())
        self.assertTrue(node_a.has_d)
        self.assertFalse(node_a.has_h)
        self.assertGreaterEqual(sim.metrics.dh_to_d_l2_dedup_evictions, 1)
        sim.sanity_check()

    def test_fixed_trace_metrics_are_stable(self):
        records = [
            TraceRecord(timestamp=1, hash_ids=(1, 2, 3)),
            TraceRecord(timestamp=2, hash_ids=(1, 2, 4)),
            TraceRecord(timestamp=3, hash_ids=(1, 2, 3)),
        ]
        metrics = simulate_policies(
            records,
            policies=["boundary_l1_l2"],
            l1_pages=2,
            l2_pages=3,
        )["boundary_l1_l2"]

        self.assertEqual(metrics.requests, 3)
        self.assertEqual(metrics.total_input_pages, 9)
        self.assertEqual(metrics.l1_hit_pages, 4)
        self.assertEqual(metrics.l2_hit_pages, 0)
        self.assertEqual(metrics.miss_pages, 5)
        self.assertLessEqual(metrics.d_pages, 2)
        self.assertLessEqual(metrics.h_pages, 3)


if __name__ == "__main__":
    unittest.main()
