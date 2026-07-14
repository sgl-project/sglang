import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
    build_uninitialized_sps_table,
    is_uninitialized_sps_table,
    load_sps_table_from_path,
    profile_sps_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[8, 16, 32, 64],
        sample_steps_per_sec=[1000.0, 950.0, 500.0, 480.0],
        max_batch_tokens=128,
    )


class TestSpsCostTableInvariants(CustomTestCase):
    def test_rejects_non_increasing_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 8, 16],
                sample_steps_per_sec=[1.0, 2.0, 3.0],
                max_batch_tokens=16,
            )

    def test_rejects_unsorted_batch_tokens(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[16, 8],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=16,
            )

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0],
                max_batch_tokens=16,
            )

    def test_rejects_empty_table(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[],
                sample_steps_per_sec=[],
                max_batch_tokens=0,
            )

    def test_rejects_max_below_largest_probe(self):
        with self.assertRaises(ValueError):
            SpsCostTable(
                sample_batch_tokens=[8, 16],
                sample_steps_per_sec=[1.0, 2.0],
                max_batch_tokens=15,
            )


class TestSpsCostTableLookup(CustomTestCase):
    def test_lookup_exact_probe_returns_that_sps(self):
        table = _make_table()
        self.assertEqual(table.lookup(8), 1000.0)
        self.assertEqual(table.lookup(16), 950.0)
        self.assertEqual(table.lookup(32), 500.0)
        self.assertEqual(table.lookup(64), 480.0)

    def test_lookup_floors_to_lower_captured_probe(self):
        table = _make_table()
        self.assertEqual(table.lookup(31), 950.0)
        self.assertEqual(table.lookup(63), 500.0)

    def test_lookup_below_first_probe_clamps_to_first(self):
        table = _make_table()
        self.assertEqual(table.lookup(1), 1000.0)
        self.assertEqual(table.lookup(7), 1000.0)

    def test_lookup_above_last_probe_clamps_to_last(self):
        table = _make_table()
        self.assertEqual(table.lookup(65), 480.0)
        self.assertEqual(table.lookup(10_000), 480.0)


class TestLoadSpsTableFromPath(CustomTestCase):
    def test_load_from_path_round_trips_table_and_lookup(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = load_sps_table_from_path(str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)
        for batch_tokens in (1, 8, 31, 64, 200):
            self.assertEqual(loaded.lookup(batch_tokens), table.lookup(batch_tokens))


class TestFlatTableLookupIsConstant(CustomTestCase):
    def test_flat_table_lookup_is_one_for_any_batch(self):
        flat = SpsCostTable(
            sample_batch_tokens=[1],
            sample_steps_per_sec=[1.0],
            max_batch_tokens=4096,
        )
        for batch_tokens in (0, 1, 2, 17, 256, 100_000):
            self.assertEqual(flat.lookup(batch_tokens), 1.0)


class TestProfileSpsTable(CustomTestCase):
    def test_profile_sorts_out_of_order_probes(self):
        table = profile_sps_table(
            probes=[(32, 500.0), (8, 1000.0), (16, 950.0)],
        )
        self.assertEqual(table.sample_batch_tokens, [8, 16, 32])
        self.assertEqual(table.sample_steps_per_sec, [1000.0, 950.0, 500.0])

    def test_profile_rejects_duplicate_batch_tokens(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[(8, 1000.0), (8, 900.0)])

    def test_profile_rejects_empty_probes(self):
        with self.assertRaises(ValueError):
            profile_sps_table(probes=[])

    def test_profile_max_batch_tokens_defaults_to_largest_probe(self):
        table = profile_sps_table(probes=[(8, 1000.0), (64, 480.0), (16, 950.0)])
        self.assertEqual(table.max_batch_tokens, 64)

    def test_profile_honors_explicit_max_batch_tokens(self):
        table = profile_sps_table(
            probes=[(8, 1000.0), (16, 950.0)], max_batch_tokens=256
        )
        self.assertEqual(table.max_batch_tokens, 256)


def _build_sps_cost_table_for(*, sps_table_path):
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        build_sps_cost_table,
    )

    server_args = SimpleNamespace(
        speculative_dspark_sps_table_path=sps_table_path,
        max_running_requests=4,
    )
    return build_sps_cost_table(server_args=server_args, verify_num_draft_tokens=5)


class TestBuildSpsCostTableContract(CustomTestCase):
    def test_unset_table_path_returns_flat_table(self):
        for sps_table_path in (None, ""):
            table = _build_sps_cost_table_for(sps_table_path=sps_table_path)
            self.assertEqual(table.sample_batch_tokens, [1])
            self.assertEqual(table.sample_steps_per_sec, [1.0])
            self.assertEqual(table.max_batch_tokens, 20)

    def test_real_path_loads_table(self):
        table = _make_table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sps.json"
            path.write_text(table.to_json(), encoding="utf-8")
            loaded = _build_sps_cost_table_for(sps_table_path=str(path))
        self.assertEqual(loaded.sample_batch_tokens, table.sample_batch_tokens)
        self.assertEqual(loaded.sample_steps_per_sec, table.sample_steps_per_sec)
        self.assertEqual(loaded.max_batch_tokens, table.max_batch_tokens)


class TestIsUninitializedSpsTable(CustomTestCase):
    def test_additive_table_is_never_uninitialized(self):
        table = SpsAdditiveCostTable(
            bias_seconds=0.1,
            bs_probes=[128, 192, 256],
            alpha_seconds=[0.0, 0.008, 0.016],
            m_probes=[384, 512, 1024],
            theta_seconds=[0.0, 0.02, 0.1],
        )
        self.assertFalse(is_uninitialized_sps_table(table))

    def test_placeholder_diagonal_table_is_uninitialized(self):
        self.assertTrue(
            is_uninitialized_sps_table(
                build_uninitialized_sps_table(max_batch_tokens=128)
            )
        )

    def test_real_diagonal_table_is_initialized(self):
        self.assertFalse(is_uninitialized_sps_table(_make_table()))


if __name__ == "__main__":
    unittest.main()
