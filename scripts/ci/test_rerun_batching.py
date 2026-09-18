"""Run with python -m unittest discover -s scripts/ci -p test_rerun_batching.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
import slash_command_handler as handler


class TestRerunBatching(unittest.TestCase):
    def test_unified_cache_group_fits_separate_jobs(self):
        """The five cache tests must not share one 60-minute test step."""
        root = Path(__file__).resolve().parents[2]
        prefix = "registered/radix_cache/unified_radix_tree/"
        paths = [
            "linker/test_unified_cache_linker_kl_dsv4.py",
            "test_unified_radix_cache_hicache_pp_kl.py",
            "test_unified_radix_cache_kl_cp.py",
            "test_unified_radix_cache_kl_dsv4.py",
            "test_unified_radix_cache_kl_mamba.py",
        ]
        previous_cwd = os.getcwd()
        try:
            os.chdir(root)
            entries = [
                entry
                for path in paths
                for entry in handler._resolve_test_spec(prefix + path)
            ]
        finally:
            os.chdir(previous_cwd)
        self.assertTrue(all(entry.get("error") is None for entry in entries))
        self.assertEqual(len(entries), 5)
        batches = handler._split_rerun_batches(entries)
        self.assertGreater(len(batches), 1)
        self.assertCountEqual(
            [entry["test_command"] for batch in batches for entry in batch],
            [prefix + path for path in paths],
        )
        for batch in batches:
            self.assertLessEqual(sum(entry["est_time"] for entry in batch), 2400)

    def test_unknown_and_oversized_tests_are_isolated(self):
        entries = [{"est_time": value} for value in [None, 3000, 1200, 1200, 1]]
        batches = handler._split_rerun_batches(entries)
        self.assertEqual(
            batches, [[entries[0]], [entries[1]], entries[2:4], [entries[4]]]
        )
        self.assertEqual(handler._split_rerun_batches([]), [])

    def test_estimates_are_literals_not_executed(self):
        for args, expected in [
            ('est_time=210, runner_config="4-gpu-h100"', 210),
            ('8, suite="base-a-test-cpu"', 8),
            ("est_time=1.5", 1.5),
            ("est_time=0", None),
            ("est_time=-1", None),
            ("est_time=True", None),
            ("est_time=1e309", None),
            ('est_time="210"', None),
            ("est_time=unknown()", None),
            ("est_time=", None),
            ("", None),
        ]:
            with self.subTest(args=args):
                self.assertEqual(handler._extract_est_time(args), expected)


if __name__ == "__main__":
    unittest.main()
