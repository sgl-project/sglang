"""CPU tests for the standalone HiCache warm-cache benchmark."""

import random
import runpy
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The benchmark script lives outside the installed sglang package.
script = Path(__file__).resolve().parents[4] / "benchmark/hicache/bench_warm_cache.py"
build_prompts = runpy.run_path(str(script))["build_prompts"]


class TestWarmCachePrompts(unittest.TestCase):
    def test_full_prefix_is_shared_by_independent_requests(self):
        vocab = list(range(16))
        rng = random.Random(42)
        rows = build_prompts(vocab, 100, 100, 3, rng)

        expected_rng = random.Random(42)
        expected = expected_rng.choices(vocab, k=100)
        self.assertEqual(rows, [{"input_ids": expected, "prompt_len": 100}] * 3)
        self.assertEqual(rng.getstate(), expected_rng.getstate())
        self.assertEqual(len({id(row) for row in rows}), 3)
        for row in rows:
            self.assertIs(row["input_ids"], rows[0]["input_ids"])

    def test_partial_prefix_draws_a_suffix_for_each_request(self):
        vocab = list(range(16))
        rng = random.Random(42)
        rows = build_prompts(vocab, 100, 99, 3, rng)

        expected_rng = random.Random(42)
        prefix = expected_rng.choices(vocab, k=99)
        for row in rows:
            self.assertEqual(
                row["input_ids"], prefix + expected_rng.choices(vocab, k=1)
            )
        self.assertEqual(len({id(row["input_ids"]) for row in rows}), 3)
        self.assertEqual(rng.getstate(), expected_rng.getstate())


if __name__ == "__main__":
    unittest.main()
