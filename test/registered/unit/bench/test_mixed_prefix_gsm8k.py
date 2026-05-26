"""Unit tests for MixedPrefixGSM8KEval prefix construction.

These tests do not hit any server; they exercise only the deterministic
prefix-selection logic of ``MixedPrefixGSM8KEval``. A small synthetic jsonl
file stands in for the GSM8K dataset to keep the test hermetic.
"""

import json
import os
import tempfile
import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    _MODE_LABELS,
    MixedPrefixGSM8KEval,
)


def _write_synthetic_dataset(path: str, n: int) -> None:
    """Write ``n`` synthetic GSM8K-shaped records to ``path``."""
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "question": f"Synthetic question {i}: what is {i} + {i}?",
                        "answer": f"The answer is {2 * i}. #### {2 * i}",
                    }
                )
                + "\n"
            )


class TestMixedPrefixGSM8KEval(CustomTestCase):
    NUM_SHOTS = 4
    NUM_CLUSTERS = 3
    RANDOM_POOL_SIZE = 12
    NUM_EXAMPLES = 40

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._data_path = os.path.join(cls._tmpdir.name, "synthetic.jsonl")
        # Pool reservation = num_shots (standard) + num_shots*num_clusters
        # (clusters) + random_pool_size = 4 + 12 + 12 = 28. We write 100
        # records so the trailing 72 lines act as the test pool; the eval
        # then slices it down to ``num_examples`` (40 by default below).
        _write_synthetic_dataset(cls._data_path, 100)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _make_eval(self, seed: int = 42, num_examples=None):
        return MixedPrefixGSM8KEval(
            num_examples=(
                num_examples if num_examples is not None else self.NUM_EXAMPLES
            ),
            num_threads=1,
            num_shots=self.NUM_SHOTS,
            num_clusters=self.NUM_CLUSTERS,
            random_pool_size=self.RANDOM_POOL_SIZE,
            data_path=self._data_path,
            seed=seed,
        )

    def test_mode_0_all_share_standard_prefix(self):
        evaluator = self._make_eval()
        mode0_prefixes = {
            evaluator._build_prefix(i) for i in range(self.NUM_EXAMPLES) if i % 4 == 0
        }
        self.assertEqual(len(mode0_prefixes), 1)
        # Standard prefix must be non-empty.
        self.assertTrue(next(iter(mode0_prefixes)))

    def test_mode_1_uses_n_distinct_cluster_prefixes(self):
        evaluator = self._make_eval()
        mode1_prefixes = {
            evaluator._build_prefix(i) for i in range(self.NUM_EXAMPLES) if i % 4 == 1
        }
        # With NUM_EXAMPLES=40 (10 questions in mode 1) and NUM_CLUSTERS=3,
        # we must see exactly NUM_CLUSTERS distinct prefixes (cycling).
        self.assertEqual(len(mode1_prefixes), self.NUM_CLUSTERS)

    def test_mode_2_all_prefixes_unique(self):
        evaluator = self._make_eval()
        mode2_prefixes = [
            evaluator._build_prefix(i) for i in range(self.NUM_EXAMPLES) if i % 4 == 2
        ]
        # With NUM_SHOTS=4 sampled from a pool of 12, the chance of two
        # 4-element samples being identical (in order) is astronomically low.
        # We assert strict uniqueness here as the deterministic seed is fixed.
        self.assertEqual(len(set(mode2_prefixes)), len(mode2_prefixes))

    def test_mode_3_all_zero_shot(self):
        evaluator = self._make_eval()
        for i in range(self.NUM_EXAMPLES):
            if i % 4 == 3:
                self.assertEqual(evaluator._build_prefix(i), "")

    def test_build_prefix_is_deterministic(self):
        a = self._make_eval(seed=42)
        b = self._make_eval(seed=42)
        for i in range(self.NUM_EXAMPLES):
            self.assertEqual(a._build_prefix(i), b._build_prefix(i))

    def test_mode_2_seed_actually_matters(self):
        # Changing the seed must change at least one mode-2 prefix.
        a = self._make_eval(seed=42)
        b = self._make_eval(seed=43)
        mode2_indices = [i for i in range(self.NUM_EXAMPLES) if i % 4 == 2]
        differences = sum(
            1 for i in mode2_indices if a._build_prefix(i) != b._build_prefix(i)
        )
        self.assertGreater(differences, 0)
        # Modes 0/1/3 are seed-independent.
        for i in range(self.NUM_EXAMPLES):
            if i % 4 != 2:
                self.assertEqual(a._build_prefix(i), b._build_prefix(i))

    def test_random_pool_disjoint_from_test_lines(self):
        evaluator = self._make_eval(num_examples=None)
        train_questions = {item["question"] for item in evaluator._random_pool}
        test_questions = {item["question"] for item in evaluator._lines}
        self.assertEqual(train_questions & test_questions, set())

    def test_per_mode_metric_key(self):
        evaluator = self._make_eval()
        for i in range(20):
            metrics = evaluator._extra_sample_metrics(i, 1.0)
            self.assertEqual(metrics, {f"score_{_MODE_LABELS[i % 4]}": 1.0})

    def test_insufficient_dataset_raises(self):
        tiny = os.path.join(self._tmpdir.name, "tiny.jsonl")
        _write_synthetic_dataset(tiny, n=5)
        with self.assertRaises(ValueError):
            MixedPrefixGSM8KEval(
                num_examples=1,
                num_threads=1,
                num_shots=self.NUM_SHOTS,
                num_clusters=self.NUM_CLUSTERS,
                random_pool_size=self.RANDOM_POOL_SIZE,
                data_path=tiny,
            )


if __name__ == "__main__":
    unittest.main()
