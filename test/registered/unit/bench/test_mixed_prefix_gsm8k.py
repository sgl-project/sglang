"""Unit tests for MixedPrefixGSM8KEval prefix construction.

Exercises only the deterministic prefix-selection logic; no server needed.
A small synthetic jsonl stands in for the GSM8K dataset to keep the test
hermetic.
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


from sglang.test.simple_eval_gsm8k import get_one_example
from sglang.test.simple_eval_mixed_prefix_gsm8k import MixedPrefixGSM8KEval


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
    SECONDARY_POOL_SIZE = 12
    NUM_EXAMPLES = 40

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._data_path = os.path.join(cls._tmpdir.name, "synthetic.jsonl")
        # Pool reservation = num_shots + secondary_pool_size = 4 + 12 = 16.
        # 100 records overall leaves 84 for the test pool, which the eval
        # truncates to NUM_EXAMPLES=40.
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
            secondary_pool_size=self.SECONDARY_POOL_SIZE,
            data_path=self._data_path,
            seed=seed,
        )

    def _primary_question_set(self, evaluator):
        return {item["question"] for item in evaluator._primary_shots}

    def _secondary_question_set(self, evaluator):
        return {item["question"] for item in evaluator._secondary_pool}

    def test_primary_then_secondary_layout(self):
        # Every produced prefix decomposes into a (possibly empty) prefix of
        # primary_shots followed by some subset of secondary_pool.
        evaluator = self._make_eval()
        primary_qs = [item["question"] for item in evaluator._primary_shots]
        secondary_qs = self._secondary_question_set(evaluator)

        for i in range(self.NUM_EXAMPLES):
            prefix = evaluator._build_prefix(i)
            # Walk down primary_shots tokens in order; whatever matches at
            # the start is the chosen primary segment.
            k = 0
            for shot_idx in range(len(primary_qs)):
                shot_line = (
                    get_one_example(
                        evaluator._primary_shots, shot_idx, include_answer=True
                    )
                    + "\n\n"
                )
                if prefix[: len(shot_line)] == shot_line:
                    prefix = prefix[len(shot_line) :]
                    k += 1
                else:
                    break
            # Anything left must come from secondary_pool only.
            remaining_questions = []
            for line in prefix.strip().split("Question: ")[1:]:
                # 'foo?\nAnswer: ...' -> we just need to recover the question text
                q = "Question: " + line.split("\nAnswer:")[0]
                remaining_questions.append(q)
            for q in remaining_questions:
                self.assertIn(q, {f"Question: {x}" for x in secondary_qs})

    def test_build_prefix_is_deterministic(self):
        a = self._make_eval(seed=42)
        b = self._make_eval(seed=42)
        for i in range(self.NUM_EXAMPLES):
            self.assertEqual(a._build_prefix(i), b._build_prefix(i))

    def test_seed_actually_matters(self):
        a = self._make_eval(seed=42)
        b = self._make_eval(seed=43)
        differences = sum(
            1
            for i in range(self.NUM_EXAMPLES)
            if a._build_prefix(i) != b._build_prefix(i)
        )
        self.assertGreater(differences, self.NUM_EXAMPLES // 2)

    def test_primary_and_secondary_disjoint_from_test_lines(self):
        evaluator = self._make_eval(num_examples=None)
        primary_qs = self._primary_question_set(evaluator)
        secondary_qs = self._secondary_question_set(evaluator)
        test_qs = {item["question"] for item in evaluator._lines}
        self.assertEqual(primary_qs & test_qs, set())
        self.assertEqual(secondary_qs & test_qs, set())
        self.assertEqual(primary_qs & secondary_qs, set())

    def test_num_primary_distribution_spans_range(self):
        # Across the 40 questions, num_primary should sample most of {0..N};
        # bare minimum: not all zero and not all NUM_SHOTS.
        evaluator = self._make_eval()
        primary_lines = [
            (get_one_example(evaluator._primary_shots, i, include_answer=True) + "\n\n")
            for i in range(self.NUM_SHOTS)
        ]
        chosen_ks = []
        for i in range(self.NUM_EXAMPLES):
            prefix = evaluator._build_prefix(i)
            k = 0
            for line in primary_lines:
                if prefix[: len(line)] == line:
                    prefix = prefix[len(line) :]
                    k += 1
                else:
                    break
            chosen_ks.append(k)
        self.assertGreater(len(set(chosen_ks)), 1)

    def test_insufficient_dataset_raises(self):
        tiny = os.path.join(self._tmpdir.name, "tiny.jsonl")
        _write_synthetic_dataset(tiny, n=5)
        with self.assertRaises(ValueError):
            MixedPrefixGSM8KEval(
                num_examples=1,
                num_threads=1,
                num_shots=self.NUM_SHOTS,
                secondary_pool_size=self.SECONDARY_POOL_SIZE,
                data_path=tiny,
            )


if __name__ == "__main__":
    unittest.main()
