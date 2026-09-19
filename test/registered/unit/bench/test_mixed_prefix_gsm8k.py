import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_common import CompletionSampler
from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    GSM8KEval,
    MixedPrefixGSM8KEval,
    get_one_example,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


def _write_synthetic_dataset(path: str, n: int) -> None:
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

    def _primary_lines(self, evaluator) -> List[str]:
        return [
            get_one_example(evaluator._primary_shots, j, include_answer=True) + "\n\n"
            for j in range(self.NUM_SHOTS)
        ]

    def _decompose(self, evaluator, prefix: str) -> Tuple[int, List[str]]:
        k = 0
        for line in self._primary_lines(evaluator):
            if prefix.startswith(line):
                prefix = prefix[len(line) :]
                k += 1
            else:
                break
        remainder_questions: List[str] = []
        if prefix:
            chunks = prefix.split("\n\n")
            for chunk in chunks:
                if chunk.startswith("Question: "):
                    q_text = chunk[len("Question: ") :].split("\nAnswer:")[0]
                    remainder_questions.append(q_text)
        return k, remainder_questions

    def test_primary_segment_is_strict_prefix_of_primary_shots(self):
        e = self._make_eval()
        for i in range(self.NUM_EXAMPLES):
            k, _ = self._decompose(e, e._build_prefix(i))
            self.assertGreaterEqual(k, 0)
            self.assertLessEqual(k, self.NUM_SHOTS)

    def test_remainder_questions_come_from_secondary_pool(self):
        e = self._make_eval()
        secondary_qs = {item["question"] for item in e._secondary_pool}
        for i in range(self.NUM_EXAMPLES):
            _, remainder = self._decompose(e, e._build_prefix(i))
            for q in remainder:
                self.assertIn(q, secondary_qs)

    def test_remainder_no_duplicates_within_one_query(self):
        e = self._make_eval()
        for i in range(self.NUM_EXAMPLES):
            _, remainder = self._decompose(e, e._build_prefix(i))
            self.assertEqual(
                len(remainder),
                len(set(remainder)),
                f"query {i} has duplicate secondary samples",
            )

    def test_remainder_size_within_secondary_pool_bound(self):
        e = self._make_eval()
        for i in range(self.NUM_EXAMPLES):
            _, remainder = self._decompose(e, e._build_prefix(i))
            self.assertGreaterEqual(len(remainder), 0)
            self.assertLessEqual(len(remainder), self.SECONDARY_POOL_SIZE)

    def test_primary_depth_takes_multiple_values(self):
        e = self._make_eval()
        ks = {
            self._decompose(e, e._build_prefix(i))[0] for i in range(self.NUM_EXAMPLES)
        }
        self.assertGreater(len(ks), 2, f"k values seen: {ks}")

    def test_secondary_size_takes_multiple_values(self):
        e = self._make_eval()
        sizes = {
            len(self._decompose(e, e._build_prefix(i))[1])
            for i in range(self.NUM_EXAMPLES)
        }
        self.assertGreater(len(sizes), 2, f"sizes seen: {sizes}")

    def test_two_queries_share_min_primary_prefix(self):
        e = self._make_eval()
        lines = self._primary_lines(e)
        prefixes = [e._build_prefix(i) for i in range(self.NUM_EXAMPLES)]
        ks = [self._decompose(e, p)[0] for p in prefixes]
        for i in range(self.NUM_EXAMPLES):
            for j in range(i + 1, self.NUM_EXAMPLES):
                shared = "".join(lines[: min(ks[i], ks[j])])
                self.assertTrue(prefixes[i].startswith(shared))
                self.assertTrue(prefixes[j].startswith(shared))

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

    def test_pools_and_test_lines_pairwise_disjoint(self):
        e = self._make_eval(num_examples=None)
        primary_qs = {item["question"] for item in e._primary_shots}
        secondary_qs = {item["question"] for item in e._secondary_pool}
        test_qs = {item["question"] for item in e._lines}
        self.assertEqual(primary_qs & secondary_qs, set())
        self.assertEqual(primary_qs & test_qs, set())
        self.assertEqual(secondary_qs & test_qs, set())

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
                seed=42,
            )


class TestOrderedGSM8KBatching(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._data_path = os.path.join(cls._tmpdir.name, "synthetic.jsonl")
        _write_synthetic_dataset(cls._data_path, 12)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_evaluator_uses_ordered_request_batches(self):
        class FakeBatchSampler:
            def __init__(self):
                self.batch_sizes = []

            @staticmethod
            def _pack_message(role, content):
                return {"role": role, "content": content}

            def __call__(self, _messages):
                raise AssertionError("per-example request path must not run")

            def sample_batch(self, message_lists):
                self.batch_sizes.append(len(message_lists))
                responses = []
                for messages in message_lists:
                    match = re.findall(
                        r"Synthetic question (\d+)", messages[0]["content"]
                    )
                    responses.append(str(2 * int(match[-1])))
                return responses

        evaluator = GSM8KEval(
            num_examples=5,
            num_threads=128,
            num_shots=2,
            data_path=self._data_path,
            request_batch_size=3,
        )
        sampler = FakeBatchSampler()

        result = evaluator(sampler)

        self.assertEqual(sampler.batch_sizes, [3, 2])
        self.assertEqual(result.score, 1.0)

    def test_completion_sampler_preserves_batch_choice_order(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(index=1, text="second"),
                SimpleNamespace(index=0, text="first"),
            ],
            usage=SimpleNamespace(completion_tokens=7),
        )
        sampler = CompletionSampler.__new__(CompletionSampler)
        sampler.client = SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(return_value=response))
        )
        sampler.model = "model"
        sampler.temperature = 0.0
        sampler.top_p = 1.0
        sampler.max_tokens = 16
        sampler.stop = None
        sampler._completion_tokens = []

        outputs = sampler.sample_batch(
            [
                [{"role": "user", "content": "prompt one"}],
                [{"role": "user", "content": "prompt two"}],
            ]
        )

        self.assertEqual(outputs, ["first", "second"])
        self.assertEqual(sampler._completion_tokens, [7])
        self.assertEqual(
            sampler.client.completions.create.call_args.kwargs["prompt"],
            ["prompt one", "prompt two"],
        )


if __name__ == "__main__":
    unittest.main()
