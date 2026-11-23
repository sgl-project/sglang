"""Dynamic workload regressions for accidental prefix-cache reuse."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.benchmark.datasets.dynamic import DynamicDataset
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Tokenizer:
    """A reversible vocabulary with a BOS token, without model downloads."""

    vocab_size = 256
    all_special_ids = [0]

    def get_vocab(self):
        return {str(index): index for index in range(self.vocab_size)}

    def num_special_tokens_to_add(self):
        return 1

    def encode(self, text, add_special_tokens=True):
        tokens = [int(word) for word in text.split()]
        return ([0] if add_special_tokens else []) + tokens

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class TestDynamicWorkload(CustomTestCase):
    def load(self, plan, tokenizer=None, dataset=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "workload.json"
            plan_path.write_text(json.dumps(plan))
            dataset_path = root / "sharegpt.json"
            if dataset is not None:
                dataset_path.write_text(json.dumps(dataset))
            # Keep the shared-prefix generator's on-disk cache test-local.
            with patch(
                "sglang.benchmark.datasets.generated_shared_prefix."
                "get_gen_prefix_cache_path",
                return_value=root / "prefix.pkl",
            ):
                return DynamicDataset(str(plan_path), str(dataset_path), seed=7).load(
                    tokenizer or Tokenizer()
                )

    def plan(self, input_len=64, source="random-ids"):
        return {
            "source": source,
            "prompt_pool_size": 2,
            "prompt_seed": 123,
            "phases": [
                {
                    "name": name,
                    "duration": 1,
                    "request_rate": 12,
                    "input_len": input_len,
                    "output_len": 2,
                }
                for name in ("steady", "burst", "steady-c002")
            ],
        }

    def test_ordinary_requests_do_not_reuse_cached_prefixes(self):
        """The old eight-prompt pool repeated requests within and across phases."""
        dataset = [
            {"conversations": [{"value": "1 2 3"}, {"value": "4 5"}]},
            {"conversations": [{"value": "6 7 8"}, {"value": "9 10"}]},
        ]
        tokenizer = Tokenizer()
        for source in ("random-ids", "sharegpt"):
            for length in (4, 64):
                with self.subTest(source=source, length=length):
                    rows = self.load(self.plan(length, source), dataset=dataset)
                    encoded = [tuple(tokenizer.encode(row.prompt)) for row in rows]
                    self.assertEqual(len(rows), 36)
                    self.assertEqual(len(set(encoded)), len(rows))
                    # Uniqueness at the end would still allow long cache hits.
                    self.assertEqual(len({tokens[:8] for tokens in encoded}), len(rows))
                    self.assertTrue(all(len(tokens) == length for tokens in encoded))
                    self.assertTrue(all(row.prompt_len == length for row in rows))
                    self.assertTrue(all(row.text_prompt_len == length for row in rows))

    def test_prompt_seed_replays_content_without_changing_arrivals(self):
        """A fresh prompt seed separates runs even when arrival seeds match."""
        plan = self.plan()
        first = self.load(plan)
        replay = self.load(plan)
        plan["prompt_seed"] = 456
        other = self.load(plan)
        self.assertEqual([row.prompt for row in first], [row.prompt for row in replay])
        self.assertTrue(all(a.prompt != b.prompt for a, b in zip(first, other)))
        self.assertEqual(
            [row.timestamp for row in first], [row.timestamp for row in other]
        )
        del plan["prompt_seed"]
        with patch(
            "sglang.benchmark.datasets.dynamic.secrets.randbits", side_effect=[123, 456]
        ):
            fresh_first = self.load(plan)
            fresh_other = self.load(plan)
        self.assertEqual(
            [row.prompt for row in first], [row.prompt for row in fresh_first]
        )
        self.assertEqual(
            [row.prompt for row in other], [row.prompt for row in fresh_other]
        )

    def test_shared_prefix_still_reuses_its_configured_pool(self):
        """Cache scenarios must retain intentional reuse after uniqueness fixes."""
        rows = self.load(self.plan(source="generated-shared-prefix"))
        self.assertEqual(len({row.prompt for row in rows}), 2)
        self.assertEqual(rows[0].prompt, rows[2].prompt)
        self.assertEqual(rows[0].prompt, rows[12].prompt)
        self.assertTrue(all(len(Tokenizer().encode(row.prompt)) == 64 for row in rows))

    def test_skipped_phases_leave_the_planned_time_gap(self):
        """Capability filtering must not compress later phases into skipped slots."""
        plan = self.plan()
        for phase, start in zip(plan["phases"], (0, 8, 9)):
            phase["start_time"] = start
        rows = self.load(plan)
        self.assertEqual(
            [rows[index].timestamp for index in (0, 12, 24)], [0, 8000, 9000]
        )

    def test_exhausted_short_prompts_fail_instead_of_silently_reusing(self):
        """A one-content-token vocabulary cannot supply two uncached requests."""
        tokenizer = Tokenizer()
        tokenizer.vocab_size = 2
        with self.assertRaisesRegex(ValueError, "Cannot generate another unique"):
            self.load(self.plan(input_len=2), tokenizer=tokenizer)


if __name__ == "__main__":
    unittest.main()
