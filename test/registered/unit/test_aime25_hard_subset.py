import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.aime25_hard_subset import (
    DEFAULT_DSV4_FLASH_HARD_IDS,
    parse_problem_ids,
    select_aime25_rows,
    write_jsonl,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAime25HardSubset(CustomTestCase):
    def test_parse_problem_ids(self):
        self.assertEqual(
            parse_problem_ids("aime25-13, aime25-14,,aime25-29"),
            ("aime25-13", "aime25-14", "aime25-29"),
        )
        self.assertEqual(parse_problem_ids(None), DEFAULT_DSV4_FLASH_HARD_IDS)
        with self.assertRaises(ValueError):
            parse_problem_ids(" , ")

    def test_select_rows_preserves_requested_order_and_schema(self):
        rows = [
            {"question": "q0", "answer": 0},
            {"id": "aime25-13", "problem": "q13", "expected_answer": "60"},
            {"id": "aime25-29", "problem": "q29", "expected_answer": "240"},
        ]

        selected = select_aime25_rows(rows, ("aime25-29", "aime25-13"))

        self.assertEqual(
            selected,
            [
                {"id": "aime25-29", "problem": "q29", "expected_answer": "240"},
                {"id": "aime25-13", "problem": "q13", "expected_answer": "60"},
            ],
        )

    def test_select_rows_reports_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "aime25-99"):
            select_aime25_rows(
                [{"id": "aime25-13", "problem": "q13", "expected_answer": "60"}],
                ("aime25-99",),
            )

    def test_write_jsonl(self):
        rows = [{"id": "aime25-13", "problem": "q13", "expected_answer": "60"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_jsonl(rows, Path(tmpdir) / "subset.jsonl")
            loaded = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
