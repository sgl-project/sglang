import tempfile
import unittest
from pathlib import Path

from sglang.test.aime25_hard_subset import write_jsonl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_aime25 import AIME25Eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAime25DataPath(CustomTestCase):
    def test_aime25_eval_loads_hard_subset_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_jsonl(
                [
                    {
                        "id": "aime25-13",
                        "problem": "q13",
                        "expected_answer": "60",
                    },
                    {
                        "id": "aime25-29",
                        "question": "q29",
                        "answer": "240",
                    },
                ],
                Path(tmpdir) / "subset.jsonl",
            )

            eval_obj = AIME25Eval(num_examples=None, num_threads=1, data_path=path)

        self.assertEqual(
            eval_obj.examples,
            [
                {"question": "q13", "answer": "60"},
                {"question": "q29", "answer": "240"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
