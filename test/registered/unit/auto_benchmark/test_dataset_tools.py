import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from auto_benchmark import AutoBenchmarkTestCase

from sglang.auto_benchmark_lib import infer_backend, prepare_dataset
from sglang.benchmark.datasets.autobench import sample_autobench_requests
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-test-1-gpu-small")


class TestAutoBenchmarkDatasetTools(AutoBenchmarkTestCase):
    def test_prepare_custom_autobench_dataset(self):
        dataset_path = self._write_autobench_jsonl()
        output_path = self.tmpdir_path / "prepared.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "custom",
                "path": dataset_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertTrue(Path(prepared_path).exists())
        converted_rows = sample_autobench_requests(
            dataset_path=prepared_path,
            num_requests=0,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(converted_rows), 2)

    def test_invalid_json_like_prompt_falls_back_to_plain_text(self):
        path = self.tmpdir_path / "jsonlike.autobench.jsonl"
        path.write_text(
            json.dumps({"prompt": "[not actually json", "output_len": 8}) + "\n",
            encoding="utf-8",
        )

        rows = sample_autobench_requests(
            dataset_path=str(path),
            num_requests=0,
            tokenizer=self.tokenizer,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].prompt, "[not actually json")

    def test_prepare_sharegpt_dataset(self):
        sharegpt_path = self._write_sharegpt_json()
        output_path = self.tmpdir_path / "sharegpt.autobench.jsonl"

        prepared_path, rows, summary = prepare_dataset(
            dataset_cfg={
                "kind": "sharegpt",
                "path": sharegpt_path,
                "num_prompts": 2,
            },
            tokenizer_path=str(self.tokenizer_dir),
            model=None,
            output_path=str(output_path),
        )

        self.assertEqual(prepared_path, str(output_path))
        self.assertEqual(summary["num_requests"], 2)
        self.assertEqual(len(rows), 2)

    def test_prepare_custom_dataset_requires_path(self):
        with self.assertRaisesRegex(ValueError, "dataset.path is required"):
            prepare_dataset(
                dataset_cfg={"kind": "custom"},
                tokenizer_path=str(self.tokenizer_dir),
                model=None,
                output_path=str(self.tmpdir_path / "missing.autobench.jsonl"),
            )

    def test_infer_backend(self):
        prompt_rows = [SimpleNamespace(prompt="tok_1 tok_2")]
        chat_rows = [SimpleNamespace(prompt=[{"role": "user", "content": "tok_1"}])]
        token_id_rows = [SimpleNamespace(prompt=[1, 2, 3])]

        self.assertEqual(infer_backend("auto", prompt_rows), "sglang-oai")
        self.assertEqual(infer_backend("auto", chat_rows), "sglang-oai-chat")
        self.assertEqual(infer_backend("auto", token_id_rows), "sglang")


if __name__ == "__main__":
    unittest.main()
