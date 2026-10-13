"""Client contract tests using real httpx response streams, without model weights."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "cpu_prefix_example",
    Path(__file__).resolve().parents[4] / "examples/runtime/cpu_prefix_reuse.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class TestCPUPrefixReuse(CustomTestCase):
    def run_cli(self, tokenizer, **completion):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "completed.jsonl"
            with (
                patch(
                    "sys.argv",
                    [
                        "example",
                        "--model-path",
                        "local",
                        "--model",
                        "test",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(
                    example.AutoTokenizer, "from_pretrained", return_value=tokenizer
                ),
                patch.object(example, "stream_completion", **completion),
                patch("builtins.print"),
            ):
                code = example.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        return code, rows

    def test_cli_serializes_ids_across_tokenizer_return_types(self):
        # Transformers can return a mapping unless return_dict=False is explicit.
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = lambda *args, **kwargs: (
            [1, 2] if kwargs.get("return_dict") is False else {"input_ids": [1, 2]}
        )
        result = {"text": '{"diagnosis":"normal","action":"KEEP","reason":"ok"}'}
        code, rows = self.run_cli(tokenizer, return_value=result)
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), len(example.CASES))
        self.assertTrue(all(row["status"] == "completed" for row in rows))
        self.assertTrue(all(row["input_tokens"] == 2 for row in rows))

    def test_cli_retains_request_errors_and_returns_failure(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = [1, 2]
        code, rows = self.run_cli(
            tokenizer, side_effect=ValueError("backend unavailable")
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(rows), len(example.CASES))
        self.assertTrue(all(row["status"] == "error" for row in rows))
        self.assertTrue(all(row["error"] == "backend unavailable" for row in rows))

    def test_fences_do_not_repair_wrong_labels(self):
        text = '```json\n{"diagnosis":"wrong", "action":"KEEP", "reason":"x"}\n```'
        value, raw = example.parse_output(text)
        self.assertFalse(raw)
        self.assertEqual(value["diagnosis"], "wrong")

    def test_schema_and_extra_commentary_rejected(self):
        for text in (
            '{"diagnosis":"normal"}',
            '```json\n{"diagnosis":"normal","action":"KEEP","reason":"x","extra":1}\n```',
            'comment\n{"diagnosis":"normal","action":"KEEP","reason":"x"}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                example.parse_output(text)

    def run_stream(self, usage, *, report=True, done=True, error=False):
        chunks = [{"choices": [{"text": ""}]}, {"choices": [{"text": "hello"}]}]
        if error:
            chunks.append({"error": {"message": "backend failure"}})
        if usage is not None:
            chunks.append({"choices": [], "usage": usage})
        wire = ": heartbeat\n\n" + "".join(
            "data: " + json.dumps(c) + "\n\n" for c in chunks
        )
        if done:
            wire += "data: [DONE]\n\n"
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=wire)
            )
        )
        with client:
            return example.stream_completion(
                client,
                "http://test/completions",
                {"prompt": list(range(10)), "max_tokens": 96},
                report,
            )

    def test_complete_stream_and_reported_hit(self):
        result = self.run_stream(
            {"completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 8}}
        )
        self.assertEqual(
            (result["text"], result["cached_tokens"], result["completion_tokens"]),
            ("hello", 8, 1),
        )
        self.assertGreaterEqual(result["full_s"], result["ttft_s"])

    def test_verified_zero_and_null_details(self):
        self.assertEqual(self.run_stream({"completion_tokens": 1})["cached_tokens"], 0)
        self.assertEqual(
            self.run_stream({"completion_tokens": 1, "prompt_tokens_details": None})[
                "cached_tokens"
            ],
            0,
        )

    def test_unknown_stays_unknown(self):
        self.assertIsNone(
            self.run_stream({"completion_tokens": 1}, report=False)["cached_tokens"]
        )

    def test_no_usage_is_error_under_report_contract(self):
        with self.assertRaisesRegex(ValueError, "final usage"):
            self.run_stream(None)

    def test_missing_done_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_stream({"completion_tokens": 1}, done=False)

    def test_hit_outside_prompt_rejected(self):
        for count in (-1, 11, True, "8"):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "cached-token"),
            ):
                self.run_stream({"prompt_tokens_details": {"cached_tokens": count}})

    def test_backend_error_is_retained_as_failure(self):
        with self.assertRaisesRegex(ValueError, "backend failure"):
            self.run_stream({"completion_tokens": 1}, error=True)


if __name__ == "__main__":
    unittest.main()
