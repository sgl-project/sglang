"""Exercise the installed sgl-eval against a local OpenAI-compatible server."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval import (
    _print_truncated_samples,
    api_base_url,
    run_sgl_eval,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="stage-a-test-cpu-intel")


class TestSglEvalAdapter(CustomTestCase):
    def test_base_urls(self):
        for args in (
            SimpleNamespace(host="127.0.0.1", port=30000),
            SimpleNamespace(host="http://127.0.0.1", port=30000),
            SimpleNamespace(base_url="http://127.0.0.1:30000/v1/"),
        ):
            self.assertEqual(api_base_url(args), "http://127.0.0.1:30000/v1")

    def test_truncated_sample_previews_are_bounded(self):
        from sgl_eval.types import Example, ExampleResult, Sample

        response = "start " + "repeated reasoning " * 10000 + "end"
        examples = [
            ExampleResult(
                example=Example(id=str(index), inputs={}, target="7"),
                samples=[
                    Sample(text="Normal answer", finish_reason="stop"),
                    Sample(
                        text=response,
                        finish_reason="length",
                        completion_tokens=16384,
                    ),
                ],
                scores=[1.0, 0.0],
                extracted=["7", None],
            )
            for index in range(5)
        ]
        with patch("builtins.print") as output:
            _print_truncated_samples(SimpleNamespace(per_example=examples))
        self.assertEqual(output.call_count, 3)
        for index, call in enumerate(output.call_args_list):
            line = call.args[0]
            self.assertLess(len(line), 1500)
            preview = json.loads(line.removeprefix("sgl-eval truncated sample: "))
            self.assertEqual(preview["example_id"], str(index))
            self.assertEqual(preview["repeat"], 1)
            self.assertEqual(preview["completion_tokens"], 16384)
            self.assertEqual(preview["score"], 0.0)
            self.assertEqual(preview["text_chars"], len(response))
            self.assertTrue(preview["head"].startswith("start "))
            self.assertTrue(preview["tail"].endswith("end"))

    def test_gsm8k_sgl_eval_opt_in_dispatches_to_adapter(self):
        args = SimpleNamespace(eval_name="gsm8k", api="sgl_eval")
        with patch(
            "sglang.test.sgl_eval.run_sgl_eval", return_value={"score": 1}
        ) as evaluate:
            self.assertEqual(run_eval(args), {"score": 1})
            evaluate.assert_called_once_with(args)

    def test_real_sgl_eval_chat_grading_and_metrics(self):
        self._check_chat_grading()

    def test_real_sgl_eval_grading_from_worker_thread(self):
        self._check_chat_grading(from_worker=True, thinking=True)

    def test_request_errors_are_not_accuracy_results(self):
        self._check_chat_grading(request_error=True)

    def test_explicit_chat_template_kwargs_override_thinking_flag(self):
        self._check_chat_grading(thinking=True, explicit_thinking=False)

    def _check_chat_grading(
        self,
        *,
        from_worker=False,
        thinking=False,
        request_error=False,
        explicit_thinking=None,
    ):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                requests.append((self.path, payload))
                if request_error:
                    body = json.dumps(
                        {
                            "error": {
                                "message": "Missing chat template",
                                "type": "BadRequest",
                            }
                        }
                    ).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                prompt = json.dumps(payload["messages"])
                answer = (
                    r"The answer is \boxed{7}. Confidence: 99"
                    if "case-good" in prompt
                    else "No answer."
                )
                body = json.dumps(
                    {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": answer},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "total_tokens": 14,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                dataset = Path(directory) / "examples.jsonl"
                dataset.write_text(
                    "\n".join(
                        json.dumps({"problem": case, "expected_answer": "7"})
                        for case in ("case-good", "case-empty")
                    )
                )
                args = SimpleNamespace(
                    eval_name="gsm8k",
                    model="test-model",
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    from_dataset=str(dataset),
                    sgl_eval_out_dir=directory,
                    num_examples=2,
                    num_threads=2,
                    max_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=20,
                    presence_penalty=1.5,
                    seed=42,
                    min_p=0.1,
                    sgl_eval_thinking=thinking,
                    return_latency=True,
                )
                if explicit_thinking is not None:
                    args.chat_template_kwargs = json.dumps(
                        {
                            "thinking": explicit_thinking,
                            "enable_thinking": explicit_thinking,
                        }
                    )
                if request_error:
                    with self.assertRaisesRegex(RuntimeError, "error_rate=1.0"):
                        run_sgl_eval(args)
                    self.assertTrue(list(Path(directory).glob("*/metrics.json")))
                    return
                if from_worker:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        metrics, latency = executor.submit(run_sgl_eval, args).result(
                            timeout=60
                        )
                else:
                    metrics, latency = run_sgl_eval(args)
                self.assertEqual(metrics["score"], 0.5)
                self.assertEqual(metrics["accuracy"], metrics["score"])
                self.assertEqual(metrics["invalid"], 0.5)
                self.assertEqual(latency, metrics["latency"])
                self.assertAlmostEqual(metrics["output_throughput"] * latency, 8)
                self.assertTrue(Path(metrics["sgl_eval_metrics_path"]).exists())
                saved = json.loads(Path(metrics["sgl_eval_metrics_path"]).read_text())
                self.assertEqual(saved["generation"]["presence_penalty"], 1.5)
                self.assertEqual(saved["generation"]["seed"], 42)
                self.assertTrue(
                    list(Path(directory).glob("sgl_eval_gsm8k_*/output-rs0.jsonl"))
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(requests), 2)
        for path, payload in requests:
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(payload["max_tokens"], 128)
            self.assertEqual(payload["top_k"], 20)
            self.assertEqual(payload["presence_penalty"], 1.5)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["min_p"], 0.1)
            self.assertEqual(payload["temperature"], 0.7)
            self.assertEqual(payload["top_p"], 0.9)
            expected_thinking = (
                thinking if explicit_thinking is None else explicit_thinking
            )
            self.assertEqual(
                payload["chat_template_kwargs"]["enable_thinking"], expected_thinking
            )
            self.assertEqual(
                payload["chat_template_kwargs"]["thinking"], expected_thinking
            )


if __name__ == "__main__":
    unittest.main()
