"""Exercise the installed sgl-eval against a local OpenAI-compatible server."""

import json
import runpy
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.sgl_eval import api_base_url, run_sgl_eval
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="stage-a-test-cpu-intel")


class TestSglEvalAdapter(CustomTestCase):
    def test_gateway_mmlu_uses_sgl_eval_prompt_and_grader(self):
        from sgl_eval.registry import get
        from sgl_eval.types import Example, Sample

        benchmark = get("mmlu")
        messages = []

        class Sampler:
            model = "test-model"

            def __init__(self, **kwargs):
                pass

            def __call__(self, prompt, gen):
                messages.append(prompt)
                return Sample(text=r"The answer is \boxed{A}.", finish_reason="stop")

            def abort(self):
                pass

        def evaluate(**kwargs):
            kwargs["load_examples"] = lambda count: [
                Example(
                    id="one",
                    inputs={"problem": "What is 1 + 1?\nA. 2\nB. 3\nC. 4\nD. 5"},
                    target="A",
                )
            ]
            return benchmark.run(**kwargs)

        path = (
            Path(__file__).resolve().parents[4]
            / "sgl-model-gateway/e2e_test/infra/run_eval.py"
        )
        gateway_eval = runpy.run_path(str(path))["run_eval"]
        with (
            patch(
                "sgl_eval.registry.get", return_value=replace(benchmark, run=evaluate)
            ),
            patch("sgl_eval.sampler.ChatCompletionSampler", Sampler),
        ):
            metrics = gateway_eval(
                SimpleNamespace(
                    base_url="http://localhost:30000/v1/",
                    eval_name="mmlu",
                    num_examples=1,
                )
            )
        self.assertEqual(metrics["score"], 1)
        self.assertEqual(messages[0][0]["role"], "user")
        self.assertIn("What is 1 + 1?", messages[0][0]["content"])

    def test_base_urls(self):
        for args in (
            SimpleNamespace(host="127.0.0.1", port=30000),
            SimpleNamespace(host="http://127.0.0.1", port=30000),
            SimpleNamespace(base_url="http://127.0.0.1:30000/v1/"),
        ):
            self.assertEqual(api_base_url(args), "http://127.0.0.1:30000/v1")

    def test_gsm8k_always_dispatches_to_sgl_eval(self):
        for api in (None, "chat", "completion", "generate", "sgl_eval"):
            args = SimpleNamespace(eval_name="gsm8k", api=api)
            with patch(
                "sglang.test.sgl_eval.run_sgl_eval", return_value={"score": 1}
            ) as evaluate:
                self.assertEqual(run_eval(args), {"score": 1})
                evaluate.assert_called_once_with(args)

    def test_real_sgl_eval_chat_grading_and_metrics(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                requests.append((self.path, payload))
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
                    min_p=0.1,
                    chat_template_kwargs={"enable_thinking": False},
                    return_latency=True,
                )
                metrics, latency = run_sgl_eval(args)
                self.assertEqual(metrics["score"], 0.5)
                self.assertEqual(metrics["accuracy"], metrics["score"])
                self.assertEqual(metrics["invalid"], 0.5)
                self.assertEqual(latency, metrics["latency"])
                self.assertAlmostEqual(metrics["output_throughput"] * latency, 8)
                self.assertTrue(Path(metrics["sgl_eval_metrics_path"]).exists())
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
            self.assertEqual(payload["min_p"], 0.1)
            self.assertEqual(payload["temperature"], 0.7)
            self.assertEqual(payload["top_p"], 0.9)
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
