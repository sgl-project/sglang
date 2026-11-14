"""HTTP protocol coverage for benchmark per-request diagnostics."""

import asyncio
import contextlib
import io
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ResponseHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        status, payload, content_type = self.server.responses.pop(0)
        if status is None:
            self.close_connection = True
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def _response(chunks, stream=True):
    if stream:
        payload = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
        return 200, payload + b"data: [DONE]\n\n", "text/event-stream"
    return 200, json.dumps(chunks[0]).encode(), "application/json"


class TestBenchServingDiagnostics(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _ResponseHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.options = Namespace(
            disable_stream=False,
            disable_ignore_eos=True,
            return_logprob=False,
            top_logprobs_num=0,
            return_routed_experts=False,
            logprob_start_len=-1,
            token_ids_logprob=None,
            temperature=0.0,
            top_p=1.0,
            header=None,
            cache_report=False,
        )
        self.args_patch = patch.object(serving, "args", self.options, create=True)
        self.args_patch.start()
        self.addCleanup(self.args_patch.stop)

    def _request(self, endpoint):
        return serving.RequestFuncInput(
            prompt="hello",
            api_url=self.base_url + endpoint,
            prompt_len=1,
            output_len=1,
            model="test",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )

    def test_terminal_reason_survives_empty_and_usage_only_chunks(self):
        """Terminal chunks can carry no text, followed by usage with choices=[]."""
        usage = {"choices": [], "usage": {"completion_tokens": 1}}
        completion = [
            {"choices": [{"text": "hi", "finish_reason": None}]},
            {"choices": [{"text": "", "finish_reason": "length"}]},
            usage,
        ]
        chat = [
            {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            usage,
        ]
        native = [
            {"text": "hi", "meta_info": {"completion_tokens": 1}},
            {"meta_info": {"finish_reason": {"type": "length", "length": 1}}},
        ]
        cases = [
            (
                serving.async_request_openai_completions,
                "/v1/completions",
                completion,
                True,
                "length",
            ),
            (serving.async_request_truss, "/predict", completion, True, "length"),
            (
                serving.async_request_openai_chat_completions,
                "/v1/chat/completions",
                chat,
                True,
                "stop",
            ),
            (
                serving.async_request_sglang_generate,
                "/generate",
                native,
                True,
                "length",
            ),
            (
                serving.async_request_openai_completions,
                "/v1/completions",
                [{"choices": [{"text": "hi", "finish_reason": "stop"}]}],
                False,
                "stop",
            ),
            (
                serving.async_request_openai_chat_completions,
                "/v1/chat/completions",
                [
                    {
                        "choices": [
                            {
                                "message": {"content": "hi"},
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                ],
                False,
                "tool_calls",
            ),
            (
                serving.async_request_sglang_generate,
                "/generate",
                [
                    {
                        "text": "hi",
                        "meta_info": {
                            "completion_tokens": 1,
                            "finish_reason": {"type": "stop", "matched": 2},
                        },
                    }
                ],
                False,
                "stop",
            ),
        ]
        for func, endpoint, chunks, stream, reason in cases:
            with self.subTest(backend=func.__name__, stream=stream):
                self.options.disable_stream = not stream
                self.server.responses = [_response(chunks, stream)]
                output = asyncio.run(func(self._request(endpoint)))
                self.assertTrue(output.success, output.error)
                self.assertEqual(output.generated_text, "hi")
                self.assertEqual(output.status_code, 200)
                self.assertEqual(output.finish_reason, reason)

    def test_http_status_is_retained_on_failures(self):
        """HTTP errors, response parsing errors, and transport errors stay distinct."""
        adapters = [
            (serving.async_request_trt_llm, "/generate_stream"),
            (serving.async_request_openai_completions, "/v1/completions"),
            (serving.async_request_openai_chat_completions, "/v1/chat/completions"),
            (serving.async_request_truss, "/predict"),
            (serving.async_request_sglang_generate, "/generate"),
            (serving.async_request_openai_embeddings, "/v1/embeddings"),
        ]
        for func, endpoint in adapters:
            for status in (429, 503, 200, None):
                with self.subTest(backend=func.__name__, status=status):
                    self.server.responses = [
                        (status, b"invalid response", "application/json")
                    ]
                    output = asyncio.run(func(self._request(endpoint)))
                    self.assertFalse(output.success)
                    self.assertTrue(output.error)
                    self.assertEqual(output.status_code, status)
                    self.assertIsNone(output.finish_reason)

    def test_output_details_keep_diagnostics_in_request_order(self):
        """JSONL diagnostics must include failed requests in the same order as errors."""
        self.options.dataset_name = "custom"
        self.options.backend = "vllm-chat"
        self.options.num_prompts = 2
        self.options.warmup_requests = 0
        self.options.plot_throughput = False
        self.options.goodput = None
        self.options.output_details = True
        self.options.sharegpt_output_len = None
        self.options.random_input_len = 1
        self.options.random_output_len = 1
        self.options.random_range_ratio = 1.0
        self.server.responses = [
            _response(
                [
                    {
                        "choices": [
                            {"delta": {"content": "hi"}, "finish_reason": "length"}
                        ]
                    },
                    {"choices": [], "usage": {"completion_tokens": 1}},
                ]
            ),
            (429, b"rate limited", "text/plain"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            self.options.output_file = str(Path(directory) / "results.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                result = asyncio.run(
                    serving.benchmark(
                        backend="vllm-chat",
                        api_url=self.base_url + "/v1/chat/completions",
                        base_url=self.base_url,
                        model_id="test",
                        tokenizer=Namespace(
                            encode=lambda text, **kwargs: [1] if text else []
                        ),
                        input_requests=[
                            DatasetRow("hello", 1, 1),
                            DatasetRow("world", 1, 1),
                        ],
                        request_rate=float("inf"),
                        max_concurrency=1,
                        disable_tqdm=True,
                        lora_names=[],
                        lora_request_distribution=None,
                        lora_zipf_alpha=None,
                        extra_request_body={},
                        profile=False,
                        warmup_requests=0,
                    )
                )
            saved = json.loads(Path(self.options.output_file).read_text())
        for details in (result, saved):
            self.assertEqual(details["status_codes"], [200, 429])
            self.assertEqual(details["finish_reasons"], ["length", None])
            self.assertEqual(details["errors"][0], "")
            self.assertIn("rate limited", details["errors"][1])


if __name__ == "__main__":
    unittest.main()
