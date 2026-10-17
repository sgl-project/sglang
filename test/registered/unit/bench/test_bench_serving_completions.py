"""OpenAI completions benchmark separates text from response metadata."""

import asyncio
import json
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_completions,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _text(text, **extra):
    return {"choices": [{"index": 0, "text": text}], **extra}


def _usage(completion_tokens=2):
    return {
        "prompt_tokens": 123,
        "completion_tokens": completion_tokens,
        "total_tokens": 123 + completion_tokens,
    }


class _CompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        if body["stream"] and self.server.status == 200:
            payload = (
                b"".join(
                    f"data: {json.dumps(chunk)}\n\n".encode()
                    for chunk in self.server.response
                )
                + b"data: [DONE]\n\n"
            )
            content_type = "text/event-stream"
        else:
            payload = json.dumps(self.server.response).encode()
            content_type = "application/json"
        self.send_response(self.server.status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class TestBenchServingCompletions(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _CompletionHandler)
        self.server.bodies = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _run(
        self,
        response,
        *,
        include_usage=False,
        cache_report=False,
        stream=True,
        status=200,
        success=True,
        text="hello world",
    ):
        self.server.response = response
        self.server.status = status
        self.server.bodies = []
        set_global_args(
            Namespace(
                disable_stream=not stream,
                disable_ignore_eos=True,
                return_logprob=False,
                top_logprobs_num=0,
                header=None,
                cache_report=cache_report,
            )
        )
        extra = {"stream_options": {"include_usage": True}} if include_usage else {}
        if cache_report:
            extra["return_cached_tokens_details"] = True
        request = RequestFuncInput(
            prompt="hello",
            api_url=f"http://127.0.0.1:{self.server.server_port}/v1/completions",
            prompt_len=9999,
            output_len=8,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body=extra,
        )
        output = asyncio.run(async_request_openai_completions(request))
        self.assertEqual(output.success, success, output.error)
        if success:
            self.assertEqual(output.generated_text, text)
        else:
            self.assertTrue(output.error)
        self.assertEqual(self.server.bodies[0]["stream"], stream)
        self.assertEqual(self.server.bodies[0]["max_tokens"], 8)
        return output

    def test_final_usage_only_chunk(self):
        for tokens, chunks, text in (
            (2, [_text("hello"), _text(" world")], "hello world"),
            (0, [], ""),
        ):
            with self.subTest(completion_tokens=tokens):
                output = self._run(
                    chunks + [{"choices": [], "usage": _usage(tokens)}],
                    include_usage=True,
                    text=text,
                )
                self.assertEqual(output.prompt_len, 123)
                self.assertEqual(output.output_len, tokens)
                self.assertEqual(
                    self.server.bodies[0]["stream_options"], {"include_usage": True}
                )

    def test_text_chunks_without_usage_keep_fallback_counts(self):
        output = self._run([_text("hello"), _text(" world")])
        self.assertEqual(output.prompt_len, 9999)
        self.assertEqual(output.output_len, 8)
        self.assertNotIn("stream_options", self.server.bodies[0])

    def test_cache_only_chunk(self):
        details = {"device": 2, "host": 3, "storage": 5}
        output = self._run(
            [
                _text("hello"),
                _text(" world"),
                {"choices": [], "sglext": {"cached_tokens_details": details}},
            ],
            cache_report=True,
        )
        self.assertEqual(output.cached_tokens, 10)
        self.assertEqual(output.cached_tokens_details, details)
        self.assertEqual(output.output_len, 8)
        self.assertTrue(self.server.bodies[0]["return_cached_tokens_details"])

    def test_empty_text_final_usage(self):
        output = self._run(
            [_text("hello"), _text(" world"), _text("", usage=_usage())],
            include_usage=True,
        )
        self.assertEqual(output.prompt_len, 123)
        self.assertEqual(output.output_len, 2)
        self.assertEqual(output.text_chunks, [" world"])
        self.assertEqual(len(output.itl), 1)

    def test_null_or_missing_usage_keeps_reported_count(self):
        for final in (_text("", usage=None), _text("")):
            with self.subTest(final=final):
                output = self._run(
                    [_text("hello", usage=None), _text(" world", usage=_usage()), final]
                )
                self.assertEqual(output.prompt_len, 123)
                self.assertEqual(output.output_len, 2)

    def test_metadata_between_text_chunks_does_not_add_token_events(self):
        output = self._run(
            [
                _text("hello"),
                {"choices": [], "usage": _usage()},
                _text(" world"),
                _text(""),
            ],
            include_usage=True,
        )
        self.assertEqual(output.output_len, 2)
        self.assertEqual(output.prompt_len, 123)
        self.assertEqual(output.text_chunks, [" world"])
        self.assertEqual(len(output.itl), 1)
        self.assertGreater(output.ttft, 0)

    def test_non_streaming_json(self):
        output = self._run(_text("hello world", usage=_usage()), stream=False)
        self.assertEqual(output.prompt_len, 123)
        self.assertEqual(output.output_len, 2)

    def test_error_responses_remain_failures(self):
        error = {
            "error": {
                "message": "fixture request rejected",
                "type": "invalid_request_error",
            }
        }
        for status in (200, 400):
            with self.subTest(status=status):
                self._run(
                    [error] if status == 200 else error,
                    status=status,
                    success=False,
                )


if __name__ == "__main__":
    unittest.main()
