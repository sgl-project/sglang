"""Shared localhost HTTP fixture for serving benchmark tests."""

import asyncio
import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_completions,
    async_request_sglang_generate,
    set_global_args,
)
from sglang.test.test_utils import CustomTestCase

# The dataset row's value; never what the stub reports.
ROW_PROMPT_LEN = 9999


class _ServingHandler(BaseHTTPRequestHandler):
    """Serve native events or chat responses with opt-in streaming usage."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        usage = {
            "completion_tokens": 1,
            "prompt_tokens": self.server.prompt_tokens[len(self.server.bodies) - 1],
        }
        if body.get("stream"):
            if self.path == "/generate":
                chunks = self.server.native_chunks
            else:
                choice = (
                    {"index": 0, "text": "hi"}
                    if self.path == "/v1/completions"
                    else {"index": 0, "delta": {"content": "hi"}}
                )
                chunks = [{"choices": [choice]}]
                if (body.get("stream_options") or {}).get(
                    "include_usage"
                ) and not self.server.omit_usage:
                    chunks.append({"choices": [], "usage": usage})
            content_type = "text/event-stream"
            payload = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
            payload += b"data: [DONE]\n\n"
        else:
            message = {"role": "assistant", "content": "hi"}
            choice = {"index": 0, "message": message, "finish_reason": "length"}
            content_type = "application/json"
            payload = json.dumps({"choices": [choice], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class _BenchServingHTTPTestCase(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _ServingHandler)
        self.server.omit_usage = False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _run(
        self, request_func, prompt, prompt_tokens, stream=True, extra=None, success=True
    ):
        self.server.bodies = []
        self.server.prompt_tokens = prompt_tokens
        set_global_args(
            Namespace(
                disable_stream=not stream,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
                temperature=0.0,
                top_p=1.0,
                return_logprob=False,
                return_routed_experts=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
            )
        )
        endpoint = {
            async_request_sglang_generate: "/generate",
            async_request_openai_completions: "/v1/completions",
        }.get(request_func, "/v1/chat/completions")
        request = RequestFuncInput(
            prompt=prompt,
            api_url=f"http://127.0.0.1:{self.server.server_port}{endpoint}",
            prompt_len=ROW_PROMPT_LEN,
            output_len=8,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body=extra or {},
        )
        out = asyncio.run(request_func(request))
        outputs = out if isinstance(out, list) else [out]
        for o in outputs:
            self.assertEqual(o.success, success, o.error)
        return outputs
