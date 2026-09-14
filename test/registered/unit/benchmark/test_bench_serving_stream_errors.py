"""An HTTP 200 SSE response can still end in a server-side error."""

import json
import unittest
from argparse import Namespace
from unittest.mock import patch

from aiohttp import web

import sglang.benchmark.serving as serving
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
    async_request_sglang_generate,
    calculate_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestBenchServingStreamErrors(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        args_patch = patch.object(
            serving,
            "args",
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                return_logprob=False,
                return_routed_experts=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                temperature=0.0,
                top_p=1.0,
                header=None,
                tokenizer="",
                print_requests=False,
            ),
            create=True,
        )
        args_patch.start()
        self.addCleanup(args_patch.stop)

    async def _run(self, request_func, chunks):
        async def handle(request):
            await request.json()
            body = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
            )
            return web.Response(
                body=body + b"data: [DONE]\n\n", content_type="text/event-stream"
            )

        path = {
            async_request_openai_chat_completions: "/v1/chat/completions",
            async_request_openai_completions: "/v1/completions",
            async_request_sglang_generate: "/generate",
        }[request_func]
        app = web.Application()
        app.router.add_post(path, handle)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            request = RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}{path}",
                prompt_len=1,
                output_len=64,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body={},
            )
            return await request_func(request)
        finally:
            await runner.cleanup()

    @staticmethod
    def _backends():
        return (
            (
                async_request_openai_chat_completions,
                {
                    "choices": [{"delta": {"content": "partial answer"}}],
                    "usage": {"completion_tokens": 2},
                },
            ),
            (
                async_request_openai_completions,
                {
                    "choices": [{"text": "partial answer"}],
                    "usage": {"completion_tokens": 2},
                },
            ),
            (
                async_request_sglang_generate,
                {
                    "text": "partial answer",
                    "meta_info": {"completion_tokens": 2},
                },
            ),
        )

    async def test_stream_error_before_or_after_content_fails_request(self):
        error = {"message": "Generation timed out", "code": 503}
        for request_func, content in self._backends():
            for prefix in ([], [content]):
                with self.subTest(backend=request_func.__name__, prefix=bool(prefix)):
                    output = await self._run(request_func, prefix + [{"error": error}])
                    self.assertFalse(output.success)
                    self.assertIn("Generation timed out", output.error)
                    self.assertIn("503", output.error)
                    self.assertEqual(output.output_len, 0)

    async def test_string_error_preserves_server_message(self):
        for request_func, _ in self._backends():
            with self.subTest(backend=request_func.__name__):
                output = await self._run(
                    request_func, [{"error": "Engine unavailable"}]
                )
                self.assertFalse(output.success)
                self.assertIn("Engine unavailable", output.error)

    async def test_stream_errors_do_not_count_toward_success_metrics(self):
        for request_func, content in self._backends():
            with self.subTest(backend=request_func.__name__):
                success = await self._run(request_func, [content])
                failed = await self._run(
                    request_func,
                    [content, {"error": {"message": "Generation timed out"}}],
                )
                self.assertTrue(success.success, success.error)
                self.assertEqual(success.generated_text, "partial answer")
                self.assertEqual(success.output_len, 2)
                metrics, output_lens = calculate_metrics(
                    input_requests=[
                        DatasetRow(prompt="hello", prompt_len=1, output_len=64),
                        DatasetRow(prompt="hello", prompt_len=1, output_len=64),
                    ],
                    outputs=[success, failed],
                    dur_s=1.0,
                    tokenizer=_Tokenizer(),
                    backend="sglang",
                )
                self.assertEqual(metrics.completed, 1)
                self.assertEqual(metrics.total_input, 1)
                self.assertEqual(metrics.total_output, 2)
                self.assertEqual(metrics.output_throughput, 2.0)
                self.assertAlmostEqual(metrics.mean_ttft_ms, success.ttft * 1000)
                self.assertAlmostEqual(
                    metrics.mean_e2e_latency_ms, success.latency * 1000
                )
                self.assertEqual(output_lens, [2, 0])

    async def test_chat_usage_only_chunk_remains_successful(self):
        output = await self._run(
            async_request_openai_chat_completions,
            [
                {"choices": [{"delta": {"content": "answer"}}]},
                {"choices": [], "usage": {"completion_tokens": 1}},
            ],
        )
        self.assertTrue(output.success, output.error)
        self.assertEqual(output.generated_text, "answer")
        self.assertEqual(output.output_len, 1)


if __name__ == "__main__":
    unittest.main()
