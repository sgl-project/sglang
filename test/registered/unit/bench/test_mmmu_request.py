"""Test MMMU request bodies with the real OpenAI SDK and offline transport."""

import asyncio
import base64
import copy
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx
from openai import AsyncOpenAI
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load_mmmu_benchmark():
    root = Path(__file__).resolve().parents[4]
    benchmark_dir = root / "benchmark" / "mmmu"
    spec = importlib.util.spec_from_file_location(
        "mmmu_bench_sglang", benchmark_dir / "bench_sglang.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Only isolate the script aliases. Unloading real dependencies such as
    # PyArrow can make subsequent dataset imports fail in the same process.
    aliases = (spec.name, "data_utils", "eval_utils")
    previous = {name: sys.modules[name] for name in aliases if name in sys.modules}
    with mock.patch.object(sys, "path", [str(benchmark_dir), *sys.path]):
        try:
            for name in aliases:
                sys.modules.pop(name, None)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        finally:
            for name in aliases:
                sys.modules.pop(name, None)
            sys.modules.update(previous)
    return module


class TestMMMURequestBody(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_mmmu_benchmark()

    def _request_body(
        self,
        sampling_params,
        expected_headers=None,
        expected_query=None,
        expected_timeout=None,
        **kwargs,
    ):
        bodies = []

        def respond(request):
            for key, value in (expected_headers or {}).items():
                self.assertEqual(request.headers[key], value)
            for key, value in (expected_query or {}).items():
                self.assertEqual(request.url.params[key], value)
            if expected_timeout is not None:
                self.assertEqual(
                    request.extensions["timeout"]["read"], expected_timeout
                )
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "test-mmmu",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-vlm",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        async def run():
            async with AsyncOpenAI(
                api_key="test-key",
                base_url="http://test.invalid/v1",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as client:
                image = Image.new("RGB", (1, 1))
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_data = base64.b64encode(image_buffer.getvalue()).decode()
                sample = {
                    "final_input_prompt": "Question <image> Answer:",
                    "image": image,
                    "image_path": f"data:image/png;base64,{image_data}",
                }
                result_sample, answer = await self.benchmark.process_sample(
                    client, sample, sampling_params, "test-vlm", **kwargs
                )
                self.assertIs(result_sample, sample)
                self.assertEqual(answer, "A")

        original = copy.deepcopy(sampling_params)
        asyncio.run(run())
        self.assertEqual(sampling_params, original)
        self.assertEqual(len(bodies), 1)
        self.assertNotIn("extra_body", bodies[0])
        return bodies[0]

    def test_default_parameters(self):
        args = self.benchmark.EvalArgs()
        body = self._request_body(self.benchmark.get_sampling_params(args))
        self.assertEqual(body["model"], "test-vlm")
        self.assertEqual(len(body["messages"][0]["content"]), 3)
        self.assertNotIn("lora_path", body)
        self.assertNotIn("top_k", body)

    def test_explicit_cli_sampling_options_override_json(self):
        args = self.benchmark.EvalArgs(
            extra_request_body='{"temperature":0.8,"max_completion_tokens":40,"top_p":0.5}',
            temperature=0.2,
            max_new_tokens=7,
        )
        params = self.benchmark.get_sampling_params(args)
        self.assertNotIn("extra_body", params)  # Also consumed by bench_hf.py.
        body = self._request_body(params)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["max_completion_tokens"], 7)
        self.assertEqual(body["top_p"], 0.5)

    def test_sglang_sampling_extensions(self):
        args = self.benchmark.EvalArgs(
            extra_request_body='{"top_k":5,"min_p":0.1,"ignore_eos":true}'
        )
        body = self._request_body(self.benchmark.get_sampling_params(args))
        self.assertEqual(body["top_k"], 5)
        self.assertEqual(body["min_p"], 0.1)
        self.assertIs(body["ignore_eos"], True)

    def test_lora_and_sampling_extensions_coexist(self):
        body = self._request_body({"top_k": 3}, lora_path="test-adapter")
        self.assertEqual(body["lora_path"], "test-adapter")
        self.assertEqual(body["top_k"], 3)

    def test_explicit_reasoning_effort_overrides_json(self):
        body = self._request_body({"reasoning_effort": "low"}, reasoning_effort="high")
        self.assertEqual(body["reasoning_effort"], "high")

    def test_nested_sdk_extra_body_remains_supported(self):
        body = self._request_body({"extra_body": {"top_k": 3}, "top_p": 0.8})
        self.assertEqual(body["top_k"], 3)
        self.assertEqual(body["top_p"], 0.8)

    def test_nested_body_keeps_precedence_over_named_options(self):
        body = self._request_body(
            {
                "temperature": 0.2,
                "reasoning_effort": "low",
                "extra_body": {
                    "temperature": 0.8,
                    "reasoning_effort": "medium",
                    "lora_path": "nested-adapter",
                },
            },
            reasoning_effort="high",
            lora_path="cli-adapter",
        )
        self.assertEqual(body["temperature"], 0.8)
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["lora_path"], "nested-adapter")

    def test_sdk_request_controls_remain_outside_json_body(self):
        body = self._request_body(
            {
                "extra_headers": {"X-Test-Header": "header-value"},
                "extra_query": {"test-query": "query-value"},
                "timeout": 10.0,
            },
            expected_headers={"X-Test-Header": "header-value"},
            expected_query={"test-query": "query-value"},
            expected_timeout=10.0,
        )
        for key in ("extra_headers", "extra_query", "timeout"):
            self.assertNotIn(key, body)


if __name__ == "__main__":
    unittest.main()
