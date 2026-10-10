"""Test benchmark result serialization without a server or model weights."""

import asyncio
import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBenchServingOutputDetails(CustomTestCase):
    def _run_benchmark(self, output_details):
        outputs = {
            "slow": serving.RequestFuncOutput(
                success=True,
                generated_text="one two three",
                prompt_len=11,
                output_len=3,
                latency=1.23456789,
                ttft=0.125,
                itl=[0.4, 0.5],
            ),
            "failed": serving.RequestFuncOutput(
                prompt_len=22, latency=0.375, error="request failed"
            ),
            "single": serving.RequestFuncOutput(
                success=True,
                generated_text="one",
                prompt_len=33,
                output_len=1,
                latency=0.25,
                ttft=0.25,
            ),
        }
        requests = [
            DatasetRow(prompt=prompt, prompt_len=out.prompt_len, output_len=3)
            for prompt, out in outputs.items()
        ]
        completion_order = []

        async def run():
            release_slow = asyncio.Event()

            async def request_func(request_func_input, pbar):
                prompt = request_func_input.prompt
                if prompt == "slow":
                    await release_slow.wait()
                elif prompt == "failed":
                    release_slow.set()
                completion_order.append(prompt)
                return outputs[prompt]

            with mock.patch.dict(serving.ASYNC_REQUEST_FUNCS, vllm=request_func):
                return await serving.benchmark(
                    backend="vllm",
                    api_url="http://unused/v1/completions",
                    base_url="http://unused",
                    model_id="test-model",
                    tokenizer=mock.Mock(encode=lambda text, **kwargs: text.split()),
                    input_requests=requests,
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution="uniform",
                    lora_zipf_alpha=1.0,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "results.jsonl"
            args = Namespace(
                backend="vllm",
                dataset_name="random",
                num_prompts=len(requests),
                warmup_requests=0,
                plot_throughput=False,
                cache_report=False,
                sharegpt_output_len=None,
                random_input_len=32,
                random_output_len=3,
                random_range_ratio=1.0,
                output_file=str(output_file),
                output_details=output_details,
            )
            with (
                mock.patch.object(serving, "args", args, create=True),
                mock.patch.object(
                    serving.requests, "get", return_value=mock.Mock(status_code=404)
                ),
                redirect_stdout(io.StringIO()),
            ):
                result = asyncio.run(run())
            lines = output_file.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            saved = json.loads(lines[0])

        self.assertEqual(completion_order, ["failed", "single", "slow"])
        return result, saved

    def test_e2e_latencies_preserve_seconds_and_request_order(self):
        result, saved = self._run_benchmark(output_details=True)
        expected = [1.23456789, 0.375, 0.25]
        self.assertEqual(result["e2e_latencies"], expected)
        self.assertEqual(saved["e2e_latencies"], expected)
        self.assertEqual(saved["input_lens"], [11, 22, 33])
        self.assertEqual(saved["output_lens"], [3, 0, 1])
        self.assertEqual(saved["ttfts"], [0.125, 0.0, 0.25])
        self.assertEqual(saved["errors"], ["", "request failed", ""])
        self.assertEqual(saved["generated_texts"], ["one two three", "", "one"])
        # Raw details retain the failed request; aggregate metrics exclude it.
        self.assertEqual(saved["completed"], 2)
        self.assertAlmostEqual(
            saved["mean_e2e_latency_ms"], (expected[0] + expected[2]) / 2 * 1000
        )

    def test_e2e_latencies_are_not_saved_without_output_details(self):
        result, saved = self._run_benchmark(output_details=False)
        # The programmatic return value follows the existing details contract.
        self.assertEqual(result["e2e_latencies"], [1.23456789, 0.375, 0.25])
        for field in ("e2e_latencies", "ttfts", "itls", "errors"):
            self.assertNotIn(field, saved)
        self.assertIn("mean_e2e_latency_ms", saved)


if __name__ == "__main__":
    unittest.main()
