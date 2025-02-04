"""
Usage:
python3 -m unittest test_hip_attention_backend.TestHiPAttnBackend.test_mmlu
"""

import os
import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


class TestHiPAttnBackend(unittest.TestCase):
    def _measure_latency(self, extra_args):
        output_throughput = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                # "--input",
                # "32000",
                "--enable-hip-attention",
                "--cuda-graph-max-bs",
                "1",
                *extra_args,
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 40)

    def _measure_mmlu(self, extra_args):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hip-attention",
                "--cuda-graph-max-bs",
                "1",
                *extra_args,
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)

            self.assertGreaterEqual(metrics["score"], 0.65)
        finally:
            kill_process_tree(process.pid)

    def _run_passkey(self, extra_args):
        target_length = int(os.getenv("SRT_TEST_PASSKEY_PROMPT_LENGTH", "35000"))
        correct_answer = "$000310$"
        query_string = "You need to find the passkey. Read the following text carefully and remember the passkey.\n\n"
        filler = "Sky is blue, grass is green, sun is red. And here we go again. "
        query_string += filler * (target_length // 35)
        query_string += f"\n\nThe passkey is {correct_answer}. Remember, the passkey is {correct_answer}.\n\n"
        query_string += f"\n\nThe passkey is {correct_answer}. Remember, the passkey is {correct_answer}.\n\n"
        query_string += f"\n\nThe passkey is {correct_answer}. Remember, the passkey is {correct_answer}.\n\n"
        query_string += filler * (target_length // 35)
        query_string += "What was the passkey? The passkey is"

        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hip-attention",
                "--cuda-graph-max-bs",
                "1",
                "--context-length",
                f"{target_length + 10000}",
                *extra_args,
            ],
        )

        try:
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = "EMPTY"

            sampler = ChatCompletionSampler(
                model=model,
                max_tokens=16,
                base_url=f"{base_url}/v1",
                temperature=0.0,
            )

            # Run eval
            tic = time.time()
            result = sampler([{"role": "user", "content": query_string}])
            latency = time.time() - tic

            # Print results
            print("Result:", result)
            print(f"Total latency: {latency:.3f} s")

            self.assertIn(correct_answer, result)
        finally:
            kill_process_tree(process.pid)

    def test_latency(self):
        self._measure_latency([])

    def test_latency_offload(self):
        self._measure_latency(["--enable-hip-offload"])

    def test_mmlu(self):
        self._measure_mmlu([])

    def test_mmlu_offload(self):
        self._measure_mmlu(["--enable-hip-offload"])

    def test_passkey(self):
        self._run_passkey([])

    def test_passkey_offload(self):
        self._run_passkey(["--enable-hip-offload"])


if __name__ == "__main__":
    unittest.main()
