"""
Usage:
python -m unittest test_flashinfer_cutedsl_standard_path_parity.TestFlashinferCuteDslStandardPathParity.test_quality_and_throughput_parity
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestFlashinferCuteDslStandardPathParity(CustomTestCase):
    MODEL = DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4
    BASE_URL = DEFAULT_URL_FOR_TEST
    SCORE_DELTA_THRESHOLD = 0.10

    BACKEND_CONFIGS = {
        "flashinfer_cutlass": [
            "--trust-remote-code",
            "--moe-runner-backend",
            "flashinfer_cutlass",
            "--quantization",
            "modelopt_fp4",
            "--attention-backend",
            "torch_native",
            "--sampling-backend",
            "pytorch",
            "--disable-cuda-graph",
        ],
        "flashinfer_trtllm": [
            "--trust-remote-code",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--quantization",
            "modelopt_fp4",
            "--attention-backend",
            "torch_native",
            "--sampling-backend",
            "pytorch",
            "--disable-cuda-graph",
        ],
        "flashinfer_cutedsl": [
            "--trust-remote-code",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--moe-a2a-backend",
            "none",
            "--quantization",
            "modelopt_fp4",
            "--attention-backend",
            "torch_native",
            "--sampling-backend",
            "pytorch",
            "--disable-cuda-graph",
        ],
    }

    def _run_small_mmlu(self):
        args = SimpleNamespace(
            base_url=self.BASE_URL,
            model=self.MODEL,
            eval_name="mmlu",
            num_examples=32,
            num_threads=4,
        )
        return run_eval(args)

    def _run_small_bench_serving(self):
        with tempfile.NamedTemporaryFile(
            prefix="moe_backend_bench_", suffix=".jsonl", delete=False
        ) as tf:
            output_file = tf.name
        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--base-url",
            self.BASE_URL,
            "--dataset-name",
            "random",
            "--num-prompts",
            "48",
            "--random-input-len",
            "128",
            "--random-output-len",
            "64",
            "--random-range-ratio",
            "0.0",
            "--max-concurrency",
            "8",
            "--disable-tqdm",
            "--output-file",
            output_file,
        ]
        try:
            subprocess.run(cmd, check=True)
            with open(output_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) > 0
            return json.loads(lines[-1])
        finally:
            try:
                os.remove(output_file)
            except OSError:
                pass

    def _run_backend(self, backend_name: str):
        env = dict(os.environ)
        process = popen_launch_server(
            self.MODEL,
            self.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.BACKEND_CONFIGS[backend_name],
            env=env,
        )
        try:
            eval_metrics = self._run_small_mmlu()
            bench_metrics = self._run_small_bench_serving()
            return {
                "score": float(eval_metrics["score"]),
                "output_throughput": float(bench_metrics["output_throughput"]),
                "mean_tpot_ms": float(bench_metrics["mean_tpot_ms"]),
                "mean_ttft_ms": float(bench_metrics["mean_ttft_ms"]),
            }
        finally:
            kill_process_tree(process.pid)

    def test_quality_and_throughput_parity(self):
        results = {
            backend: self._run_backend(backend)
            for backend in self.BACKEND_CONFIGS.keys()
        }
        print(f"{results=}")

        cutlass_score = results["flashinfer_cutlass"]["score"]
        trtllm_score = results["flashinfer_trtllm"]["score"]
        cutedsl_score = results["flashinfer_cutedsl"]["score"]

        self.assertLessEqual(
            abs(cutedsl_score - cutlass_score), self.SCORE_DELTA_THRESHOLD
        )
        self.assertLessEqual(
            abs(cutedsl_score - trtllm_score), self.SCORE_DELTA_THRESHOLD
        )

        # Basic sanity: benchmark run completed and produced finite numbers.
        for backend, metrics in results.items():
            self.assertGreater(metrics["output_throughput"], 0.0, backend)
            self.assertGreater(metrics["mean_ttft_ms"], 0.0, backend)
            self.assertGreater(metrics["mean_tpot_ms"], 0.0, backend)


if __name__ == "__main__":
    unittest.main()
