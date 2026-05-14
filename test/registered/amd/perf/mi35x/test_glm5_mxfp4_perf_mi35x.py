"""MI35x Nightly performance benchmark for GLM-5-MXFP4 model.

Benchmarks the AMD Quark MXFP4-quantized GLM-5 model on MI35x with 8 GPUs.

Model: amd/GLM-5-MXFP4 (MOE-only MXFP4 quantization of zai-org/GLM-5)
Reference: https://huggingface.co/amd/GLM-5-MXFP4

Registry: nightly-perf-8-gpu-mi35x-glm5-mxfp4 suite
"""

import os

os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

import unittest
from typing import List

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, _parse_int_list_env

register_amd_ci(
    est_time=18000,
    suite="nightly-perf-8-gpu-mi35x-glm5-mxfp4",
    nightly=True,
)


def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
    """Generate a simplified markdown report without traces and cost columns.

    Skips the first result if it's a warmup run (duplicate batch_size).
    """
    model_header = results[0].model_path
    if results[0].run_name and results[0].run_name != "default":
        model_header += f" ({results[0].run_name})"

    gpu_config = os.getenv("GPU_CONFIG", "MI35x")
    if gpu_config:
        model_header += f" [{gpu_config}]"

    summary = f"### {model_header}\n"
    summary += "| batch size | input len | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |\n"
    summary += "| ---------- | --------- | ----------- | ------------------------ | ------------------------- | -------- |\n"

    report_results = (
        results[1:]
        if len(results) > 1 and results[0].batch_size == results[1].batch_size
        else results
    )

    for result in report_results:
        itl = (
            1 / (result.output_throughput / result.batch_size) * 1000
            if result.output_throughput > 0
            else 0
        )
        summary += f"| {result.batch_size} | {result.input_len} | {result.latency:.2f} | {result.input_throughput:.2f} | {result.output_throughput:.2f} | {itl:.2f} |\n"

    return summary


GLM5_MXFP4_LOCAL_PATH = "/data2/models/amd-GLM-5-MXFP4"
GLM5_MXFP4_HF_MODEL_ID = "amd/GLM-5-MXFP4"
PROFILE_DIR = "performance_profiles_glm5_mxfp4_mi35x"


def get_model_path() -> str:
    """Get effective model path: env var > local path > HF model ID."""
    env_path = os.environ.get("GLM5_MXFP4_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists(GLM5_MXFP4_LOCAL_PATH):
        return GLM5_MXFP4_LOCAL_PATH
    return GLM5_MXFP4_HF_MODEL_ID


class TestGLM5MXFP4PerfMI35x(unittest.TestCase):
    """MI35x Nightly performance benchmark for GLM-5-MXFP4 model."""

    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path()
        print(f"Using model path: {cls.model}")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "1024"))

        cls.variants = [
            {
                "name": "basic",
                "other_args": [
                    "--trust-remote-code",
                    "--tp",
                    "8",
                    "--chunked-prefill-size",
                    "131072",
                    "--disable-radix-cache",
                    "--mem-fraction-static",
                    "0.85",
                    "--context-length",
                    "4096",
                    "--model-loader-extra-config",
                    '{"enable_multithread_load": true}',
                    "--watchdog-timeout",
                    "1200",
                    "--reasoning-parser",
                    "glm45",
                    "--tool-call-parser",
                    "glm47",
                ],
            },
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()
        cls.runner.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):
        """Run benchmark across all configured variants."""
        failed_variants = []

        is_local_path = self.model.startswith("/")
        if is_local_path and not os.path.exists(self.model):
            print(f"\nSKIPPING: Local model not found at {self.model}")
            self.runner.full_report += (
                f"\nTest skipped: Local model not found at {self.model}\n"
            )
            self.runner.write_final_report()
            return

        if is_local_path:
            print(f"Using local model: {self.model}")
        else:
            print(
                f"Using HuggingFace model: {self.model} (will download if not cached)"
            )

        old_env = {}
        env_vars = {"SGLANG_USE_AITER": "1"}
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            for variant_config in self.variants:
                with self.subTest(variant=variant_config["name"]):
                    result_tuple = self.runner.run_benchmark_for_model(
                        model_path=self.model,
                        batch_sizes=self.batch_sizes,
                        input_lens=self.input_lens,
                        output_lens=self.output_lens,
                        other_args=variant_config["other_args"],
                        variant=variant_config["name"],
                        extra_bench_args=["--trust-remote-code"],
                        enable_profile=False,
                    )
                    results = result_tuple[0]
                    success = result_tuple[1]

                    if not success:
                        failed_variants.append(variant_config["name"])

                    if results:
                        self.runner.full_report += (
                            generate_simple_markdown_report(results) + "\n"
                        )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            self.runner.write_final_report()

        if failed_variants:
            raise AssertionError(
                f"Benchmark failed for {self.model} with the following variants: "
                f"{', '.join(failed_variants)}"
            )


if __name__ == "__main__":
    unittest.main()
