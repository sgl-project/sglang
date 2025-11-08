import os
import subprocess
import time
import unittest

from sglang.bench_one_batch_server import BenchmarkResult, generate_markdown_report
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

PROFILE_DIR = "performance_profiles_gpt_oss_4gpu"


class TestNightlyGptOss4GpuPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = [
            ("lmsys/gpt-oss-120b-bf16", ["--tp", "4", "--cuda-graph-max-bs", "200"]),
            (
                "openai/gpt-oss-120b",
                [
                    "--tp",
                    "4",
                    "--cuda-graph-max-bs",
                    "200",
                    "--mem-fraction-static",
                    "0.93",
                ],
            ),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = (4096,)
        cls.output_lens = (512,)
        os.makedirs(PROFILE_DIR, exist_ok=True)
        cls.full_report = f"## {cls.__name__}\n" + BenchmarkResult.help_str()

    def test_bench_one_batch(self):
        all_benchmark_results = []
        all_model_succeed = True
        for model_path, other_args in self.models:
            benchmark_results = []
            with self.subTest(model=model_path):
                process = popen_launch_server(
                    model=model_path,
                    base_url=self.base_url,
                    other_args=other_args,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
                try:

                    profile_filename = (
                        f"{model_path.replace('/', '_')}_{int(time.time())}"
                    )
                    profile_path_prefix = os.path.join(PROFILE_DIR, profile_filename)
                    json_output_file = f"results_{model_path.replace('/', '_')}_{int(time.time())}.json"

                    command = [
                        "python3",
                        "-m",
                        "sglang.bench_one_batch_server",
                        "--model",
                        model_path,
                        "--base-url",
                        self.base_url,
                        "--batch-size",
                        *[str(x) for x in self.batch_sizes],
                        "--input-len",
                        *[str(x) for x in self.input_lens],
                        "--output-len",
                        *[str(x) for x in self.output_lens],
                        "--show-report",
                        "--profile",
                        "--profile-by-stage",
                        "--profile-filename-prefix",
                        profile_path_prefix,
                        f"--output-path={json_output_file}",
                        "--no-append-to-github-summary",
                    ]

                    print(f"Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(
                            f"Error running benchmark for {model_path} with batch size:"
                        )
                        print(result.stderr)
                        all_model_succeed = False
                        continue

                    # Load and deserialize JSON results
                    if os.path.exists(json_output_file):
                        import json

                        with open(json_output_file, "r") as f:
                            json_data = json.load(f)

                        # Convert JSON data to BenchmarkResult objects
                        for data in json_data:
                            benchmark_result = BenchmarkResult(**data)
                            all_benchmark_results.append(benchmark_result)
                            benchmark_results.append(benchmark_result)

                        print(
                            f"Loaded {len(benchmark_results)} benchmark results from {json_output_file}"
                        )

                        # Clean up JSON file
                        os.remove(json_output_file)
                    else:
                        all_model_succeed = False
                        print(f"Warning: JSON output file {json_output_file} not found")

                finally:
                    kill_process_tree(process.pid)

                report_part = generate_markdown_report(PROFILE_DIR, benchmark_results)
                self.full_report += report_part + "\n"

        if is_in_ci():
            write_github_step_summary(self.full_report)

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
