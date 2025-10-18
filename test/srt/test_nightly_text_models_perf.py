import os
import subprocess
import time
import unittest

from sglang.bench_one_batch_server import BenchmarkResult, generate_markdown_report
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
)

PROFILE_DIR = "performance_profiles_text_models"


class TestNightlyTextModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = []
        # TODO: replace with DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1 or other model lists
        for model_path in parse_models("meta-llama/Llama-3.1-8B-Instruct"):
            cls.models.append(ModelLaunchSettings(model_path, tp_size=1))
        for model_path in parse_models("Qwen/Qwen2-57B-A14B-Instruct"):
            cls.models.append(ModelLaunchSettings(model_path, tp_size=2))
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
        # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))
        os.makedirs(PROFILE_DIR, exist_ok=True)
        cls.full_report = f"## {cls.__name__}\n" + BenchmarkResult.help_str()

    def test_bench_one_batch(self):
        all_benchmark_results = []
        all_model_succeed = True
        for model_setup in self.models:
            benchmark_results = []
            with self.subTest(model=model_setup.model_path):
                process = popen_launch_server(
                    model=model_setup.model_path,
                    base_url=self.base_url,
                    other_args=model_setup.extra_args,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                )
                try:

                    profile_filename = (
                        f"{model_setup.model_path.replace('/', '_')}_{int(time.time())}"
                    )
                    profile_path_prefix = os.path.join(PROFILE_DIR, profile_filename)
                    json_output_file = f"results_{model_setup.model_path.replace('/', '_')}_{int(time.time())}.json"

                    command = [
                        "python3",
                        "-m",
                        "sglang.bench_one_batch_server",
                        "--model",
                        model_setup.model_path,
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
                            f"Error running benchmark for {model_setup.model_path} with batch size:"
                        )
                        print(result.stderr)
                        # Continue to next batch size even if one fails
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
