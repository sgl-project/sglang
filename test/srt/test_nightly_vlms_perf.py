import os
import subprocess
import unittest
import warnings

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

PROFILE_DIR = "performance_profiles_vlms"

MODEL_DEFAULTS = [
    # Keep conservative defaults. Can be overridden by env NIGHTLY_VLM_MODELS
    ModelLaunchSettings(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        extra_args=["--mem-fraction-static=0.7"],
    ),
    ModelLaunchSettings(
        "google/gemma-3-27b-it",
    ),
    ModelLaunchSettings("Qwen/Qwen3-VL-30B-A3B-Instruct", extra_args=["--tp=2"]),
    # "OpenGVLab/InternVL2_5-2B",
    # buggy in official transformers impl
    # "openbmb/MiniCPM-V-2_6",
]


class TestNightlyVLMModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )

        nightly_vlm_models_str = os.environ.get("NIGHTLY_VLM_MODELS")
        if nightly_vlm_models_str:
            cls.models = []
            model_paths = parse_models(nightly_vlm_models_str)
            for model_path in model_paths:
                cls.models.append(ModelLaunchSettings(model_path))
        else:
            cls.models = MODEL_DEFAULTS

        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "1,1,2,8,16")
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "512"))
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
                    # Run bench_one_batch_server against the launched server
                    profile_filename = f"{model_setup.model_path.replace('/', '_')}"
                    # path for this run
                    profile_path_prefix = os.path.join(PROFILE_DIR, profile_filename)

                    # JSON output file for this model
                    json_output_file = (
                        f"results_{model_setup.model_path.replace('/', '_')}.json"
                    )

                    command = [
                        "python3",
                        "-m",
                        "sglang.bench_one_batch_server",
                        f"--model={model_setup.model_path}",
                        "--base-url",
                        self.base_url,
                        "--batch-size",
                        *[str(x) for x in self.batch_sizes],
                        "--input-len",
                        *[str(x) for x in self.input_lens],
                        "--output-len",
                        *[str(x) for x in self.output_lens],
                        "--trust-remote-code",
                        "--dataset-name=mmmu",
                        "--profile",
                        "--profile-by-stage",
                        f"--profile-filename-prefix={profile_path_prefix}",
                        "--show-report",
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
                        continue

                    print(f"Output for {model_setup.model_path} with batch size:")
                    print(result.stdout)

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

                    else:
                        all_model_succeed = False
                        print(f"Warning: JSON output file {json_output_file} not found")

                finally:
                    kill_process_tree(process.pid)

                report_part = generate_markdown_report(
                    PROFILE_DIR,
                    benchmark_results,
                )
                self.full_report += report_part + "\n"

        if is_in_ci():
            write_github_step_summary(self.full_report)

        if not all_model_succeed:
            raise AssertionError("Some models failed the perf tests.")


if __name__ == "__main__":
    unittest.main()
