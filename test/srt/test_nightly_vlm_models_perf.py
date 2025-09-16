import os
import subprocess
import time
import unittest
import warnings

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    extract_trace_link_from_bench_one_batch_server_output,
    find_traces_under_path,
    generate_markdown_report_nightly,
    is_in_ci,
    parse_models,
    popen_launch_server_wrapper,
    write_github_step_summary,
)

PROFILE_DIR = "performance_profiles_vlms"

MODEL_DEFAULTS = [
    # Keep conservative defaults. Can be overridden by env NIGHTLY_VLM_MODELS
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "google/gemma-3-27b-it",
    "openbmb/MiniCPM-V-2_6",
]


class TestNightlyVLMModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, message="unclosed.*socket"
        )
        cls.models = parse_models(
            os.environ.get("NIGHTLY_VLM_MODELS", ",".join(MODEL_DEFAULTS))
        )
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "1,1,2,8,16")
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "32"))
        cls.full_report = f"## {cls.__name__}\n"

    def test_vlm_models_mmmu_performance(self):
        for model in self.models:
            model_results = []
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model, PROFILE_DIR)
                try:
                    # Run bench_one_batch_server against the launched server
                    os.makedirs(PROFILE_DIR, exist_ok=True)
                    for batch_size in self.batch_sizes:
                        profile_filename = f"{model.replace('/', '_')}_bs{batch_size}_{int(time.time())}"
                        profile_path_prefix = os.path.join(
                            PROFILE_DIR, profile_filename
                        )

                        command = [
                            "python3",
                            "-m",
                            "sglang.bench_one_batch_server",
                            f"--model={model}",
                            "--base-url",
                            self.base_url,
                            "--batch-size",
                            str(batch_size),
                            "--trust-remote-code",
                            "--input-len",
                            *[str(x) for x in self.input_lens],
                            "--output-len",
                            *[str(x) for x in self.output_lens],
                            "--dataset-name=mmmu",
                            "--show-report",
                            "--profile",
                            "--profile-by-stage",
                            "--profile-filename-prefix",
                            profile_path_prefix,
                        ]

                        print(f"Running command: {' '.join(command)}")
                        result = subprocess.run(command, capture_output=True, text=True)

                        if result.returncode != 0:
                            print(
                                f"Error running benchmark for {model} with batch size {batch_size}:"
                            )
                            print(result.stderr)
                            # Continue to next batch size even if one fails
                            continue

                        print(f"Output for {model} with batch size {batch_size}:")
                        print(result.stdout)

                        trace_dir = (
                            extract_trace_link_from_bench_one_batch_server_output(
                                result.stdout
                            )
                        )

                        trace_files = find_traces_under_path(trace_dir)
                        extend_trace_filename = [
                            trace_file
                            for trace_file in trace_files
                            if trace_file.endswith(".EXTEND.trace.json.gz")
                        ][0]

                        # because the profile_id dir under PROFILE_DIR
                        extend_trace_file_relative_path_from_profile_dir = trace_dir[
                            trace_dir.find(PROFILE_DIR) + len(PROFILE_DIR) + 1 :
                        ]

                        model_results.append(
                            {
                                "output": result.stdout,
                                "trace_link": f"{extend_trace_file_relative_path_from_profile_dir}/{extend_trace_filename}",
                            }
                        )

                finally:
                    kill_process_tree(process.pid)

            if model_results:
                report_part = generate_markdown_report_nightly(
                    model,
                    model_results,
                    self.input_lens,
                    self.output_lens,
                )
                self.full_report += report_part + "\n"

        if is_in_ci():
            write_github_step_summary(self.full_report)


if __name__ == "__main__":
    unittest.main()
