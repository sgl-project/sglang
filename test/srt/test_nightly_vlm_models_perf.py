import os
import re
import subprocess
import time
import unittest
import warnings

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
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
    # "google/gemma-3-27b-it",
    # "openbmb/MiniCPM-V-2_6",
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
        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "1,2")
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "1024"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "32"))
        cls.full_report = f"## {cls.__name__}\n"

    def test_vlm_models_mmmu_performance(self):
        for model in self.models:
            model_results = []
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)
                try:
                    # Run bench_one_batch_server against the launched server
                    profile_filename = f"{model.replace('/', '_')}_{int(time.time())}"
                    # path for this run
                    profile_path_prefix = os.path.join(PROFILE_DIR, profile_filename)

                    command = [
                        "python3",
                        "-m",
                        "sglang.bench_one_batch_server",
                        f"--model={model}",
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
                        "--show-report",
                        "--profile",
                        "--profile-by-stage",
                        "--profile-filename-prefix",
                        profile_path_prefix,
                    ]

                    print(f"Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"Error running benchmark for {model} with batch size:")
                        print(result.stderr)
                        # Continue to next batch size even if one fails
                        continue

                    print(f"Output for {model} with batch size:")
                    print(result.stdout)

                    pattern = r"\[Profile\]\((.*?)\)"
                    trace_dirs = re.findall(pattern, result.stdout)

                    trace_filenames_from_all_dirs = [
                        find_traces_under_path(trace_dir) for trace_dir in trace_dirs
                    ]

                    extend_trace_filenames = [
                        trace_file
                        for trace_files in trace_filenames_from_all_dirs
                        for trace_file in trace_files
                        if trace_file.endswith(".EXTEND.trace.json.gz")
                    ]


                    # because the profile_id dir under PROFILE_DIR
                    extend_trace_file_relative_path_from_profile_dirs = [
                        f"{trace_dir[trace_dir.find(PROFILE_DIR) + len(PROFILE_DIR) + 1:]}/{extend_trace_filename}"
                        for extend_trace_filename, trace_dir in zip(
                            extend_trace_filenames, trace_dirs
                        )
                    ]


                    model_results.append(
                        {
                            "output": result.stdout,
                            "trace_links": extend_trace_file_relative_path_from_profile_dirs,
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

        print(f"{self.full_report=}")
        if is_in_ci():
            write_github_step_summary(self.full_report)


if __name__ == "__main__":
    unittest.main()
