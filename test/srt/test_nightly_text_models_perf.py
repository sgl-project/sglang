import os
import subprocess
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
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

PROFILE_DIR = "performance_profiles_text_models"


class TestNightlyTextModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (parse_models("meta-llama/Llama-3.1-8B-Instruct"), False, False),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1), False, False),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 32, 64, 160, 256, 384]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "1024"))
        os.makedirs(PROFILE_DIR, exist_ok=True)
        cls.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self, batch_sizes: tuple = None):

        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):

                    process = popen_launch_server_wrapper(
                        self.base_url, model, "", ["--tp", "2"] if is_tp2 else []
                    )
                    model_results = []

                    trace_filename = f"{model.replace('/', '_')}_{int(time.time())}"
                    profile_path_prefix = os.path.join(PROFILE_DIR, trace_filename)

                    command = [
                        "python3",
                        "-m",
                        "sglang.bench_one_batch_server",
                        "--model",
                        model,
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
                    ]

                    print(f"Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"Error running benchmark for {model} with batch size :")
                        print(result.stderr)
                        # Continue to next batch size even if one fails
                        continue

                    print(f"Output for {model} with batch size:")
                    print(result.stdout)

                    trace_dir = extract_trace_link_from_bench_one_batch_server_output(
                        result.stdout
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

                    print(f"{model_results=}")

                    kill_process_tree(process.pid)

                    if model_results:
                        report_part = generate_markdown_report_nightly(
                            model, model_results, self.input_lens, self.output_lens
                        )
                        self.full_report += report_part + "\n"

        if is_in_ci():
            write_github_step_summary(self.full_report)


if __name__ == "__main__":
    unittest.main()
