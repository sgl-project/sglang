import os
import subprocess
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    generate_markdown_report_nightly,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
)

PROFILE_DIR = "performance_profiles_text_models"
PROFILE_URL_PLACEHOLDER = "<<ARTIFACT_URL>>"
REPORT_MD_FILENAME = "performance_report.md"


def popen_launch_server_wrapper(base_url, model, is_tp2):
    other_args = ["--log-level-http", "warning", "--trust-remote-code"]
    if is_tp2:
        other_args.extend(["--tp", "2"])
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


class TestNightlyTextModelsPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_groups = [
            (
                parse_models(
                    "meta-llama/Llama-3.1-8B-Instruct,mistralai/Mistral-7B-Instruct-v0.3"
                ),
                False,
                False,
            ),
            # (parse_models("meta-llama/Llama-3.1-8B-Instruct"), False, False),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2), False, True),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1), True, False),
            # (parse_models(DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2), True, True),
        ]
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 32, 64, 160, 256, 384]
        cls.batch_sizes = [1, 1, 8, 32]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "1024"))
        os.makedirs(PROFILE_DIR, exist_ok=True)

    def test_bench_one_batch(self, batch_sizes: tuple = None):
        full_report = f"## {self.__class__.__name__}\n"

        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):

                    process = popen_launch_server_wrapper(self.base_url, model, is_tp2)
                    model_results = []

                    for batch_size in batch_sizes or self.batch_sizes:
                        profile_filename = f"{model.replace('/', '_')}_bs{batch_size}_{int(time.time())}"
                        profile_path_prefix = os.path.join(
                            PROFILE_DIR, profile_filename
                        )

                        command = [
                            "python3",
                            "-m",
                            "sglang.bench_one_batch_server",
                            "--model",
                            model,
                            "--base-url",
                            self.base_url,
                            "--batch-size",
                            str(batch_size),
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
                            print(
                                f"Error running benchmark for {model} with batch size {batch_size}:"
                            )
                            print(result.stderr)
                            # Continue to next batch size even if one fails
                            continue

                        print(f"Output for {model} with batch size {batch_size}:")
                        print(result.stdout)

                        model_results.append(
                            {
                                "output": result.stdout,
                                "profile_filename": f"{profile_filename}.json",
                            }
                        )

                    kill_process_tree(process.pid)

                    if model_results:
                        report_part = generate_markdown_report_nightly(
                            model, model_results, self.input_lens, self.output_lens
                        )
                        full_report += report_part + "\n"

        # Persist the report for later substitution of artifact URL in the workflow
        with open(REPORT_MD_FILENAME, "w") as f:
            print(f"results written to {REPORT_MD_FILENAME=}")
            f.write(full_report)

        if is_in_ci():
            write_github_step_summary(full_report)


if __name__ == "__main__":
    unittest.main()
