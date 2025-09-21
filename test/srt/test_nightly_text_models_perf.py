import os
import re
import subprocess
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_FP8_TP2,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP1,
    DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_TP2,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    find_traces_under_path,
    is_in_ci,
    parse_models,
    popen_launch_server,
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
        cls.batch_sizes = [1, 1, 8]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "512"))
        os.makedirs(PROFILE_DIR, exist_ok=True)
        cls.full_report = f"## {cls.__name__}\n"

    def test_bench_one_batch(self):

        for model_group, is_fp8, is_tp2 in self.model_groups:
            for model in model_group:
                with self.subTest(model=model):
                    process = popen_launch_server(
                        model=model,
                        base_url=self.base_url,
                        other_args=["--tp", "2"] if is_tp2 else [],
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    )
                    try:

                        model_results = []

                        profile_filename = (
                            f"{model.replace('/', '_')}_{int(time.time())}"
                        )
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
                            "--append-to-github-summary=False",
                        ]

                        print(f"Running command: {' '.join(command)}")
                        result = subprocess.run(command, capture_output=True, text=True)

                    finally:
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
