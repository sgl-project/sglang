import os
import subprocess
import unittest
import warnings

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    generate_markdown_report,
    is_in_ci,
    parse_models,
    popen_launch_server,
    write_github_step_summary,
)

PROFILE_URL_PLACEHOLDER = "<<ARTIFACT_URL>>"
REPORT_MD_FILENAME = "vlm_performance_report.md"
PROFILE_DIR = "performance_profiles_vlms"

MODEL_DEFAULTS = [
    # Keep conservative defaults. Can be overridden by env NIGHTLY_VLM_MODELS
    "Qwen/Qwen2-VL-7B-Instruct",
]


def _extra_args_for_model(model: str):
    # Align with TestNightlyVLMMmmuEval defaults
    args = ["--trust-remote-code", "--cuda-graph-max-bs", "4"]
    if model.startswith("google/gemma-3"):
        args += ["--enable-multimodal"]
    return args


def popen_launch_server_wrapper(base_url: str, model: str):
    other_args = _extra_args_for_model(model)
    env = os.environ.copy()
    env["SGLANG_TORCH_PROFILER_DIR"] = os.path.abspath(PROFILE_DIR)
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=env,
    )
    return process


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
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "16"))

    def test_vlm_models_mmmu_performance(self):
        full_report = (
            "## TestNightlyVLMModelsPerformance (with bench_one_batch_server)\n"
        )
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)
                model_results = []
                try:
                    # Run bench_one_batch_server against the launched server
                    os.makedirs(PROFILE_DIR, exist_ok=True)
                    profile_filename = model.replace("/", "_")
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
                        "--dataset-name=mmmu",
                        "--show-report",
                        "--profile",
                        "--profile-by-stage",
                        "--profile-filename-prefix",
                        f"{PROFILE_DIR}/{profile_filename}",
                    ]

                    print(f"Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"Error running benchmark for {model}:")
                        print(result.stderr)
                        continue

                    print(f"Output for {model}:")
                    print(result.stdout)
                    model_results.append(
                        {
                            "output": result.stdout,
                            "profile_filename": f"{profile_filename}.json",
                        }
                    )

                finally:
                    kill_process_tree(process.pid)

            if model_results:
                report_part = generate_markdown_report(
                    model,
                    model_results,
                    self.input_lens,
                    self.output_lens,
                )
                full_report += report_part + "\n"

        with open(REPORT_MD_FILENAME, "w") as f:
            print(f"results written to {REPORT_MD_FILENAME=}")
            f.write(full_report)

        if is_in_ci():
            write_github_step_summary(full_report)


if __name__ == "__main__":
    unittest.main()
