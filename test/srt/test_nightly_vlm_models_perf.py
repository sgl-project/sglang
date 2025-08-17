import os
import re
import subprocess
import unittest
import warnings

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

PROFILE_URL_PLACEHOLDER = "<<ARTIFACT_URL>>"
REPORT_MD_FILENAME = "vlm_performance_report.md"
PROFILE_DIR = "performance_profiles_vlms"

MODEL_DEFAULTS = [
    # Keep conservative defaults. Can be overridden by env NIGHTLY_VLM_MODELS
    "Qwen/Qwen2-VL-7B-Instruct",
]


def parse_models(model_string: str):
    return [model.strip() for model in model_string.split(",") if model.strip()]


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


def _parse_bench_one_batch_metrics(stdout: str):
    # Extract key metrics from bench_one_batch_server stdout (last run)
    def _m_last(pattern: str, default: str = "n/a"):
        matches = list(re.finditer(pattern, stdout))
        return matches[-1].group(1) if matches else default

    metrics = {
        "latency_s": _m_last(r"latency:\s+([0-9.]+) s"),
        "ttft_s": _m_last(r"ttft:\s+([0-9.]+) s"),
        "input_throughput": _m_last(r"input throughput:\s+([0-9.]+) tok/s"),
        "output_throughput": _m_last(r"output throughput:\s+([0-9.]+) tok/s"),
        "last_gen_throughput": _m_last(
            r"last generation throughput:\s+([0-9.]+) tok/s"
        ),
    }
    return metrics


def _generate_markdown_report(
    model: str,
    metrics: dict,
    batch_sizes: list[int],
    input_lens: list[int],
    output_lens: list[int],
):
    summary = f"### {model}\n"
    summary += f"Batch sizes: {batch_sizes}. Input lens: {input_lens}. Output lens: {output_lens}.\n"
    summary += "| latency (s) | TTFT (s) | input tok/s | output tok/s | last gen tok/s | profile |\n"
    summary += "| ----------- | -------- | ----------- | ------------ | -------------- | ------- |\n"
    row = f"| {metrics['latency_s']} | {metrics['ttft_s']} | {metrics['input_throughput']} | {metrics['output_throughput']} | {metrics['last_gen_throughput']} | [Profile]({PROFILE_URL_PLACEHOLDER}) |\n"
    summary += row
    return summary


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

        # Bench knobs for bench_one_batch_server (override by env)
        def _parse_int_list_env(name: str, default_val: str):
            val = os.environ.get(name, default_val)
            return [int(x) for x in val.split() if x]

        cls.batch_sizes = _parse_int_list_env("NIGHTLY_VLM_BATCH_SIZES", "16")
        cls.input_lens = _parse_int_list_env("NIGHTLY_VLM_INPUT_LENS", "1024")
        cls.output_lens = _parse_int_list_env("NIGHTLY_VLM_OUTPUT_LENS", "16")

    def test_vlm_models_mmmu_performance(self):
        full_report = "## Nightly VLM Performance (bench_one_batch_server)\n"
        for model in self.models:
            with self.subTest(model=model):
                process = popen_launch_server_wrapper(self.base_url, model)
                try:
                    # Run bench_one_batch_server against the launched server
                    os.makedirs(PROFILE_DIR, exist_ok=True)
                    safe_model = model.replace("/", "_")
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
                        f"{PROFILE_DIR}/{safe_model}",
                    ]

                    print(f"Running command: {' '.join(command)}")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"Error running benchmark for {model}:")
                        print(result.stderr)
                        continue

                    print(f"Output for {model}:")
                    print(result.stdout)

                    metrics = _parse_bench_one_batch_metrics(result.stdout)
                    report_part = _generate_markdown_report(
                        model,
                        metrics,
                        self.batch_sizes,
                        self.input_lens,
                        self.output_lens,
                    )
                    full_report += report_part + "\n"
                finally:
                    kill_process_tree(process.pid)

        with open(REPORT_MD_FILENAME, "w") as f:
            print(f"results written to {REPORT_MD_FILENAME=}")
            f.write(full_report)


if __name__ == "__main__":
    unittest.main()
