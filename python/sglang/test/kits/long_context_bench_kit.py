import os
import subprocess
import sys


class LongContextBenchServingMixin:
    bench_backend: str = "sglang-oai-chat"
    bench_dataset_name: str = "random"
    bench_random_input_len: str = "131072"
    bench_random_output_len: str = "8192"
    bench_random_range_ratio: str = "1.0"
    bench_num_prompts: str = "32"
    bench_max_concurrency: str = "8"
    bench_warmup_requests: str = "8"
    bench_timeout_sec: int = 1800

    def test_long_context_bench_serving(self):
        num_prompts = os.environ.get("NUM_PROMPTS", self.bench_num_prompts)
        max_concurrency = os.environ.get("MAX_CONCURRENCY", self.bench_max_concurrency)
        warmup_requests = os.environ.get("WARMUP_REQUESTS", self.bench_warmup_requests)
        random_input_len = os.environ.get(
            "RANDOM_INPUT_LEN", self.bench_random_input_len
        )
        random_output_len = os.environ.get(
            "RANDOM_OUTPUT_LEN", self.bench_random_output_len
        )
        timeout_sec = int(os.environ.get("BENCH_TIMEOUT_SEC", self.bench_timeout_sec))

        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            str(self.bench_backend),
            "--base-url",
            str(self.base_url),
            "--model",
            str(self.model),
            "--dataset-name",
            str(self.bench_dataset_name),
            "--random-input-len",
            str(random_input_len),
            "--random-output-len",
            str(random_output_len),
            "--random-range-ratio",
            str(self.bench_random_range_ratio),
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(max_concurrency),
            "--warmup-requests",
            str(warmup_requests),
        ]

        print(f"Running command: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            print(exc.stdout or "")
            print(exc.stderr or "")
            raise
        print(result.stdout)
        print(result.stderr)

        self.assertEqual(result.returncode, 0)
