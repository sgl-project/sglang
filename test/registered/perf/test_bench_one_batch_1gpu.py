import os
import re
import subprocess
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    kill_process_tree,
    run_bench_one_batch,
    write_github_step_summary,
)

register_cuda_ci(est_time=95, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-large-amd")


class TestBenchOneBatch1GPU(CustomTestCase):

    def test_bs1_small(self):
        _, output_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, ["--cuda-graph-max-bs", "2"]
        )
        self.assertGreater(output_throughput, 50)

    def test_bs1_default(self):
        env = os.environ.copy()
        env["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] = "1"

        command = [
            "python3",
            "-m",
            "sglang.bench_offline_throughput",
            "--num-prompts",
            "1",
            "--dataset-name",
            "random",
            "--random-input-len",
            "256",
            "--random-output-len",
            "1024",
            "--model-path",
            DEFAULT_MODEL_NAME_FOR_TEST,
            "--cuda-graph-max-bs",
            "2",
        ]

        print(f"command={' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        try:
            stdout, stderr = process.communicate()
            output = stdout.decode(errors="backslashreplace")
            error = stderr.decode(errors="backslashreplace")
            print(f"Output: {output}", flush=True)
            print(f"Error: {error}", flush=True)

            output_throughput = -1
            for line in output.split("\n"):
                if "Last generation throughput (tok/s):" in line:
                    output_throughput = float(line.split(":")[-1])
        finally:
            kill_process_tree(process.pid)

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs1_default (llama-3.1-8b)\n"
                f"output_throughput: {output_throughput:.2f} token/s\n"
            )
            self.assertGreater(output_throughput, 135)

        gpu_busy_values = []
        for line in error.split("\n"):
            match = re.search(r"GPU busy:\s*([\d.]+|nan)%", line)
            if match:
                val = match.group(1)
                if val != "nan":
                    gpu_busy_values.append(float(val))

        print(f"{gpu_busy_values=}", flush=True)
        self.assertGreater(len(gpu_busy_values), 0, "No GPU busy values found in logs")

        gpu_busy_values_p90 = float(np.percentile(gpu_busy_values, 90))
        print(f"{gpu_busy_values_p90=}", flush=True)
        self.assertGreater(gpu_busy_values_p90, 99.1)


if __name__ == "__main__":
    unittest.main()
