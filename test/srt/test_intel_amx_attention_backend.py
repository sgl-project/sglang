"""
Usage:
python3 -m unittest test_intel_amx_attention_backend.TestIntelAMXAttnBackend.test_mmlu
"""

import copy
import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import get_cpu_ids_by_node, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch,
)


class TestIntelAMXAttnBackend(CustomTestCase):
    def test_latency(self):
        prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
            DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--batch-size",
                "4",
            ],
        )

        print(f"{prefill_latency=}")
        print(f"{decode_throughput=}")
        print(f"{decode_latency=}")

        if is_in_ci():
            self.assertGreater(decode_throughput, 10)

    def test_mmlu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreater(metrics["score"], 0.5)
        finally:
            kill_process_tree(process.pid)

    def test_torch_compile_cpu(self):
        model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        env = copy.deepcopy(os.environ)
        env["SGLANG_CPU_OMP_THREADS_BIND"] = "all"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "intel_amx",
                "--mem-fraction-static",
                "0.05",
                "--disable-radix",
                "--trust-remote-code",
                "--disable-overlap-schedule",
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                "4",
                "--tp",
                f"{n_numa_node}",
            ],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            self.assertGreater(metrics["score"], 0.5)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
