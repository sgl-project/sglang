"""
Usage:
python3 -m unittest test_cpu_graph.TestCPUGraph.test_mmlu_torch_compile_cpu
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
    intel_amx_benchmark,
    is_in_ci,
    popen_launch_server,
)


class TestCPUGraph(CustomTestCase):

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "1",
            "--mem-fraction-static",
            "0.05",
            "--enable-torch-compile",
            "--torch-compile-max-bs",
            "1",
        ],
        min_throughput=10,
    )
    def test_latency_torch_compile_cpu(self):
        return DEFAULT_MLA_MODEL_NAME_FOR_TEST

    def test_mmlu_torch_compile_cpu(self):
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
                "1",
                "--tp",
                f"{n_numa_node}",
            ],
            env=env,
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
            if is_in_ci():
                self.assertGreater(metrics["score"], 0.45)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
