import json
import tempfile
import unittest
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestNpuPerformanceTestCaseBase,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="stage-b-test-2-npu-a3", nightly=False)

MODEL = "Qwen/Qwen3.5-35B-A3B"
INPUT_THROUGHPUT_THRESHOLD = 8500
MAX_CONCURRENCY = 32
NUM_PROMPTS = 128
INPUT_LEN = 4096
OUTPUT_LEN = 1
WARMUP_REQUESTS = 20
MAX_BENCHMARK_ATTEMPTS = 2

ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "HCCL_BUFFSIZE": "1200",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM": "1",
    "SGLANG_NPU_ENABLE_PIECEWISE_CUDA_GRAPH": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    0.7,
    "--max-running-requests",
    MAX_CONCURRENCY,
    "--attention-backend",
    "ascend",
    "--disable-radix-cache",
    "--cuda-graph-max-bs",
    32,
    "--tp-size",
    2,
    "--chunked-prefill-size",
    2048,
    "--cuda-graph-backend-prefill",
    "tc_piecewise",
    "--cuda-graph-bs-prefill",
    128,
    256,
    512,
    1024,
    2048,
]


def load_benchmark_result(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    return json.loads(lines[-1])


def benchmark_passed(metrics):
    return (
        metrics["completed"] == NUM_PROMPTS
        and metrics["max_concurrency"] == MAX_CONCURRENCY
        and metrics["input_throughput"] > INPUT_THROUGHPUT_THRESHOLD
    )


class TestPiecewiseGraphPrefillTp2(TestNpuPerformanceTestCaseBase):
    model = MODEL
    other_args = OTHER_ARGS
    envs = ENVS
    max_concurrency = MAX_CONCURRENCY
    num_prompts = NUM_PROMPTS
    input_len = INPUT_LEN
    output_len = OUTPUT_LEN
    warmup_requests = WARMUP_REQUESTS
    seed = 1

    def run_bench_serving_once(self, parsed_url):
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as output_file:
            run_bench_serving(
                host=parsed_url.hostname,
                port=parsed_url.port,
                model_path=self.model,
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                request_rate=self.request_rate,
                max_concurrency=self.max_concurrency,
                num_prompts=self.num_prompts,
                input_len=self.input_len,
                output_len=self.output_len,
                random_range_ratio=self.random_range_ratio,
                warmup_requests=self.warmup_requests,
                seed=self.seed,
                output_file=output_file.name,
            )
            return load_benchmark_result(output_file.name)

    def test_npu_piecewise_graph_prefill_tp2(self):
        print(f"##=== Testing PCG serving: {MODEL} ===##")
        parsed_url = urlparse(self.base_url)
        metrics_list = []
        for attempt in range(MAX_BENCHMARK_ATTEMPTS):
            metrics = self.run_bench_serving_once(parsed_url)
            metrics_list.append(metrics)
            if benchmark_passed(metrics):
                break

            print(
                "PCG benchmark attempt "
                f"{attempt + 1} failed: completed={metrics['completed']}, "
                f"max_concurrency={metrics['max_concurrency']}, "
                f"input_throughput={metrics['input_throughput']}"
            )

        passed_metrics = [
            metrics for metrics in metrics_list if benchmark_passed(metrics)
        ]
        metrics = (
            passed_metrics[-1]
            if passed_metrics
            else max(metrics_list, key=lambda x: x["input_throughput"])
        )

        self.assertEqual(metrics["completed"], self.num_prompts)
        self.assertEqual(metrics["max_concurrency"], self.max_concurrency)
        self.assertGreater(metrics["input_throughput"], INPUT_THROUGHPUT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
