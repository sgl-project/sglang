import logging
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
import sglang.test.ascend.e2e.test_npu_performance_utils as npu_perf_utils
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH, TestNpuPerfMultiNodePdMixTestCaseBase, AISBENCHMARK_DATASET_DEFAULT,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Metrics dict of the latest benchmark run. The runtime base class does not
# expose metrics from run_throughput(), so intercept run_bench_serving (called
# by run_throughput via the module global) to capture them for this file.
LAST_METRICS = {}

_orig_run_bench_serving = npu_perf_utils.run_bench_serving


def _run_bench_serving_capture(**kwargs):
    metrics = _orig_run_bench_serving(**kwargs)
    LAST_METRICS.clear()
    if metrics:
        LAST_METRICS.update(metrics)
    return metrics


npu_perf_utils.run_bench_serving = _run_bench_serving_capture

register_npu_ci(
    est_time=9600,
    suite="",
    nightly=True,
)

# Environment variables for DSV4-Pro-0813 two-node mix deployment,
DEEPSEEK_V4_PRO_W4A8_16P_ENVS = {
    "DEEPEP_HCCL_BUFFSIZE": "1536",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    # [DEEPEP]
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "30",
}


# Server launch arguments for DSV4-Pro W4A8 two-node 16p PD-mix.
DEEPSEEK_V4_PRO_W4A8_16P_OTHER_ARGS = [
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.8,
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    65536,
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    4,
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-metrics",
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
]


DEEPSEEK_V4_PRO_W4A8_16P_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "other_args": DEEPSEEK_V4_PRO_W4A8_16P_OTHER_ARGS,
    "node_envs": DEEPSEEK_V4_PRO_W4A8_16P_ENVS,
}

# Same launch arguments as above but with radix-cache enabled
# (remove "--disable-radix-cache") to compare TTFT against the disabled case.
DEEPSEEK_V4_PRO_W4A8_16P_RADIX_CACHE_OTHER_ARGS = [
    arg for arg in DEEPSEEK_V4_PRO_W4A8_16P_OTHER_ARGS if arg != "--disable-radix-cache"
]

DEEPSEEK_V4_PRO_W4A8_16P_RADIX_CACHE_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "other_args": DEEPSEEK_V4_PRO_W4A8_16P_RADIX_CACHE_OTHER_ARGS,
    "node_envs": DEEPSEEK_V4_PRO_W4A8_16P_ENVS,
}

# Cross-case TTFT store (ms), shared within the same test process.
TTFT_RESULTS = {}

# With repeat_rate=0.9 shared prefix, enabling radix-cache must cut TTFT
# to at most this ratio of the radix-cache-disabled TTFT.
RADIX_CACHE_TTFT_MAX_RATIO = 0.2


class TestNPUDeepSeekV4ProW4A88PIn128kOut1kPrefix90(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Pro W4A8 16p two-node in128k out1k prefix90."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = DEEPSEEK_V4_PRO_W4A8_16P_MODEL_CONFIG
    dataset_name = "generated-shared-prefix"
    warmup_requests = 16
    max_concurrency = 16
    num_prompts = 32
    repeat_rate = 0.9
    input_len = 117965
    output_len = 1024
    random_range_ratio = 1
    request_rate = float("inf")
    max_attempts = 3

    def test_npu_deepseek_v4_pro_w4a8_8p_in128k_out1k_prefix90(self):
        """Run NPU perf test for DeepSeek-V4-Pro W4A8 16p in128k out1k prefix90."""
        self.run_throughput()
        if LAST_METRICS.get("mean_ttft") is not None:
            TTFT_RESULTS["radix_cache_disabled"] = float(
                LAST_METRICS["mean_ttft"]
            )
            logger.info(
                "TTFT with radix-cache disabled: %s ms",
                TTFT_RESULTS["radix_cache_disabled"],
            )


# NOTE: This class must run AFTER TestNPUDeepSeekV4ProW4A88PIn128kOut1kPrefix90
# (alphabetical class ordering guarantees it) so that the radix-cache-disabled
# TTFT has been recorded in TTFT_RESULTS for comparison.
class TestNPUDeepSeekV4ProW4A88PIn128kOut1kPrefix90RadixCache(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Pro W4A8 16p two-node in128k out1k
    prefix90 with radix-cache enabled, and compare TTFT against the
    radix-cache-disabled case."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = DEEPSEEK_V4_PRO_W4A8_16P_RADIX_CACHE_MODEL_CONFIG
    dataset_name = "generated-shared-prefix"
    warmup_requests = 16
    max_concurrency = 16
    num_prompts = 32
    repeat_rate = 0.9
    input_len = 117965
    output_len = 1024
    random_range_ratio = 1
    request_rate = float("inf")
    max_attempts = 3

    def test_npu_deepseek_v4_pro_w4a8_8p_in128k_out1k_prefix90_radix_cache(self):
        """Run NPU perf test with radix-cache enabled; TTFT must drop
        significantly compared with the radix-cache-disabled case."""
        self.run_throughput()
        ttft_enabled = LAST_METRICS.get("mean_ttft")
        self.assertIsNotNone(ttft_enabled, "mean_ttft not found in bench_serving output")
        ttft_enabled = float(ttft_enabled)
        ttft_disabled = TTFT_RESULTS.get("radix_cache_disabled")
        self.assertIsNotNone(
            ttft_disabled,
            "TTFT of the radix-cache-disabled case was not recorded; "
            "TestNPUDeepSeekV4ProW4A88PIn128kOut1kPrefix90 must run first",
        )
        logger.info(
            "TTFT comparison: radix-cache enabled %s ms vs disabled %s ms (ratio %.2f)",
            ttft_enabled,
            ttft_disabled,
            ttft_enabled / ttft_disabled,
        )
        self.assertLessEqual(
            ttft_enabled,
            ttft_disabled * RADIX_CACHE_TTFT_MAX_RATIO,
            f"Enabling radix-cache should significantly reduce TTFT: "
            f"enabled {ttft_enabled} ms, disabled {ttft_disabled} ms, "
            f"expected ratio <= {RADIX_CACHE_TTFT_MAX_RATIO}",
        )

if __name__ == "__main__":
    unittest.main()