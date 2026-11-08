import os
import subprocess
import unittest

from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression, record_pass1
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.utils import wait_for_server

register_npu_ci(
    est_time=3600,
    suite="nightly-perf-4-npu-a5",
    nightly=True,
)

CASE_ID = "test_npu_deepseek_v4_flash_w8a8_4p_in16k_out1k_a5_overlap"

# Based on the DSV4-w8a8 launch scripts in PR #46805.
DEEPSEEK_V4_FLASH_W8A8_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # mtp
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "SGLANG_DSPARK_USE_GREEDY_DRAFT": "1",
}

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
DEEPSEEK_V4_FLASH_W8A8_4P_PASS1_ENVS = {
    **{k: v for k, v in DEEPSEEK_V4_FLASH_W8A8_4P_ENVS.items() if k != "ASCEND_USE_FIA"},
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
}

DEEPSEEK_V4_FLASH_W8A8_4P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--mem-fraction-static",
    0.68,
    "--prefill-max-requests",
    192,
    "--max-prefill-tokens",
    80000,
    "--chunked-prefill-size",
    65536,
    "--max-running-requests",
    192,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "fp8",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
]


class DeepSeekV4FlashW8A8_4PIn16kOut1kA5OverlapBase(TestNpuPerformanceTestCaseBase):
    """Shared config for the two-pass (FIA unset vs FIA=0) throughput comparison."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_4P_OTHER_ARGS
    dataset_name = "random"
    input_len = 16000
    output_len = 1000
    num_prompts = 256
    max_concurrency = 256
    random_range_ratio = 1
    request_rate = float("inf")
    seed = 1
    # tpot/output_token_throughput baselines commented out; the +/-2% comparison
    # against pass 1 is the gate.

    @classmethod
    def setUpClass(cls):
        """Launch server via `python3 -m sglang.launch_server` instead of `sglang serve`."""
        cls._setup_per_case_output()
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        if cls.envs:
            env.update(cls.envs)

        _, host, port = cls.base_url.split(":")
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            *[str(x) for x in cls.other_args],
            "--host",
            host[2:],
            "--port",
            port,
        ]
        cls.process = subprocess.Popen(command, env=env)
        wait_for_server(cls.base_url, timeout=cls.timeout, process=cls.process)


class TestNPUDeepSeekV4FlashW8A8_4PIn16kOut1kA5OverlapPass1(
    DeepSeekV4FlashW8A8_4PIn16kOut1kA5OverlapBase
):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    envs = DEEPSEEK_V4_FLASH_W8A8_4P_PASS1_ENVS

    def test_npu_deepseek_v4_flash_w8a8_4p_in16k_out1k_a5_overlap_pass1(self):
        """Run pass-1 throughput and record the baseline metrics."""
        self.run_throughput()
        record_pass1(CASE_ID, self.last_metrics)


class TestNPUDeepSeekV4FlashW8A8_4PIn16kOut1kA5OverlapPass2(
    DeepSeekV4FlashW8A8_4PIn16kOut1kA5OverlapBase
):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    envs = DEEPSEEK_V4_FLASH_W8A8_4P_ENVS

    def test_npu_deepseek_v4_flash_w8a8_4p_in16k_out1k_a5_overlap_pass2(self):
        """Run pass-2 throughput and assert no regression vs pass 1."""
        self.run_throughput()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
