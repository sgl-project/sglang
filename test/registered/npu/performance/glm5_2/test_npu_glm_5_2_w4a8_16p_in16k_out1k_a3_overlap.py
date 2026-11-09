import os
import subprocess
import unittest

from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression, record_pass1
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_2_W4A8_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.utils import wait_for_server

register_npu_ci(
    est_time=3600,
    suite="nightly-perf-16-npu-a3",
    nightly=True,
)

CASE_ID = "test_npu_glm_5_2_w4a8_16p_in16k_out1k_a3_overlap"

# Based on the GLM-5.2-w4a8 benchmark scripts in PR #45949.
GLM_5_2_W4A8_16P_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "HCCL_BUFFSIZE": "500",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TRANSFORMERS_VERBOSITY": "error",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
}

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
GLM_5_2_W4A8_16P_PASS1_ENVS = {
    **{k: v for k, v in GLM_5_2_W4A8_16P_ENVS.items() if k != "ASCEND_USE_FIA"},
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
}

GLM_5_2_W4A8_16P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--max-running-requests",
    64,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--cuda-graph-bs-decode",
    8,
    16,
    24,
    32,
    "--enable-metrics",
]


class Glm52W4A8_16PIn16kOut1kA3OverlapBase(TestNpuPerformanceTestCaseBase):
    """Shared config for the two-pass (FIA unset vs FIA=0) throughput comparison."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = GLM_5_2_W4A8_MODEL_PATH
    other_args = GLM_5_2_W4A8_16P_OTHER_ARGS
    dataset_name = "random"
    input_len = 16000
    output_len = 1000
    num_prompts = 16
    max_concurrency = 4
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


class TestNPUGLM5_2W4A8_16PIn16kOut1kA3OverlapPass1(Glm52W4A8_16PIn16kOut1kA3OverlapBase):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    envs = GLM_5_2_W4A8_16P_PASS1_ENVS

    def test_npu_glm_5_2_w4a8_16p_in16k_out1k_a3_overlap_pass1(self):
        """Run pass-1 throughput and record the baseline metrics."""
        self.run_throughput()
        record_pass1(CASE_ID, self.last_metrics)


class TestNPUGLM5_2W4A8_16PIn16kOut1kA3OverlapPass2(Glm52W4A8_16PIn16kOut1kA3OverlapBase):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    envs = GLM_5_2_W4A8_16P_ENVS

    def test_npu_glm_5_2_w4a8_16p_in16k_out1k_a3_overlap_pass2(self):
        """Run pass-2 throughput and assert no regression vs pass 1."""
        self.run_throughput()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
