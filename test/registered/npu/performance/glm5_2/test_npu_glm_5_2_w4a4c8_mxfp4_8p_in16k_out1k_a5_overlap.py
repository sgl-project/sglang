import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_2_W4A4C8_MXFP4_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

CASE_ID = "test_npu_glm_5_2_w4a4c8_mxfp4_8p_in16k_out1k_a5_overlap"

# Based on the GLM-5.2 W4A4C8-mxfp4 single-node A5 mix launch script.
# Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0.
GLM_5_2_W4A4C8_MXFP4_ENVS = {
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "CCU_SCHED",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "0",
    "SGLANG_NPU_USE_MLAPO": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": "data0.3001",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
}

GLM_5_2_W4A4C8_MXFP4_OTHER_ARGS = [
    "--served-model-name",
    "glm52",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--nnodes",
    1,
    "--tp-size",
    8,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--model-loader-extra-config",
    '{"enable_multithread_load":false}',
    "--load-balance-method",
    "round_robin",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    128,
    "--mem-fraction-static",
    0.85,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-max-bs-decode",
    8,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--dtype",
    "bfloat16",
    "--disable-shared-experts-fusion",
    "--enable-metrics",
]

MODEL_CONFIG = {
    "model_path": GLM_5_2_W4A4C8_MXFP4_MODEL_PATH,
    "other_args": GLM_5_2_W4A4C8_MXFP4_OTHER_ARGS,
    "node_envs": GLM_5_2_W4A4C8_MXFP4_ENVS,
}


class TestNPUGLM5_2W4A4C8MXFP4_8PIn16kOut1kA5Overlap(TestNpuPerfMultiNodePdMixTestCaseBase):
    """Pass 2: GLM-5.2-w4a4c8-mxfp4 A5 single-node mix throughput, NEEDS=0, FIA=0."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = MODEL_CONFIG
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

    def test_npu_glm_5_2_w4a4c8_mxfp4_8p_in16k_out1k_a5_overlap(self):
        """Run pass-2 throughput and assert no regression vs pass 1."""
        self.run_throughput()
        # Gates tpot (mean_tpot) and output_token_throughput (total_tps)
        # within 2% of the pass-1 snapshot.
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
