import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_2_W8A8C8_MXFP8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Based on the GLM-5.2 A5 two-node mix L2-HiCache launch script (pllimax wiki).
GLM_5_2_W8A8C8_MXFP8_ENVS = {
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "SGLANG_NPU_USE_MLAPO": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "35",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "GLOO_SOCKET_IFNAME": "data0.173",
}

GLM_5_2_W8A8C8_MXFP8_OTHER_ARGS = [
    "--served-model-name",
    "glm52",
    "--tp-size",
    8,
    "--nnodes",
    2,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    52,
    "--mem-fraction-static",
    0.84,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    2048000,
    "--max-total-tokens",
    65536,
    "--chunked-prefill-size",
    65536,
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--speculative-draft-kv-cache-dtype",
    "bf16",
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--enable-metrics",
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--enable-cache-report",
    "--hicache-ratio",
    2,
    "--hicache-write-policy",
    "write_through",
]

MODEL_CONFIG = {
    "model_path": GLM_5_2_W8A8C8_MXFP8_MODEL_PATH,
    "other_args": GLM_5_2_W8A8C8_MXFP8_OTHER_ARGS,
    "node_envs": GLM_5_2_W8A8C8_MXFP8_ENVS,
}


class TestNPUGLM5_2W8A8C8MXFP8_16PIn16kOut1kA5Overlap(TestNpuPerfMultiNodePdMixTestCaseBase):
    """GLM-5.2-w8a8c8-mxfp8 A5 two-node mix throughput with L2-HiCache, NEEDS=0, FIA=0."""

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
    # tpot/output_token_throughput baselines commented out; no A5 baseline yet.

    def test_npu_glm_5_2_w8a8c8_mxfp8_16p_in16k_out1k_a5_overlap(self):
        """Run NPU performance test for GLM-5.2-w8a8c8-mxfp8 A5 two-node mix."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
