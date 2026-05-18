import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_1_W4A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_1_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_1_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "650",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "48",
    "TASK_QUEUE_ENABLE": "0",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_1_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--mem-fraction-static",
    0.72,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    192,
    "--served-model-name",
    "glm-5",
    "--chunked-prefill-size",
    16384,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--speculative-draft-model-quantization",
    "unquant",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
]

GLM_5_1_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    64,
    "--nnodes",
    4,
    "--dp-size",
    64,
    "--ep-size",
    64,
    "--enable-dp-attention",
    "--mem-fraction-static",
    0.84,
    "--max-running-requests",
    192,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--served-model-name",
    "glm-5",
    "--moe-a2a-backend",
    "deepep",
    "--enable-dp-attention",
    "--deepep-mode",
    "low_latency",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--cuda-graph-bs",
    1,
    2,
    3,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--context-length",
    180000,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--tokenizer-worker-num",
    4,
    "--prefill-round-robin-balance",
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
]

GLM_5_1_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_1_W4A8_MODEL_PATH,
    "prefill_args": GLM_5_1_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_1_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_1_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_1_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUGLM5_1_W4A8_PD_SEP_In3k5_Out1k5(TestAscendPerfMultiNodePdSepTestCaseBase):
    """Test NPU performance for GLM-5.1-w4a8 PD separation 4 nodes in3k5 out1k5"""

    model_config = GLM_5_1_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 128
    num_prompts = 512
    input_len = 66560
    output_len = 1536
    random_range_ratio = 1
    tpot = 33
    output_token_throughput = 166145
    aisbench_repeat_rate = 1.0

    def test_npu_glm5_1_w4a8_pd_sep_in3k5_out1k5(self):
        """Run NPU performance test for GLM-5.1-w4a8 PD separation"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
