import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_3_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_3_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
}

GLM_5_3_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEPEP_HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

GLM_5_3_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--trust-remote-code",
    "--nnodes",
    2,
    "--tp-size",
    8,
    "--pp-size",
    4,
    "--dp-size",
    1,
    "--attn-cp-size",
    8,
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--load-balance-method",
    "round_robin",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    64,
    "--served-model-name",
    "glm-5",
    "--chunked-prefill-size",
    4096,
    "--max-prefill-tokens",
    180000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
]

GLM_5_3_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--trust-remote-code",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--dp-size",
    32,
    "--enable-dp-attention",
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.97,
    "--max-running-requests",
    32,
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
    "--deepep-mode",
    "low_latency",
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--context-length",
    180000,
    "--tokenizer-worker-num",
    4,
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
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
]

GLM_5_3_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_3_W8A8_MODEL_PATH,
    "prefill_args": GLM_5_3_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_3_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_3_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_3_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUGLM5_3_W8A8_2P2D_32P_In128k_Out1k_Prefix90_20ms(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for GLM-5.3-w8a8 PD separation 2p2d in128k out1k prefix90"""

    model_config = GLM_5_3_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "generated-shared-prefix"
    max_concurrency = 64
    num_prompts = 256
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    repeat_rate = 0.9
    seed = 1
    tpot = 20
    output_token_throughput = 620
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_glm_5_3_w8a8_2p2d_32p_in128k_out1k_prefix90_20ms(self):
        """Run NPU performance test for GLM-5.3-w8a8 PD separation"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
