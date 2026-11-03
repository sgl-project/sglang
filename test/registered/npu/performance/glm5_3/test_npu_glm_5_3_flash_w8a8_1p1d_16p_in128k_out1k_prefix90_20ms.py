import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_3_FLASH_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_3_FLASH_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_DSA_NPU_TOPK_NATIVE": "False",
    "SGLANG_DSA_NPU_TOPK_TRITON": "False",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "128",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "HCCL_IF_BASE_PORT": "61000",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_FLASH_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_DSA_NPU_TOPK_NATIVE": "False",
    "SGLANG_DSA_NPU_TOPK_TRITON": "False",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_BUFFSIZE": "512",
    "DEEPEP_HCCL_BUFFSIZE": "512",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "61010",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_FLASH_PD_SEP_PREFILL_ARGS = [
    "--quantization",
    "modelslim",
    "--device",
    "npu",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--enable-multimodal",
    "--served-model-name",
    "GLM-NEXT",
    "--tp-size",
    16,
    "--pp-size",
    1,
    "--dp-size",
    1,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--nnodes",
    1,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--load-balance-method",
    "round_robin",
    "--mem-fraction-static",
    0.78,
    "--page-size",
    64,
    "--max-running-requests",
    16,
    "--chunked-prefill-size",
    8192,
    "--max-prefill-tokens",
    300000,
    "--watchdog-timeout",
    9000,
]

GLM_5_3_FLASH_PD_SEP_DECODE_ARGS = [
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    # Flash MTP block has no ModelSlim description; keep the draft unquantized.
    "--speculative-draft-model-quantization",
    "unquant",
    "--quantization",
    "modelslim",
    "--device",
    "npu",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--enable-multimodal",
    "--served-model-name",
    "GLM-NEXT",
    "--tp-size",
    16,
    "--pp-size",
    1,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--nnodes",
    1,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--load-balance-method",
    "round_robin",
    "--mem-fraction-static",
    0.75,
    "--page-size",
    64,
    "--max-running-requests",
    128,
    "--context-length",
    180000,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
    "--watchdog-timeout",
    9000,
]

GLM_5_3_FLASH_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_3_FLASH_W8A8_MODEL_PATH,
    "prefill_args": GLM_5_3_FLASH_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_3_FLASH_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_3_FLASH_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_3_FLASH_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUGLM5_3_FLASH_W8A8_1P1D_16P_In128k_Out1k_Prefix90_20ms(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for GLM-5.3-Flash-w8a8 PD separation 1p1d 16p in128k out1k prefix90"""

    model_config = GLM_5_3_FLASH_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "generated-shared-prefix"
    max_concurrency = 16
    num_prompts = 128
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    repeat_rate = 0.9
    seed = 1
    tpot = None
    output_token_throughput = None
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_glm_5_3_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms(self):
        """Run NPU performance test for GLM-5.3-Flash-w8a8 PD separation 1p1d 16p"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
