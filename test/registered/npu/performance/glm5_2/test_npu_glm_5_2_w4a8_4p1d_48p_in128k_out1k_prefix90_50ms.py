import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_2_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_2_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "ZBAL_HCCL_OP": "send,recv",
    "HCCL_BUFFSIZE": "128",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "61184",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    "DEEP_USE_ALLTOALL_MODE": "1",
}

GLM_5_2_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HCCL_BUFFSIZE": "320",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "48",
    "TASK_QUEUE_ENABLE": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    4,
    "--nnodes",
    1,
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
    16,
    "--served-model-name",
    "glm-5",
    "--chunked-prefill-size",
    16384,
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
    "--speculative-draft-model-quantization",
    "unquant",
    "--enable-nsa-prefill-context-parallel",
    "--nsa-prefill-cp-mode",
    "in-seq-split",
    "--attn-cp-size",
    4,
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--pp-size",
    4,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
]

GLM_5_2_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    32,
    "--nnodes",
    2,
    "--dp-size",
    32,
    "--enable-dp-attention",
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.85,
    "--max-running-requests",
    384,
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
    "--cuda-graph-max-bs",
    12,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--context-length",
    180000,
    "--tokenizer-worker-num",
    16,
    "--prefill-round-robin-balance",
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
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
    "--disaggregation-decode-enable-radix-cache",
]

GLM_5_2_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_2_W4A8_MODEL_PATH,
    "prefill_args": GLM_5_2_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_2_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_2_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_2_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUGLM_5_2_W4A8_PD_SEP_In128k_Out1k(TestNpuPerfMultiNodePdSepTestCaseBase):
    """Test NPU performance for GLM-5.2-w4a8 PD separation 4p1d in128k out1k"""

    model_config = GLM_5_2_PD_SEP_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 192
    num_prompts = 200
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 2867

    def test_npu_glm_5_2_w4a8_pd_sep_in128k_out1k(self):
        """Run NPU performance test for GLM-5.2-w4a8 PD separation"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
