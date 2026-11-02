import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_2_PCP_PD_SEP_PREFILL_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    # "SGLANG_ENABLE_SPEC_V2": "1",
    # "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_PCP_PD_SEP_DECODE_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    # "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DEEPEP_HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    # "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_PCP_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    2,
    "--tp-size",
    8,
    "--attn-cp-size",
    8,
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-dp-attention",
    "--mem-fraction-static",
    0.88,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    64,
    "--served-model-name",
    "glm-5.2-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--chunked-prefill-size",
    8192,
    "--max-prefill-tokens",
    65536,
    "--context-length",
    220000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--pp-size",
    4,
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
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-size",
    80,
    "--enable-cache-report",
]

GLM_5_2_PCP_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.88,
    "--max-running-requests",
    32,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--served-model-name",
    "glm-5.2-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--cuda-graph-max-bs-decode",
    4,
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--context-length",
    220000,
    "--tokenizer-worker-num",
    16,
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "unquant",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--enable-cache-report",
]

GLM_5_2_PCP_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_2_W8A8_MODEL_PATH,
    "prefill_args": GLM_5_2_PCP_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_2_PCP_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_2_PCP_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_2_PCP_PD_SEP_DECODE_ENVS,
    "router_args": [
        "--policy",
        "round_robin",
        "--request-timeout-secs",
        7200,
        "--disable-health-check",
        "--disable-circuit-breaker",
    ],
    "router_envs": {},
}


# PCP variant (wiki: GLM5.2-A3-4-node 2P2D with cpp-pcp-L2HiCache):
# prefill tp8 + attn-cp8 (interleave) + pp4, decode tp32/dp32/ep32
class TestNPUGLM5_2_W8A8_2P2D_PCP_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for GLM-5.2-w8a8 PD separation 2p2d with PCP on gpqa_diamond"""

    model_config = GLM_5_2_PCP_PD_SEP_MODEL_CONFIG
    # Run evalscope only once (no retry on accuracy miss)
    max_retries = 1
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    dataset_args = {
        "gpqa_diamond": {
            "filters": {"remove_until": "</think>"},
            "system_prompt": "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}.",
        }
    }
    eval_batch_size = 32
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    def test_npu_glm_5_2_w8a8_2p2d_pcp_gpqa(self):
        """Run NPU accuracy test for GLM-5.2-w8a8 PD separation 2p2d with PCP on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
