import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_3_DCP_PD_SEP_PREFILL_ENVS = {
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
    # "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    # "SGLANG_ENABLE_SPEC_V2": "1",
    # "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_DCP_PD_SEP_DECODE_ENVS = {
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
    # "DEEPEP_HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    # "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_DCP_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    2,
    "--tp-size",
    8,
    "--ep-size",
    8,
    "--dp-size",
    8,
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
    "glm-5.3-w8a8",
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
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill": "alltoallv", "decode": "low_latency"}',
    "--disable-shared-experts-fusion",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--pp-size",
    4,
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
    # EPLB
    "--enable-eplb",
    "--expert-distribution-recorder-mode",
    "stat",
    "--ep-num-redundant-experts",
    16,
    "--eplb-rebalance-num-iterations",
    100000,
    "--expert-distribution-recorder-buffer-size",
    50,
]

GLM_5_3_DCP_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--dcp-size",
    4,
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.88,
    "--max-running-requests",
    48,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--served-model-name",
    "glm-5.3-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--moe-a2a-backend",
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill": "alltoallv", "decode": "low_latency"}',
    "--cuda-graph-max-bs-decode",
    8,
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
    # EPLB
    "--enable-eplb",
    "--expert-distribution-recorder-mode",
    "stat",
    "--ep-num-redundant-experts",
    32,
    "--eplb-rebalance-num-iterations",
    100000,
    "--expert-distribution-recorder-buffer-size",
    1000,
]

GLM_5_3_DCP_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_3_W8A8_MODEL_PATH,
    "prefill_args": GLM_5_3_DCP_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_3_DCP_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_3_DCP_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_3_DCP_PD_SEP_DECODE_ENVS,
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


# DCP variant with EPLB
# 4-node PD separation, prefill dp8 + pp4, decode tp32/dp8/dcp4,
# EPLB enabled on both P and D with GLM-5.3-w8a8
# MoE A2A backend: ascend_ops (prefill=alltoallv, decode=low_latency) on both P and D
class TestNPUGLM5_3_W8A8_1P1D_DCP_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for GLM-5.3-w8a8 PD separation 1p1d with DCP and EPLB on gpqa_diamond"""

    model_config = GLM_5_3_DCP_PD_SEP_MODEL_CONFIG
    # Run evalscope only once (no retry on accuracy miss)
    max_retries = 1
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.92,
    }
    dataset_args = {
        "gpqa_diamond": {
            "filters": {"remove_until": "</think>"},
            "system_prompt": "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}.",
        },
        "gsm8k": {"filters": {"remove_until": "</think>"}},
    }
    eval_batch_size = 48
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    def test_npu_glm_5_3_w8a8_1p1d_dcp_gpqa(self):
        """Run NPU accuracy test for GLM-5.3-w8a8 PD separation 1p1d with DCP and EPLB on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
