import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# 临时手动指定模型路径 (wiki 配置)
GLM_5_2_W8A8C8_MXFP8_MODEL_PATH = "/mnt/share/w00936111/GLM-5.2-W8A8C8-mxfp8"

# A5 wiki: GLM-5.2-四机PD分离-PCP-A5
# prefill: 2 nodes, tp4 x cp4 (interleave) x pp4 (16 NPU)
# decode : 2 nodes, tp16 x dp16 (16 NPU)
GLM_5_2_A5_PCP_PD_SEP_PREFILL_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "ASCEND_MF_LOG_LEVEL": "2",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    # PCP 必需: prefill 有 CP、decode 无 CP, 否则 P->D KV 传输失败
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_PP_OUTPUT_VIA_CPU": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_A5_PCP_PD_SEP_DECODE_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "device_urma",
    "ASCEND_MF_LOG_LEVEL": "2",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "MF_HYBM_USE_VMM_SEGMENT": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_MLAPO": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "DEEPEP_HCCL_BUFFSIZE": "400",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_A5_PCP_PD_SEP_PREFILL_ARGS = [
    "--served-model-name",
    "glm52",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--nnodes",
    2,
    "--tp-size",
    4,
    "--attn-cp-size",
    4,
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--pp-size",
    4,
    "--load-balance-method",
    "round_robin",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.88,
    "--quantization",
    "modelslim",
    "--max-prefill-tokens",
    180000,
    "--chunked-prefill-size",
    4096,
    "--disable-shared-experts-fusion",
    "--moe-dense-tp",
    1,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--enable-metrics",
]

GLM_5_2_A5_PCP_PD_SEP_DECODE_ARGS = [
    "--served-model-name",
    "glm52",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--nnodes",
    2,
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--load-balance-method",
    "round_robin",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
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
    0.88,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--cuda-graph-max-bs-decode",
    8,
    "--tokenizer-worker-num",
    16,
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
    "--context-length",
    220000,
    "--disable-shared-experts-fusion",
    "--enable-metrics",
]

GLM_5_2_A5_PCP_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_2_W8A8C8_MXFP8_MODEL_PATH,
    "prefill_args": GLM_5_2_A5_PCP_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_2_A5_PCP_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_2_A5_PCP_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_2_A5_PCP_PD_SEP_DECODE_ENVS,
    "router_args": [
        "--policy",
        "cache_aware",
        "--request-timeout-secs",
        7200,
        "--disable-circuit-breaker",
    ],
    "router_envs": {},
}


class TestNPUGLM5_2_W8A8C8_MXFP8_1P1D_PCP_GPQA_A5(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test A5 accuracy for GLM-5.2-W8A8C8-mxfp8 PD separation 1p1d with PCP on gpqa_diamond"""

    model_config = GLM_5_2_A5_PCP_PD_SEP_MODEL_CONFIG
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

    def test_npu_glm_5_2_w8a8c8_mxfp8_1p1d_pcp_gpqa_a5(self):
        """Run A5 accuracy test for GLM-5.2-W8A8C8-mxfp8 PD separation 1p1d with PCP on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
