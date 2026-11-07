import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    get_mooncake_ld_library_path,
    write_memfabric_tent_conf,
)
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


GLM_5_3_PD_SEP_COMMON_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_MF_LOG_LEVEL": "2",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    # Mooncake + TENT + MemFabric
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
}

GLM_5_3_PD_SEP_PREFILL_ENVS = {
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
}

GLM_5_3_PD_SEP_DECODE_ENVS = {
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "DEEPEP_HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

GLM_5_3_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
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
    "mooncake",
    "--max-running-requests",
    64,
    "--served-model-name",
    "glm-5",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
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
    "--mem-fraction-static",
    0.72,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
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
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    "--disaggregation-transfer-backend",
    "mooncake",
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

GLM_5_3_PD_SEP_ROUTER_ARGS = [
    "--policy",
    "round_robin",
    "--request-timeout-secs",
    7200,
    "--disable-health-check",
    "--disable-circuit-breaker",
]


class TestNPUGLM5_3_W8A8_1P1D_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """NPU GPQA accuracy for GLM-5.3-w8a8 PD separation 1p1d.

    One prefill instance spans 2 nodes (nnodes=2) and one decode instance
    spans 2 nodes (nnodes=2), i.e. 4 nodes in total.
    """

    # Run evalscope only once (no retry on accuracy miss)
    max_retries = 1
    # Baseline aligned with the GLM-5.2 PD case; calibrate on GLM-5.3.
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

    @classmethod
    def setUpClass(cls):
        tent_conf_path = write_memfabric_tent_conf()
        common_envs = {
            **GLM_5_3_PD_SEP_COMMON_ENVS,
            "MC_TENT_CONF": tent_conf_path,
            "LD_LIBRARY_PATH": get_mooncake_ld_library_path(),
        }
        cls.model_config = {
            "model_path": GLM_5_3_W8A8_MODEL_PATH,
            "prefill_args": list(GLM_5_3_PD_SEP_PREFILL_ARGS),
            "decode_args": list(GLM_5_3_PD_SEP_DECODE_ARGS),
            "prefill_envs": {**common_envs, **GLM_5_3_PD_SEP_PREFILL_ENVS},
            "decode_envs": {**common_envs, **GLM_5_3_PD_SEP_DECODE_ENVS},
            "router_args": list(GLM_5_3_PD_SEP_ROUTER_ARGS),
            "router_envs": {},
        }
        super().setUpClass()

    def test_npu_glm_5_3_w8a8_1p1d_gpqa(self):
        """Run NPU accuracy test for GLM-5.3-w8a8 PD separation 1p1d on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
