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
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

GLM_5_2_W8A8_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # L2 HiCache: memfabric host-memory backend
    "SGLANG_HICACHE_HOST_MEM_BACKEND": "memfabric",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_W8A8_PD_SEP_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # L2 HiCache: memfabric host-memory backend
    "SGLANG_HICACHE_HOST_MEM_BACKEND": "memfabric",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "TASK_QUEUE_ENABLE": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_2_W8A8_PD_SEP_PREFILL_ARGS = [
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
    0.8,
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
    "glm-5.2-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    # CPP: chunked prefill
    "--chunked-prefill-size",
    4096,
    "--max-prefill-tokens",
    180000,
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
    # L2 HiCache: hierarchical cache (L1 device / L2 host)
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--enable-cache-report",
    "--hicache-size",
    100,
]

GLM_5_2_W8A8_PD_SEP_DECODE_ARGS = [
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
    0.76,
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
    # L2 HiCache: hierarchical cache (L1 device / L2 host)
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--enable-cache-report",
    "--hicache-size",
    100,
]

GLM_5_2_W8A8_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_2_W8A8_MODEL_PATH,
    "prefill_args": GLM_5_2_W8A8_PD_SEP_PREFILL_ARGS,
    "decode_args": GLM_5_2_W8A8_PD_SEP_DECODE_ARGS,
    "prefill_envs": GLM_5_2_W8A8_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_2_W8A8_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "round_robin", "--request-timeout-secs", 7200],
    "router_envs": {},
}


class TestNPUGLM_5_2_W8A8_PD_SEP_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for GLM-5.2-w8a8 PD separation on gpqa_diamond"""

    model_config = GLM_5_2_W8A8_PD_SEP_MODEL_CONFIG
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    eval_batch_size = 32
    generation_config = {
        "max_tokens": 131072,
        "top_p": 0.95,
        "temperature": 1.0,
        "timeout": 7200,
        "retries": 2,
        "stream": True,
    }

    def test_npu_glm_5_2_w8a8_pd_sep_gpqa(self):
        """Run NPU accuracy test for GLM-5.2-w8a8 PD separation on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()