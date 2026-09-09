import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K3_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

KIMI_K3_W4A8_2P4D_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE": "1",
    "SGLANG_K3_SHARED_EXPERTS_ATTN_TP": "1",
    "SGLANG_K3_DENSE_MLP_ATTN_TP": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "HCCL_BUFFSIZE": "800",
    # "SGLANG_PP_LAYER_PARTITION": "48,45",
}

KIMI_K3_W4A8_2P4D_DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE": "1",
    "SGLANG_K3_SHARED_EXPERTS_ATTN_TP": "1",
    "SGLANG_K3_DENSE_MLP_ATTN_TP": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "HCCL_BUFFSIZE": "1200",
}

KIMI_K3_W4A8_2P4D_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--tp-size",
    32,
    "--pp-size",
    1,
    "--dp-size",
    2,
    "--nnodes",
    2,
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--tokenizer-path",
    KIMI_K3_W4A8_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--disable-custom-all-reduce",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    0.85,
    "--chunked-prefill-size",
    4096,
    "--max-running-requests",
    16,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_2P4D_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    64,
    "--pp-size",
    1,
    "--dp-size",
    4,
    "--nnodes",
    4,
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--tokenizer-path",
    KIMI_K3_W4A8_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--disable-custom-all-reduce",
    "--cuda-graph-bs",
    16,
    "--mem-fraction-static",
    0.82,
    "--max-running-requests",
    16,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--linear-attn-verify-backend",
    "triton",
    "--disaggregation-transfer-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_2P4D_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "prefill_args": KIMI_K3_W4A8_2P4D_PREFILL_ARGS,
    "decode_args": KIMI_K3_W4A8_2P4D_DECODE_ARGS,
    "prefill_envs": KIMI_K3_W4A8_2P4D_PREFILL_ENVS,
    "decode_envs": KIMI_K3_W4A8_2P4D_DECODE_ENVS,
    "router_args": ["--policy", "round_robin", "--request-timeout-secs", 7200],
    "router_envs": {},
}


class TestNPUKimiK3_W4A8_2P4D_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for Kimi-K3-w4a8 PD separation 2p4d on gpqa_diamond"""

    model_config = KIMI_K3_W4A8_2P4D_MODEL_CONFIG
    accuracy = 0.935
    datasets = ["gpqa_diamond"]
    eval_batch_size = 32
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "max"},
    }
    timeout = 10000

    def test_npu_kimi_k3_w4a8_2p4d_gpqa(self):
        """Run NPU accuracy test for Kimi-K3-w4a8 PD separation 2p4d on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
