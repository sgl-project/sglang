import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestAscendAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.test_ascend_utils import (
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    KIMI_K2_5_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-pd-sep-4-node",
    nightly=True,
    disabled="accuracy testcase",
)

KIMI_K2_5_W4A8_PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "100",
}

KIMI_K2_5_W4A8_DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "2400",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
}

KIMI_K2_5_W4A8_PREFILL_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.76,
    "--max-running-requests",
    8,
    "--chunked-prefill-size",
    16384,
    "--context-length",
    260000,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    4,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
]

KIMI_K2_5_W4A8_DECODE_ARGS = [
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    "2",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    32,
    "--mem-fraction-static",
    0.67,
    "--max-running-requests",
    32,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    65536,
    "--context-length",
    260000,
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
    "--sampling-backend",
    "ascend",
    "--enable-dp-attention",
    "--dp-size",
    32,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs",
    1,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    KIMI_K2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    1,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    2,
    "--speculative-draft-model-quantization",
    "unquant",
]

MODEL_CONFIG = {
    "model_path": KIMI_K2_5_W4A8_MODEL_PATH,
    "prefill_args": KIMI_K2_5_W4A8_PREFILL_ARGS,
    "decode_args": KIMI_K2_5_W4A8_DECODE_ARGS,
    "prefill_envs": KIMI_K2_5_W4A8_PREFILL_ENVS,
    "decode_envs": KIMI_K2_5_W4A8_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}


class TestNPUKimiK2_5_W4A8_2P1D_64P_GPQA(TestAscendAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for Kimi-K2.5-w4a8 2p1d_64p on GPQA"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    accuracy = 0.8
    dataset_type = "gpqa"
    dataset_name = "gpqa_gen_0_shot_cot_chat_prompt"
    max_concurrency = 128
    generation_kwargs = "dict(temperature=1.0, top_p=0.95)"
    output_len = 256000

    def test_npu_kimi_k2_5_w4a8_2p1d_64p_gpqa(self):
        """Run NPU accuracy test for Kimi-K2.5-w4a8 2p1d_64p on GPQA"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
