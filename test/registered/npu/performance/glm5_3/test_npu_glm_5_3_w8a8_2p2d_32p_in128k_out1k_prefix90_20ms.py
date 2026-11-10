import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_3_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression, record_pass1
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=10800,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

CASE_ID = "test_npu_glm_5_3_w8a8_2p2d_32p_in128k_out1k_prefix90_20ms"

GLM_5_3_PD_SEP_PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
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
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
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

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
GLM_5_3_PD_SEP_MODEL_CONFIG_PASS1 = {
    **GLM_5_3_PD_SEP_MODEL_CONFIG,
    # launch_pd_separation_node extends the arg lists in place on each pd pod;
    # pass 1 needs its own copies or pass 2 boots with pass 1's rank/addr flags.
    "prefill_args": list(GLM_5_3_PD_SEP_PREFILL_ARGS),
    "decode_args": list(GLM_5_3_PD_SEP_DECODE_ARGS),
    "prefill_envs": {
        **{
            k: v
            for k, v in GLM_5_3_PD_SEP_PREFILL_ENVS.items()
            if k != "ASCEND_USE_FIA"
        },
        "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
    },
    "decode_envs": {
        **{
            k: v for k, v in GLM_5_3_PD_SEP_DECODE_ENVS.items() if k != "ASCEND_USE_FIA"
        },
        "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
    },
}


class GLM5_3W8A8_2P2D_32PIn128kOut1kPrefix90_20msBase(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Shared config for the two-pass (FIA unset vs FIA=0) throughput comparison. GLM-5.3-w8a8 PD separation 2p2d in128k out1k prefix90"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_name = "generated-shared-prefix"
    max_concurrency = 64
    num_prompts = 256
    input_len = 131072
    output_len = 1024
    repeat_rate = 0.9
    dp = 32
    # tpot baseline commented out; +/-2% comparison is the gate
    # output_token_throughput baseline commented out; +/-2% comparison is the gate


class TestNPUGLM5_3_W8A8_2P2D_32P_In128k_Out1k_Prefix90_20msPass1(
    GLM5_3W8A8_2P2D_32PIn128kOut1kPrefix90_20msBase
):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    model_config = GLM_5_3_PD_SEP_MODEL_CONFIG_PASS1

    def test_npu_glm_5_3_w8a8_2p2d_32p_in128k_out1k_prefix90_20ms_pass1(self):
        """Run pass-1 throughput and record the baseline metrics."""
        self.run_throughput()
        record_pass1(CASE_ID, self.last_metrics)


class TestNPUGLM5_3_W8A8_2P2D_32P_In128k_Out1k_Prefix90_20msPass2(
    GLM5_3W8A8_2P2D_32PIn128kOut1kPrefix90_20msBase
):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    model_config = GLM_5_3_PD_SEP_MODEL_CONFIG

    def test_npu_glm_5_3_w8a8_2p2d_32p_in128k_out1k_prefix90_20ms_pass2(self):
        """Run pass-2 throughput and assert no regression vs pass 1."""
        self.run_throughput()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
