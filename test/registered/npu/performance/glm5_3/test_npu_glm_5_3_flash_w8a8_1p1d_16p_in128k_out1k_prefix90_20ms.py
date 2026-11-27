import importlib.util
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    GLM_5_3_FLASH_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
    write_memfabric_tent_conf,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MOONCAKE_INSTALLED = importlib.util.find_spec("mooncake") is not None
DISAGGREGATION_TRANSFER_BACKEND = "mooncake" if MOONCAKE_INSTALLED else "ascend"

# Mooncake + TENT + MemFabric, only used when the Mooncake package is present.
MOONCAKE_ENVS = {
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
}

# Common environment variables for GLM-5.3-Flash PD-Sep 128k1k deployment,
# aligned with the Wiki launch script (GLM5.3-FLASH-1P1D).
GLM_5_3_FLASH_PD_SEP_COMMON_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_FLASH_PD_SEP_PREFILL_ENVS = {
    **GLM_5_3_FLASH_PD_SEP_COMMON_ENVS,
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "128",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "HCCL_IF_BASE_PORT": "61000",
}

GLM_5_3_FLASH_PD_SEP_DECODE_ENVS = {
    **GLM_5_3_FLASH_PD_SEP_COMMON_ENVS,
    "HCCL_BUFFSIZE": "512",
    "DEEPEP_HCCL_BUFFSIZE": "512",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_IF_BASE_PORT": "61010",
}

# NEXTN (MTP) speculative arguments shared by the prefill and decode nodes,
# aligned with the Wiki launch script. The MTP draft layer is W8A8_DYNAMIC in
# the checkpoint; do NOT set --speculative-draft-model-quantization unquant —
# the draft inherits --quantization modelslim by default.
GLM_5_3_FLASH_PD_SEP_SPECULATIVE_ARGS = [
    "--speculative-draft-model-path",
    GLM_5_3_FLASH_W8A8_MODEL_PATH,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
]

GLM_5_3_FLASH_PD_SEP_PREFILL_ARGS = [
    "--quantization",
    "modelslim",
    "--device",
    "npu",
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    DISAGGREGATION_TRANSFER_BACKEND,
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
    128,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    16384,
    "--max-prefill-tokens",
    16384,
    "--watchdog-timeout",
    9000,
    # NPU DSA KPool widens the logical page to 512, so the mamba track
    # interval must be a multiple of 512 to land on the radix grid.
    "--mamba-track-interval",
    512,
    "--disable-radix-cache",
    *GLM_5_3_FLASH_PD_SEP_SPECULATIVE_ARGS,
]

GLM_5_3_FLASH_PD_SEP_DECODE_ARGS = [
    "--quantization",
    "modelslim",
    "--device",
    "npu",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    DISAGGREGATION_TRANSFER_BACKEND,
    "--disaggregation-decode-enable-radix-cache",
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
    "--attn-dp-size",
    16,
    "--nnodes",
    1,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--load-balance-method",
    "round_robin",
    "--mem-fraction-static",
    0.80,
    "--page-size",
    128,
    "--max-running-requests",
    64,
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    "--watchdog-timeout",
    9000,
    # NPU DSA KPool widens the logical page to 512, so the mamba track
    # interval must be a multiple of 512 to land on the radix grid.
    "--mamba-track-interval",
    512,
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-mem-layout",
    "page_first",
    *GLM_5_3_FLASH_PD_SEP_SPECULATIVE_ARGS,
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

    @classmethod
    def setUpClass(cls):
        common_envs = dict(GLM_5_3_FLASH_PD_SEP_COMMON_ENVS)
        if MOONCAKE_INSTALLED:
            common_envs.update(MOONCAKE_ENVS)
            common_envs["MC_TENT_CONF"] = write_memfabric_tent_conf()
            common_envs["LD_LIBRARY_PATH"] = append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            )
        cls.model_config = {
            **GLM_5_3_FLASH_PD_SEP_MODEL_CONFIG,
            "prefill_args": list(GLM_5_3_FLASH_PD_SEP_PREFILL_ARGS),
            "decode_args": list(GLM_5_3_FLASH_PD_SEP_DECODE_ARGS),
            "prefill_envs": {
                **common_envs,
                **GLM_5_3_FLASH_PD_SEP_PREFILL_ENVS,
            },
            "decode_envs": {
                **common_envs,
                **GLM_5_3_FLASH_PD_SEP_DECODE_ENVS,
            },
        }
        super().setUpClass()

    def test_npu_glm_5_3_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms(self):
        """Run NPU performance test for GLM-5.3-Flash-w8a8 PD separation 1p1d 16p"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
