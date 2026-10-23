import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Pure-prefill acceptance scenario for Kimi-K3 W4A8 32P (4-node mix):
# cold 131072 in / 1 out, C1/N1, mean TTFT <= 20 s.
#
# The server config reproduces the dev CPP acceptance environment
# (prefill_prefix.sh CPP mode, the topology that measured 18.2 s on 128k
# cold prefill): PP4 x TP16 x DP2 with layer partition 24,24,24,21,
# chunked-prefill 16384 (8192 per DP), no DSPARK, radix disabled, decode
# graphs off, overlap schedule off. Do NOT merge this config with the
# decode scenario: run 39917316049 crashed a DSPARK + decode-graph server
# when the pure-prefill memory tuning was applied to it - the scenarios
# need mutually exclusive configs and separate cases. The mixed TP64/DP4
# topology is architecturally slower for single-stream 128k prefill
# (2246 tok/s run 39882825354, 1336 tok/s run 39944680878, vs 7188 tok/s
# here) - knobs cannot close a 3.2x topology gap.
register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

KIMI_K3_W4A8_32P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "HCCL_BUFFSIZE": "2000",
    # Dev CPP reference (aisbench_prefill_20261017): ROUND2/PER512, FIA on,
    # and the 4-stage PP layer partition.
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "2",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ASCEND_USE_FIA": "1",
    "SGLANG_NPU_USE_FIAS_V2_BSND": "1",
    "SGLANG_PP_LAYER_PARTITION": "24,24,24,21",
}

KIMI_K3_W4A8_32P_PREFILL_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--nnodes",
    4,
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
    # CPP topology (TP16 x PP4 x DP2, layer partition 24,24,24,21) - the
    # dev acceptance configuration in which 128k cold prefill measured
    # 18.2 s. The mixed TP64/DP4 topology is architecturally slower for a
    # single-stream 128k prefill (one DP attention domain lights up):
    # 2246 tok/s with the decode-tier extras (run 39882825354) and 1336
    # tok/s with the pure-prefill tuning (run 39944680878) - knobs cannot
    # close a 3.2x topology gap.
    "--tp-size",
    16,
    "--pp-size",
    4,
    "--pp-async-batch-depth",
    0,
    "--disable-overlap-schedule",
    "--enable-dp-attention",
    "--dp-size",
    2,
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    "--mem-fraction-static",
    0.78,
    "--chunked-prefill-size",
    16384,
    "--max-total-tokens",
    250000,
    "--disable-radix-cache",
    "--cuda-graph-backend-decode",
    "disabled",
    "--max-running-requests",
    8,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_PREFILL_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn128kOut1Prefill(TestNpuPerfMultiNodePdMixTestCaseBase):
    """Pure prefill: cold 131072 in / 1 out, C1/N1, mean TTFT <= 20 s."""

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the exact-length GSM8K dataset;
    # the base-class default would silently fall back to the raw ShareGPT file.
    dataset_path = None
    input_len = 131072
    output_len = 1
    num_prompts = 1
    max_concurrency = 1
    random_range_ratio = 1
    request_rate = 0
    ttft = 20000.0

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_out1_prefill(self):
        """Run the 128k cold-prefill TTFT acceptance test."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
