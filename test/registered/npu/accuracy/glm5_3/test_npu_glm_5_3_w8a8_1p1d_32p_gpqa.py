import logging
import os
import unittest
from datetime import datetime, timedelta, timezone

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

register_npu_ci(
    est_time=4800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# The DCP and PCP deployments only differ in the context-parallel flags
# (decode --dcp-size vs prefill --attn-cp-size), so they share one nightly
# 4-node slot and alternate by date instead of running as two separate jobs.
# Export NPU_ACCURACY_VARIANT=dcp|pcp on every pod to pin a variant manually.
ACCURACY_VARIANT_ENV = "NPU_ACCURACY_VARIANT"
ACCURACY_VARIANTS = ("dcp", "pcp")


def resolve_accuracy_variant():
    requested = os.environ.get(ACCURACY_VARIANT_ENV, "").strip().lower()
    if requested:
        if requested not in ACCURACY_VARIANTS:
            raise ValueError(
                f"Unsupported {ACCURACY_VARIANT_ENV}={requested}, "
                f"expected one of {ACCURACY_VARIANTS}."
            )
        logger.info(f"[accuracy-variant] pinned by {ACCURACY_VARIANT_ENV}: {requested}")
        return requested

    # Follow the nightly schedule: the cron fires at 16:00 UTC, i.e. 00:00 the
    # next day in Beijing. Odd dates run DCP, even dates run PCP.
    beijing_day = datetime.now(timezone(timedelta(hours=8))).day
    accuracy_variant = "dcp" if beijing_day % 2 else "pcp"
    logger.info(
        f"[accuracy-variant] auto-rotated by date (Beijing day={beijing_day}): {accuracy_variant}"
    )
    return accuracy_variant


GLM_5_3_PD_SEP_PREFILL_ENVS = {
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

GLM_5_3_PD_SEP_DECODE_ENVS = {
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


def build_prefill_args(variant):
    args = [
        "--disaggregation-mode",
        "prefill",
        "--nnodes",
        2,
        "--tp-size",
        8,
    ]
    if variant == "dcp":
        args += [
            "--ep-size",
            8,
            "--attn-dp-size",
            8,
        ]
    else:
        args += [
            "--attn-cp-size",
            8,
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
        ]
    args += [
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
        "--disable-decode-cuda-graph",
        "--disable-prefill-cuda-graph",
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
        # EPLB is temporarily disabled: the community build rejects EPLB when
        # pipeline parallelism is used (pp-size=4 here). Re-enable once the
        # upstream config conflict is resolved.
        # "--enable-eplb",
        # "--expert-distribution-recorder-mode",
        # "stat",
        # "--ep-num-redundant-experts",
        # 16,
        # "--eplb-rebalance-num-iterations",
        # 100000,
        # "--expert-distribution-recorder-buffer-size",
        # 50,
    ]
    return args


def build_decode_args(variant):
    args = [
        "--disaggregation-mode",
        "decode",
        "--nnodes",
        2,
        "--tp-size",
        32,
        "--attn-dp-size",
        8,
    ]
    if variant == "dcp":
        args += [
            "--dcp-size",
            4,
        ]
    args += [
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
        # EPLB is temporarily disabled: the community build rejects EPLB when
        # pipeline parallelism is used (pp-size=4 here). Re-enable once the
        # upstream config conflict is resolved.
        # "--enable-eplb",
        # "--expert-distribution-recorder-mode",
        # "stat",
        # "--ep-num-redundant-experts",
        # 32,
        # "--eplb-rebalance-num-iterations",
        # 100000,
        # "--expert-distribution-recorder-buffer-size",
        # 1000,
    ]
    return args


ACCURACY_VARIANT = resolve_accuracy_variant()

GLM_5_3_PD_SEP_MODEL_CONFIG = {
    "model_path": GLM_5_3_W8A8_MODEL_PATH,
    "prefill_args": build_prefill_args(ACCURACY_VARIANT),
    "decode_args": build_decode_args(ACCURACY_VARIANT),
    "prefill_envs": GLM_5_3_PD_SEP_PREFILL_ENVS,
    "decode_envs": GLM_5_3_PD_SEP_DECODE_ENVS,
    "router_args": [
        "--policy",
        "round_robin",
        "--request-timeout-secs",
        7200,
        "--disable-circuit-breaker",
    ],
    "router_envs": {},
}


# GLM-5.3-w8a8, 4-node PD separation (2 prefill + 2 decode), single slot that
# rotates the DCP/PCP context-parallel configuration by date.
# DCP: prefill tp8/ep8/dp8/pp4, decode tp32/dp8/dcp4/ep32.
# PCP: prefill tp8/attn-cp8 (interleave)/pp4, decode tp32/dp8/ep32.
# MoE A2A backend: ascend_ops (prefill=alltoallv, decode=low_latency) on both P and D
class TestNPUGLM5_3_W8A8_1P1D_32P_GPQA(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Test NPU accuracy for GLM-5.3-w8a8 PD separation 1p1d with DCP/PCP rotation on gpqa_diamond"""

    model_config = GLM_5_3_PD_SEP_MODEL_CONFIG
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

    def test_npu_glm_5_3_w8a8_1p1d_32p_gpqa(self):
        """Run NPU accuracy test for GLM-5.3-w8a8 PD separation 1p1d with DCP/PCP rotation on gpqa_diamond"""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
