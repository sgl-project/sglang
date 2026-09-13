import unittest

import torch

from sglang.srt.utils.common import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

# The SM100 CI pool has no single-GPU runner. Use all four GPUs in the B200
# runner so the declared resource and the server topology stay aligned.
register_cuda_ci(
    est_time=900,
    stage="extra-b",
    runner_config="4-gpu-b200",
)

MODEL = "Qwen/Qwen3.5-9B"
GSM8K_ACCURACY_THRESHOLD = 0.70
GSM8K_NUM_QUESTIONS = 200
GSM8K_NUM_SHOTS = 8

COMMON_ARGS = [
    "--tp-size",
    "4",
    "--kv-cache-dtype",
    "nvfp4",
    "--page-size",
    "16",
    "--max-total-tokens",
    "131072",
    "--max-running-requests",
    "64",
]
MTP_ARGS = [
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

HAS_FOUR_SM100_GPUS = is_sm100_supported() and torch.cuda.device_count() >= 4


@unittest.skipUnless(HAS_FOUR_SM100_GPUS, "requires 4 SM100 GPUs with CUDA 12.8+")
class TestQwen35NVFP4KVNativePrefillSM100(GSM8KMixin, DefaultServerBase):
    """Native NVFP4 prefill and decode without MTP."""

    model = MODEL
    gsm8k_score_threshold = GSM8K_ACCURACY_THRESHOLD
    gsm8k_num_examples = GSM8K_NUM_QUESTIONS
    gsm8k_num_threads = 128
    gsm8k_num_shots = GSM8K_NUM_SHOTS
    other_args = COMMON_ARGS + ["--prefill-kv-cache-dequant-dtype", "nvfp4"]


@unittest.skipUnless(HAS_FOUR_SM100_GPUS, "requires 4 SM100 GPUs with CUDA 12.8+")
class TestQwen35NVFP4KVDQPrefillSM100(GSM8KMixin, DefaultServerBase):
    """FP8-dequantized prefill and native NVFP4 decode without MTP."""

    model = MODEL
    gsm8k_score_threshold = GSM8K_ACCURACY_THRESHOLD
    gsm8k_num_examples = GSM8K_NUM_QUESTIONS
    gsm8k_num_threads = 128
    gsm8k_num_shots = GSM8K_NUM_SHOTS
    other_args = COMMON_ARGS + ["--prefill-kv-cache-dequant-dtype", "fp8_e4m3"]


@unittest.skipUnless(HAS_FOUR_SM100_GPUS, "requires 4 SM100 GPUs with CUDA 12.8+")
class TestQwen35NVFP4KVNativePrefillMTPSM100(GSM8KMixin, DefaultServerBase):
    """Native NVFP4 prefill and decode with NEXTN MTP."""

    model = MODEL
    gsm8k_score_threshold = GSM8K_ACCURACY_THRESHOLD
    gsm8k_num_examples = GSM8K_NUM_QUESTIONS
    gsm8k_num_threads = 128
    gsm8k_num_shots = GSM8K_NUM_SHOTS
    gsm8k_accept_length_thres = 1.2
    other_args = COMMON_ARGS + ["--prefill-kv-cache-dequant-dtype", "nvfp4"] + MTP_ARGS


@unittest.skipUnless(HAS_FOUR_SM100_GPUS, "requires 4 SM100 GPUs with CUDA 12.8+")
class TestQwen35NVFP4KVDQPrefillMTPSM100(GSM8KMixin, DefaultServerBase):
    """FP8-dequantized prefill and native NVFP4 decode with NEXTN MTP."""

    model = MODEL
    gsm8k_score_threshold = GSM8K_ACCURACY_THRESHOLD
    gsm8k_num_examples = GSM8K_NUM_QUESTIONS
    gsm8k_num_threads = 128
    gsm8k_num_shots = GSM8K_NUM_SHOTS
    gsm8k_accept_length_thres = 1.2
    other_args = (
        COMMON_ARGS + ["--prefill-kv-cache-dequant-dtype", "fp8_e4m3"] + MTP_ARGS
    )


if __name__ == "__main__":
    unittest.main()
