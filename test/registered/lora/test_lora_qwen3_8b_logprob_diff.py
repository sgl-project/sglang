# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Regression test for Qwen3-8B LoRA logprob accuracy.

Compares SGLang LoRA logprobs against reference training logprobs from a
pre-computed dataset. The LoRA adapter and reference data are downloaded from:
https://huggingface.co/datasets/yushengsu/lora-diff-Qwen3-8B

Usage:
    python -m unittest test_lora_qwen3_8b_logprob_diff
"""

import multiprocessing as mp
import unittest

import torch
from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import assert_lora_matches_trainer_logprobs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=51, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=120, suite="extra-a-test-1-gpu-large-amd")

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_HF_REPO = "yushengsu/lora-diff-Qwen3-8B"
LORA_BACKEND = "triton"
MAX_LORA_RANK = 32
TP_SIZE = 1
PREFILL_ATTENTION_BACKEND = "fa4"
DECODE_ATTENTION_BACKEND = "fa4"

KL_THRESHOLD = 5e-3


def attention_backend_kwargs():
    """Engine attention-backend kwargs for the current platform.

    fa4 and flashinfer are CUDA-only: fa4 dispatches into the CUTLASS CUTE DSL
    kernel, which cannot import off CUDA. Use Triton on ROCm and the equivalent
    intel_xpu fused path on XPU.
    """
    from sglang.srt.utils import is_hip, is_xpu

    if is_xpu():
        return {"attention_backend": "intel_xpu"}
    if is_hip():
        return {
            "attention_backend": "triton",
            "prefill_attention_backend": "triton",
            "decode_attention_backend": "triton",
        }
    return {
        "attention_backend": "flashinfer",
        "prefill_attention_backend": PREFILL_ATTENTION_BACKEND,
        "decode_attention_backend": DECODE_ATTENTION_BACKEND,
    }


class TestLoRAQwen3_8BLogprobDiff(CustomTestCase):
    def test_lora_qwen3_8b_logprob_accuracy(self):
        adapter_path = snapshot_download(
            LORA_HF_REPO,
            repo_type="dataset",
        )

        engine = sgl.Engine(
            model_path=BASE_MODEL,
            tp_size=TP_SIZE,
            enable_lora=True,
            max_lora_rank=MAX_LORA_RANK,
            lora_paths={"my_lora": adapter_path},
            lora_backend=LORA_BACKEND,
            **attention_backend_kwargs(),
        )

        try:
            assert_lora_matches_trainer_logprobs(
                self, engine, adapter_path, KL_THRESHOLD
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
