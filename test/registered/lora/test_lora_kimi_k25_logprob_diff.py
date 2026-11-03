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
Regression test for Kimi-K2.5 (VLM + MLA + MoE) LoRA logprob accuracy.

Compares SGLang LoRA logprobs against reference training logprobs from a
pre-computed dataset. The LoRA adapter and reference data are downloaded from:
https://huggingface.co/datasets/yushengsu/lora-diff-Kimi-K2.5

Usage:
    python -m unittest test_lora_kimi_k25_logprob_diff
"""

import multiprocessing as mp
import unittest

import torch
from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import assert_lora_matches_trainer_logprobs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=420, stage="weekly", runner_config="8-gpu-b200")

BASE_MODEL = "moonshotai/Kimi-K2.5"
LORA_HF_REPO = "yushengsu/lora-diff-Kimi-K2.5"
LORA_BACKEND = "triton"
MAX_LORA_RANK = 32
TP_SIZE = 8
MOE_RUNNER_BACKEND = "triton"
EXPERTS_SHARED_OUTER_LORAS = True
PREFILL_ATTENTION_BACKEND = "fa4"
DECODE_ATTENTION_BACKEND = "flashinfer"

KL_THRESHOLD = 1.5e-2


class TestLoRAKimiK25LogprobDiff(CustomTestCase):
    def test_lora_kimi_k25_logprob_accuracy(self):
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
            attention_backend="flashinfer",
            moe_runner_backend=MOE_RUNNER_BACKEND,
            experts_shared_outer_loras=EXPERTS_SHARED_OUTER_LORAS,
            prefill_attention_backend=PREFILL_ATTENTION_BACKEND,
            decode_attention_backend=DECODE_ATTENTION_BACKEND,
            trust_remote_code=True,
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
