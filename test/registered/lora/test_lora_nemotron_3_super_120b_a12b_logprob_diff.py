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
Regression test for NVIDIA-Nemotron-3-Super-120B-A12B-BF16 LoRA logprob accuracy.

Compares SGLang LoRA logprobs against reference training logprobs from a
pre-computed dataset. The LoRA adapter and reference data are downloaded from:
https://huggingface.co/datasets/opherlie/lora-test-case-NVIDIA-Nemotron-3-Super-120B-A12B-BF16

Usage:
    python -m unittest test_lora_nemotron_3_super_120b_a12b_logprob_diff
"""

import multiprocessing as mp
import unittest

import torch
from huggingface_hub import snapshot_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import assert_lora_matches_trainer_logprobs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=154, stage="extra-b", runner_config="4-gpu-b200")

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
LORA_HF_REPO = "opherlie/lora-test-case-NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
LORA_BACKEND = "triton"
MAX_LORA_RANK = 64
TP_SIZE = 4
MOE_RUNNER_BACKEND = "triton"
EXPERTS_SHARED_OUTER_LORAS = True
LORA_USE_VIRTUAL_EXPERTS = True
DISABLE_SHARED_EXPERTS_FUSION = True

KL_THRESHOLD = 2.5e-3


class TestLoRANemotron3Super120B_A12B_LogprobDiff(CustomTestCase):
    def test_lora_nemotron_3_super_120b_a12b_logprob_accuracy(self):
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
            moe_runner_backend=MOE_RUNNER_BACKEND,
            experts_shared_outer_loras=EXPERTS_SHARED_OUTER_LORAS,
            lora_use_virtual_experts=LORA_USE_VIRTUAL_EXPERTS,
            disable_shared_experts_fusion=DISABLE_SHARED_EXPERTS_FUSION,
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
