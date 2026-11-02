# Copyright 2023-2024 SGLang Team
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

"""TP-sharded LoRA A/B weights must reproduce the HF adapter outputs."""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TORCH_DTYPES,
    run_lora_test_one_by_one,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=115, stage="base-b", runner_config="2-gpu-large")
# backend-specific: ROCm's Triton build of the chunked-SGMV LoRA kernels on
# TP-sharded adapter weights.
register_amd_ci(
    est_time=58,
    suite="stage-b-test-2-gpu-large-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/13107; until 2026-12-31",
)


class TestLoRATP(CustomTestCase):
    def test_ci_lora_models(self):
        for model_case in CI_LORA_MODELS:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
            model_case.tp_size = 2
            for torch_dtype in TORCH_DTYPES:
                run_lora_test_one_by_one(
                    prompts,
                    model_case,
                    torch_dtype,
                    max_new_tokens=32,
                    test_tag="tp=2",
                    attention_backend="fa3",
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
