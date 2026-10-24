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

"""Mixed-adapter batches, split across LoRA slots, must match HF row by row."""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    TEST_MULTIPLE_BATCH_PROMPTS,
    TORCH_DTYPES,
    assert_lora_routes_match_hf,
    create_multiple_batch_test_samples,
    lora_split_batches,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=65, stage="base-b", runner_config="1-gpu-large")
# backend-specific: ROCm's Triton build of the chunked-SGMV LoRA kernels on
# mixed-adapter batches.
register_amd_ci(est_time=65, suite="stage-b-test-1-gpu-small-amd")


class TestMultiLoRABackend(CustomTestCase):
    def test_ci_lora_models_multi_batch(self):
        for model_case in CI_MULTI_LORA_MODELS:
            for torch_dtype in TORCH_DTYPES:
                a, b = [adaptor.name for adaptor in model_case.adaptors]
                batches = create_multiple_batch_test_samples(
                    TEST_MULTIPLE_BATCH_PROMPTS, [a, b]
                ) + lora_split_batches(a, b)
                assert_lora_routes_match_hf(self, model_case, torch_dtype, batches)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
