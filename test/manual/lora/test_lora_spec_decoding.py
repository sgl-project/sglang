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

import multiprocessing as mp
import unittest

from lora_utils import (
    CI_MULTI_LORA_MODELS,
    LoRAAdaptor,
    LoRAModelCase,
    run_lora_multiple_batch_on_model_cases,
)

from sglang.test.test_utils import CustomTestCase

LORA_MODELS_QWEN3 = [
    LoRAModelCase(
        base="Qwen/Qwen3-4B",
        adaptors=[
            LoRAAdaptor(
                name="nissenj/Qwen3-4B-lora-v2",
                prefill_tolerance=3e-1,
            ),
            LoRAAdaptor(
                name="y9760210/Qwen3-4B-lora_model",
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=2,
    ),
]


class TestLoRASpecDecoding(CustomTestCase):
    def test_qwen(self):
        run_lora_multiple_batch_on_model_cases(
            LORA_MODELS_QWEN3,
            attention_backend="triton",
            use_spec_decoding=True,
            disable_cuda_graph=True,
            enable_deterministic_inference=True,
        )

    def test_llama(self):
        run_lora_multiple_batch_on_model_cases(
            CI_MULTI_LORA_MODELS,
            attention_backend="triton",
            use_spec_decoding=True,
            disable_cuda_graph=True,
            enable_deterministic_inference=True,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
