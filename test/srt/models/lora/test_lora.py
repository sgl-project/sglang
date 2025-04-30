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

import multiprocessing as mp
import unittest

from utils import TORCH_DTYPES, LoRAAdaptor, LoRAModelCase, run_lora_test_by_batch

from sglang.test.test_utils import CustomTestCase

PROMPTS = [
    """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
""",
    "AI is a field of computer science focused on",
]

LORA_MODELS_WITH_NONE = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
            LoRAAdaptor(
                name=None,
            ),
        ],
        max_loras_per_batch=2,
    ),
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name=None,
            ),
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
        ],
        max_loras_per_batch=2,
    ),
]


class TestLoRA(CustomTestCase):
    def test_lora_batch_with_none(self):
        for model_case in LORA_MODELS_WITH_NONE:
            prompts = PROMPTS
            for torch_dtype in TORCH_DTYPES:
                run_lora_test_by_batch(
                    prompts,
                    model_case,
                    torch_dtype,
                    max_new_tokens=32,
                    backend="triton",
                    test_tag="test_lora_batch_with_none",
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
