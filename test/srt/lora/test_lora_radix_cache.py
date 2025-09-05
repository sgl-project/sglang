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
import random
import unittest

import torch
from utils import CI_MULTI_LORA_MODELS, DEFAULT_PROMPTS, run_lora_test_one_by_one

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase

PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids.
    ### Question:
    What do you know about llamas?
    ### Answer:
    """,
]


class TestLoRARadixCache(CustomTestCase):

    def test_lora_radix_cache(self):
        # Here we need a model case with multiple adaptors for testing correctness of radix cache
        model_case = CI_MULTI_LORA_MODELS[0]

        torch_dtype = torch.float16
        max_new_tokens = 32
        backend = "triton"
        batch_prompts = (
            PROMPTS
            if not model_case.skip_long_prompt
            else [p for p in PROMPTS if len(p) < 1000]
        )

        # Test lora with radix cache
        run_lora_test_one_by_one(
            batch_prompts,
            model_case,
            torch_dtype,
            max_new_tokens=max_new_tokens,
            backend=backend,
            disable_radix_cache=False,
            test_tag="lora-with-radix-cache",
        )

        # Test lora without radix cache
        run_lora_test_one_by_one(
            batch_prompts,
            model_case,
            torch_dtype,
            max_new_tokens=max_new_tokens,
            backend=backend,
            disable_radix_cache=True,
            test_tag="lora-without-radix-cache",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
