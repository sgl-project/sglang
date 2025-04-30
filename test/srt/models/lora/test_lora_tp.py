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
import os
import unittest
from typing import List

from utils import (
    ALL_OTHER_LORA_MODELS,
    CI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    run_lora_test_one_by_one,
)

from sglang.test.test_utils import CustomTestCase, is_in_ci


class TestLoRATP(CustomTestCase):

    def _run_tp_on_model_cases(self, model_cases: List[LoRAModelCase]):
        tp_list = [2]  # Define TP sizes to iterate over
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
            for tp_size in tp_list:
                model_case.tp_size = tp_size
                for torch_dtype in TORCH_DTYPES:
                    run_lora_test_one_by_one(
                        prompts,
                        model_case,
                        torch_dtype,
                        max_new_tokens=32,
                        backend="triton",
                        test_tag=f"tp={tp_size}",
                    )

    def test_ci_lora_models(self):
        self._run_tp_on_model_cases(CI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_tp_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
