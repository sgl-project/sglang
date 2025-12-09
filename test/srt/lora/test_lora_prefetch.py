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
import sys
import unittest
from pathlib import Path

import torch

# Add test directory to path for lora_utils import
# TODO: can be removed after migration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lora_utils import DEFAULT_PROMPTS, ensure_reproducibility

from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTERS = [
    "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
    "philschmid/code-llama-3-1-8b-text-to-sql-lora",
    "pbevan11/llama-3.1-8b-ocr-correction",
]


class TestLoRAPrefetch(CustomTestCase):
    def test_lora_prefetch_basic(self):
        base_runner_kwargs = {
            "model_path": BASE_MODEL,
            "torch_dtype": torch.float16,
            "model_type": "generation",
            "lora_paths": ADAPTERS,
            "max_loras_per_batch": 2,
        }
        max_new_tokens = 256

        ensure_reproducibility()

        with SRTRunner(**base_runner_kwargs, max_loras_prefetch=0) as runner:
            baseline_output = runner.forward(
                DEFAULT_PROMPTS,
                max_new_tokens=max_new_tokens,
                lora_paths=ADAPTERS,
            ).output_ids

        with SRTRunner(**base_runner_kwargs, max_loras_prefetch=2) as runner:
            prefetch_output = runner.forward(
                DEFAULT_PROMPTS, max_new_tokens=max_new_tokens, lora_paths=ADAPTERS
            ).output_ids

        self.assertEqual(baseline_output, prefetch_output)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
