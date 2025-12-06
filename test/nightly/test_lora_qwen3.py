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
import sys
import unittest
from pathlib import Path

# Add test directory to path for lora_utils import
# TODO: can be removed after migration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lora_utils import (
    LoRAAdaptor,
    LoRAModelCase,
    run_lora_multiple_batch_on_model_cases,
)

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=97, suite="nightly-1-gpu", nightly=True)

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


class TestLoRAQwen3(CustomTestCase):
    def test_ci_lora_models(self):
        run_lora_multiple_batch_on_model_cases(LORA_MODELS_QWEN3)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
