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

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    LORA_MODELS_QWEN3,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(
    est_time=30,
    suite="stage-b-test-small-1-gpu-amd",
    disabled="see https://github.com/sgl-project/sglang/issues/13107",
)
register_cuda_ci(est_time=97, suite="nightly-1-gpu", nightly=True)


class TestLoRAQwen3(CustomTestCase):
    def test_ci_lora_models(self):
        run_lora_multiple_batch_on_model_cases(LORA_MODELS_QWEN3)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
