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
"""
End-to-end tests for the --enable-lora-overlap-loading server argument.
"""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=100, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=100, suite="stage-b-test-small-1-gpu-amd")


class TestLoRAOverlapLoading(CustomTestCase):
    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS, enable_lora_overlap_loading=True
        )

    def test_ci_lora_models_multi_batch(self):
        run_lora_multiple_batch_on_model_cases(
            CI_MULTI_LORA_MODELS, enable_lora_overlap_loading=True
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
