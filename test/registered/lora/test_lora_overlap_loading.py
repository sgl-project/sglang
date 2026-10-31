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

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    TORCH_DTYPES,
    assert_lora_routes_match_hf,
    lora_split_batches,
)
from sglang.test.test_utils import CustomTestCase

# Overlap loading is opt-in (--enable-lora-overlap-loading); the loader's
# bookkeeping is unit-tested in unit/lora/test_lora_overlap_loader_unit.py.
register_cuda_ci(est_time=225, stage="extra-a", runner_config="1-gpu-large")
# backend-specific: the HIP stream and event path behind the overlap loader's
# event.query() gate on the ROCm build.
register_amd_ci(est_time=75, suite="stage-b-test-1-gpu-small-amd")


class TestLoRAOverlapLoading(CustomTestCase):
    def test_ci_lora_models_batch_splitting(self):
        """Async loads under slot pressure must match independent HF rows,
        rejecting wrong-adapter or half-copied weights."""
        for model_case in CI_MULTI_LORA_MODELS:
            a, b = [adaptor.name for adaptor in model_case.adaptors]
            for torch_dtype in TORCH_DTYPES:
                assert_lora_routes_match_hf(
                    self,
                    model_case,
                    torch_dtype,
                    lora_split_batches(a, b),
                    enable_lora_overlap_loading=True,
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
