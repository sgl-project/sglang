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
import re
import unittest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

if is_hip():
    os.environ.setdefault("SGLANG_USE_AITER", "0")

from sglang.test.lora_utils import (
    ALL_OTHER_MULTI_LORA_MODELS,
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase, is_in_amd_ci, is_in_ci

register_cuda_ci(est_time=100, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=200, suite="stage-b-test-small-1-gpu-amd-nondeterministic")


class TestMultiLoRABackend(CustomTestCase):
    def _callTestMethod(self, method):
        from sglang.srt.environ import envs
        from sglang.srt.utils.common import retry

        max_retry = envs.SGLANG_TEST_MAX_RETRY.get()
        if max_retry is None:
            max_retry = 1 if is_in_ci() else 0

        if is_in_amd_ci():
            attempt_count = [0]

            def should_retry(e):
                attempt_count[0] += 1
                match = re.search(r"ROUGE-L score ([\d.]+)", str(e))
                if match:
                    score = float(match.group(1))
                    if score < 0.977:
                        return False
                    return True
                return attempt_count[0] <= max_retry

            retry(
                lambda: super(CustomTestCase, self)._callTestMethod(method),
                max_retry=max_retry + 3,
                should_retry=should_retry,
            )
        else:
            retry(
                lambda: super(CustomTestCase, self)._callTestMethod(method),
                max_retry=max_retry,
            )

    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(CI_MULTI_LORA_MODELS)

    def test_ci_lora_models_multi_batch(self):
        run_lora_multiple_batch_on_model_cases(CI_MULTI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_MULTI_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        run_lora_multiple_batch_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
