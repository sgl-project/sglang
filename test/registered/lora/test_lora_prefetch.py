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
End-to-end tests for the --enable-lora-prefetch server argument.
"""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TEST_MULTIPLE_BATCH_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    ensure_reproducibility,
    run_lora_test_one_by_one,
)
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l

register_cuda_ci(est_time=300, suite="stage-b-test-small-1-gpu")


class TestLoRAPrefetch(CustomTestCase):

    def _run_mixed_batch_test(
        self,
        model_case: LoRAModelCase,
        torch_dtype,
    ):
        base_path = model_case.base
        adaptor_paths = [a.name for a in model_case.adaptors]
        print(
            f"\n========== Testing mixed batch prefetch on base '{base_path}' "
            f"with dtype={torch_dtype} ==========\n"
        )
        ensure_reproducibility()
        max_new_tokens = 32

        prompts = TEST_MULTIPLE_BATCH_PROMPTS[:3]
        configs = [
            [None, adaptor_paths[0], adaptor_paths[1]],
            [adaptor_paths[0], None, adaptor_paths[1]],
            [adaptor_paths[0], adaptor_paths[1], None],
            [adaptor_paths[1], adaptor_paths[0], adaptor_paths[1]],
        ]
        common_args = dict(
            torch_dtype=torch_dtype,
            model_type="generation",
            tp_size=model_case.tp_size,
            lora_paths=adaptor_paths,
            max_loras_per_batch=model_case.max_loras_per_batch,
            max_loaded_loras=model_case.max_loras_per_batch * 2,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.65,
            sleep_on_idle=True,
        )

        results_no_prefetch = []
        with SRTRunner(base_path, enable_lora_prefetch=False, **common_args) as runner:
            for lora_paths in configs:
                results_no_prefetch.append(
                    runner.batch_forward(
                        prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
                    ).output_strs
                )

        results_prefetch = []
        with SRTRunner(base_path, enable_lora_prefetch=True, **common_args) as runner:
            for lora_paths in configs:
                results_prefetch.append(
                    runner.batch_forward(
                        prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
                    ).output_strs
                )

        for i, (res_no_prefetch, res_prefetch) in enumerate(
            zip(results_no_prefetch, results_prefetch)
        ):
            scores = calculate_rouge_l(res_prefetch, res_no_prefetch)
            for j, score in enumerate(scores):
                assert score >= 0.8, (
                    f"Batch {i} prompt {j} mismatch: {score}\n"
                    f"Prefetch: {res_prefetch[j]}\n"
                    f"No-prefetch: {res_no_prefetch[j]}"
                )

    def test_mixed_batch(self):
        for model_case in CI_MULTI_LORA_MODELS:
            for dtype in TORCH_DTYPES:
                self._run_mixed_batch_test(model_case, dtype)

    def test_prefetch_tp(self):
        for model_case in CI_MULTI_LORA_MODELS:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
            model_case.tp_size = 2
            for dtype in TORCH_DTYPES:
                run_lora_test_one_by_one(
                    prompts,
                    model_case,
                    dtype,
                    max_new_tokens=32,
                    enable_lora_prefetch=True,
                    test_tag=f"tp={model_case.tp_size}",
                )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
