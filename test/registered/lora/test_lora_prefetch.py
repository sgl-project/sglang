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
    BACKENDS,
    CI_MULTI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TEST_MULTIPLE_BATCH_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    ensure_reproducibility,
)
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l

register_cuda_ci(est_time=400, suite="stage-b-test-small-1-gpu")


def run_mixed_batch_test(
    model_case: LoRAModelCase, torch_dtype, backend, max_new_tokens=32
):
    base_path = model_case.base
    adaptor_paths = [a.name for a in model_case.adaptors]
    print(
        f"\n========== Testing mixed batch prefetch on base '{base_path}' "
        f"with dtype={torch_dtype}, backend={backend} ==========\n"
    )
    ensure_reproducibility()

    prompts = TEST_MULTIPLE_BATCH_PROMPTS[:3]
    configs = [
        (prompts, [None, adaptor_paths[0], adaptor_paths[1]]),
        (prompts, [adaptor_paths[0], None, adaptor_paths[1]]),
        (prompts, [adaptor_paths[0], adaptor_paths[1], None]),
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
        lora_backend=backend,
    )

    results_no_prefetch = []
    with SRTRunner(base_path, enable_lora_prefetch=False, **common_args) as runner:
        for prompts, lora_paths in configs:
            results_no_prefetch.append(
                runner.batch_forward(
                    prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
                ).output_strs
            )

    results_prefetch = []
    with SRTRunner(base_path, enable_lora_prefetch=True, **common_args) as runner:
        for prompts, lora_paths in configs:
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

    print("Mixed batch prefetch test passed")


def run_prefetch_vs_no_prefetch(
    model_case: LoRAModelCase, torch_dtype, backend, max_new_tokens=32
):
    base_path = model_case.base
    adaptor_paths = [a.name for a in model_case.adaptors]
    print(
        f"\n========== Testing prefetch vs no-prefetch on base '{base_path}' "
        f"with dtype={torch_dtype}, backend={backend} ==========\n"
    )
    ensure_reproducibility()
    prompts = DEFAULT_PROMPTS[:2]

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
        lora_backend=backend,
    )

    with SRTRunner(base_path, enable_lora_prefetch=False, **common_args) as runner:
        out_no_prefetch_a1 = runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=[adaptor_paths[0]] * len(prompts),
        )
        out_no_prefetch_a2 = runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=[adaptor_paths[1]] * len(prompts),
        )
        out_no_prefetch_base = runner.forward(prompts, max_new_tokens=max_new_tokens)

    with SRTRunner(base_path, enable_lora_prefetch=True, **common_args) as runner:
        out_prefetch_a1 = runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=[adaptor_paths[0]] * len(prompts),
        )
        out_prefetch_a2 = runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=[adaptor_paths[1]] * len(prompts),
        )
        out_prefetch_base = runner.forward(prompts, max_new_tokens=max_new_tokens)

    for name, no_prefetch, prefetch in [
        ("Adapter1", out_no_prefetch_a1, out_prefetch_a1),
        ("Adapter2", out_no_prefetch_a2, out_prefetch_a2),
        ("Base", out_no_prefetch_base, out_prefetch_base),
    ]:
        scores = calculate_rouge_l(prefetch.output_strs, no_prefetch.output_strs)
        for i, score in enumerate(scores):
            assert score >= 0.8, (
                f"{name} mismatch at index {i}: {score}\n"
                f"Prefetch: {prefetch.output_strs[i]}\n"
                f"No-prefetch: {no_prefetch.output_strs[i]}"
            )

    print("Prefetch vs No-Prefetch test passed")


class TestLoRAPrefetch(CustomTestCase):

    def test_mixed_batch(self):
        for case in CI_MULTI_LORA_MODELS:
            for dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    run_mixed_batch_test(case, dtype, backend)

    def test_prefetch_vs_no_prefetch(self):
        for case in CI_MULTI_LORA_MODELS:
            for dtype in TORCH_DTYPES:
                for backend in BACKENDS:
                    run_prefetch_vs_no_prefetch(case, dtype, backend)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
