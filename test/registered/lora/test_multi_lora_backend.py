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

"""Mixed-adapter batches, split across LoRA slots, must match HF row by row."""

import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    TEST_MULTIPLE_BATCH_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    create_multiple_batch_test_samples,
    ensure_reproducibility,
)
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l

register_cuda_ci(est_time=65, stage="base-b", runner_config="1-gpu-large")
# backend-specific: ROCm's Triton build of the chunked-SGMV LoRA kernels on
# mixed-adapter batches.
register_amd_ci(est_time=65, suite="stage-b-test-1-gpu-small-amd")

# The base model holds a slot too, so two slots make the scheduler split every
# three-route batch below into microbatches.
MAX_LORAS_PER_BATCH = 2
MAX_NEW_TOKENS = 32
ROUGE_L_TOLERANCE = 0.9


def _split_batches(a, b):
    """Route patterns on one shared prompt; three-route ones get split."""
    prompts = [TEST_MULTIPLE_BATCH_PROMPTS[0]] * 3
    routes = [
        [None, a, b],
        [a, None, b],
        [a, b, None],
        [None, b, None],
        [a, b, a],
        [None, None, None],
    ]
    return [(prompts, lora_paths) for lora_paths in routes]


class TestMultiLoRABackend(CustomTestCase):
    def test_ci_lora_models_multi_batch(self):
        for model_case in CI_MULTI_LORA_MODELS:
            for torch_dtype in TORCH_DTYPES:
                self._run_model_case(model_case, torch_dtype)

    def _run_model_case(self, model_case: LoRAModelCase, torch_dtype):
        a, b = [adaptor.name for adaptor in model_case.adaptors]
        batches = create_multiple_batch_test_samples(
            TEST_MULTIPLE_BATCH_PROMPTS, [a, b]
        ) + _split_batches(a, b)

        ensure_reproducibility()
        srt_runner = SRTRunner(
            model_case.base,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=[a, b],
            max_loras_per_batch=MAX_LORAS_PER_BATCH,
            max_loaded_loras=model_case.max_loaded_loras,
            sleep_on_idle=True,  # schedule each request batch as one batch
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
        )
        ensure_reproducibility()
        hf_runner = HFRunner(
            model_case.base,
            torch_dtype=torch_dtype,
            model_type="generation",
            patch_model_do_sample_false=True,
        )
        # HF generates one row at a time, so a (prompt, route) reference does
        # not depend on the batch that asked for it.
        hf_cache = {}
        with srt_runner, hf_runner:
            for i, (prompts, lora_paths) in enumerate(batches):
                print(f"\n--- Batch {i + 1} --- lora_paths: {lora_paths}")
                srt_strs = srt_runner.batch_forward(
                    prompts, max_new_tokens=MAX_NEW_TOKENS, lora_paths=lora_paths
                ).output_strs
                for prompt, route, srt_str in zip(prompts, lora_paths, srt_strs):
                    if (prompt, route) not in hf_cache:
                        hf_cache[prompt, route] = hf_runner.forward(
                            [prompt], max_new_tokens=MAX_NEW_TOKENS, lora_paths=[route]
                        ).output_strs[0]
                    hf_str = hf_cache[prompt, route]
                    rouge = calculate_rouge_l([srt_str.strip()], [hf_str.strip()])[0]
                    print(f"route={route} rouge_l={rouge:.3f}")
                    print(f"  SRT: {srt_str.strip()!r}")
                    print(f"  HF:  {hf_str.strip()!r}")
                    self.assertGreaterEqual(
                        rouge,
                        ROUGE_L_TOLERANCE,
                        f"batch {i + 1}, route {route}: SRT output diverges from HF",
                    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
