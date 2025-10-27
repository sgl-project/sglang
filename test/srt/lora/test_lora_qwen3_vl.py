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

import os
import random
import unittest
from typing import List

import torch
from utils import TORCH_DTYPES, LoRAAdaptor, LoRAModelCase, ensure_reproducibility

from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci


class TestLoRAQwen3VLFilters(CustomTestCase):
    """Unit tests for should_apply_lora gating on Qwen3‑VL (dense) and Qwen3‑VL‑MoE."""

    def test_qwen3_vl_dense_should_apply_lora_regex(self):
        # Positive cases: text-layer projections only
        positives = [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.2.mlp.gate_up_proj",
            "model.layers.3.mlp.down_proj",
        ]
        for name in positives:
            self.assertTrue(
                bool(Qwen3VLForConditionalGeneration._lora_pattern.match(name)),
                f"Expected to match: {name}",
            )

        # Negative cases: vision, malformed, or unsupported projections
        negatives = [
            "visual.blocks.0.attn.qkv_proj",
            "model.layers.x.self_attn.qkv_proj",  # non-numeric layer
            "model.layers.0.attn.qkv_proj",  # wrong block name
            "model.layers.0.mlp.not_proj",  # unsupported projection
            "model.layers.0.self_attn.q_proj",  # split q/k/v not allowed here
        ]
        for name in negatives:
            self.assertFalse(
                bool(Qwen3VLForConditionalGeneration._lora_pattern.match(name)),
                f"Should not match: {name}",
            )


LORA_MODELS_QWEN3 = [
    LoRAModelCase(
        base="Qwen/Qwen3-VL-4B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="mryufei/Qwen3-VL-4B-Instruct-trl-sft",
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=1,
    ),
]


TEST_MULTIPLE_BATCH_PROMPTS = [
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
    """
    ### Instruction:
    Write a poem about the transformers Python library.
    Mention the word "large language models" in that poem.
    ### Response:
    The Transformers are large language models,
    They're used to make predictions on text.
    """,
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


class TestLoRAQwen3VLIntegration(CustomTestCase):
    """Parity-style integration matching test_lora_qwen3.py, using a fixed model+adapters set."""

    def _run_lora_multiple_batch_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            for torch_dtype in TORCH_DTYPES:
                max_new_tokens = 16
                backend = "triton"
                base_path = model_case.base
                lora_adapter_paths = [a.name for a in model_case.adaptors]

                # build several batches with varying adapter assignments
                batches = [
                    (
                        [
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                        ],
                        [None, lora_adapter_paths[0], None],
                    ),
                    (
                        [
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                        ],
                        [lora_adapter_paths[0], None, None],
                    ),
                    (
                        [
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                        ],
                        [None, None, None],
                    ),
                ]

                print(
                    f"\n=== Qwen3-VL LoRA parity on '{base_path}', backend={backend}, dtype={torch_dtype} ==="
                )

                ensure_reproducibility()
                srt_runner = SRTRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    lora_paths=lora_adapter_paths,
                    max_loras_per_batch=model_case.max_loras_per_batch,
                    lora_backend=backend,
                    sleep_on_idle=True,
                    attention_backend="torch_native",
                )

                ensure_reproducibility()
                hf_runner = HFRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    patch_model_do_sample_false=True,
                )

                with srt_runner, hf_runner:
                    for i, (prompts, lora_paths) in enumerate(batches):
                        print(
                            f"\n--- Running Batch {i+1} --- prompts: {prompts}, lora_paths: {lora_paths}"
                        )

                        srt_outputs = srt_runner.batch_forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,
                        )

                        hf_outputs = hf_runner.forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,
                        )

                        print("SRT outputs:", [s for s in srt_outputs.output_strs])
                        print("HF outputs:", [s for s in hf_outputs.output_strs])

                        for srt_out, hf_out in zip(
                            srt_outputs.output_strs, hf_outputs.output_strs
                        ):
                            srt_str = srt_out.strip()
                            hf_str = hf_out.strip()
                            rouge_tol = model_case.rouge_l_tolerance
                            rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                            if rouge_score < rouge_tol:
                                raise AssertionError(
                                    f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                    f"for base '{base_path}', adaptor '{lora_paths}', backend '{backend}', prompt: '{prompts}...'"
                                )

                        print(f"--- Batch {i+1} Comparison Passed --- ")

    def test_ci_lora_models(self):
        self._run_lora_multiple_batch_on_model_cases(LORA_MODELS_QWEN3)

    def test_all_lora_models(self):
        if is_in_ci():
            return
        filtered = []
        for model_case in LORA_MODELS_QWEN3:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered.append(model_case)
        self._run_lora_multiple_batch_on_model_cases(filtered)


if __name__ == "__main__":
    unittest.main()
