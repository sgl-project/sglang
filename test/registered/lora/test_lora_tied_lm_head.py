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

"""
Test LoRA on models with tied lm_head (tie_word_embeddings=True).

When tie_word_embeddings=True, lm_head shares the same weight tensor as
embed_tokens. PyTorch's named_modules() deduplicates by object identity,
so lm_head won't appear as a separate module. This test validates that
SGLang correctly handles this case by untying lm_head before LoRA wrapping.

The test:
1. Programmatically creates a LoRA adapter with lm_head in target_modules
   using PEFT on a model with tie_word_embeddings=True (Qwen/Qwen2.5-0.5B).
2. Compares logprobs between HuggingFace+PEFT and SGLang to ensure numerical
   consistency. This implicitly verifies no NaN values are produced and that
   LoRA is actually being applied (since HF+PEFT is the trusted reference).
"""

import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import torch

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "peft", "--no-deps"])
    from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)

# Use a small model with tie_word_embeddings=True
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

TEST_PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]

MAX_NEW_TOKENS = 16
LOGPROB_THRESHOLD = 2e-1


def create_lora_adapter_with_lm_head(base_model_name: str, output_dir: str):
    """
    Programmatically create a LoRA adapter that targets lm_head,
    using a model with tie_word_embeddings=True.

    The adapter uses randomly initialized LoRA weights (no training).
    This is sufficient to test that:
    - SGLang can load the adapter without errors
    - lm_head LoRA is applied (output differs from base model)
    - Logprobs match between HF and SGLang
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # Verify the model actually has tied embeddings
    assert (
        model.config.tie_word_embeddings
    ), f"Expected tie_word_embeddings=True for {base_model_name}"

    # Only target lm_head to isolate the test to the tied-embedding scenario.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["lm_head"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # PEFT initializes lora_B to zeros by default, which makes the adapter
    # produce identical output to the base model. Initialize lora_B with
    # non-zero random weights so the adapter has a visible effect.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    peft_model.save_pretrained(output_dir)

    # Verify the saved adapter contains lm_head keys
    from safetensors import safe_open

    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    f = safe_open(safetensors_path, framework="pt")
    lm_head_keys = [k for k in f.keys() if "lm_head" in k]
    assert (
        len(lm_head_keys) > 0
    ), f"Expected lm_head LoRA weights in adapter, got keys: {sorted(f.keys())}"

    print(f"Created LoRA adapter at {output_dir}")
    print(f"  lm_head keys: {lm_head_keys}")

    # Clean up the model to free memory
    del peft_model, model
    torch.cuda.empty_cache()


class TestLoRATiedLMHead(CustomTestCase):
    """
    Test that LoRA works correctly on models with tied lm_head.
    """

    _adapter_dir = None

    @classmethod
    def setUpClass(cls):
        """Create a temporary LoRA adapter with lm_head targeting."""
        super().setUpClass()
        cls._adapter_dir = tempfile.mkdtemp(prefix="sglang_test_lora_tied_lm_head_")
        create_lora_adapter_with_lm_head(BASE_MODEL, cls._adapter_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary adapter directory."""
        if cls._adapter_dir and os.path.exists(cls._adapter_dir):
            shutil.rmtree(cls._adapter_dir)
        super().tearDownClass()

    def test_tied_lm_head_lora_hf_sgl_logprob_match(self):
        """
        Compare logprobs between HuggingFace+PEFT and SGLang+LoRA
        for a tied lm_head adapter, ensuring numerical consistency.
        """
        prompts = TEST_PROMPTS[:2]

        # Run SGLang with LoRA
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[self._adapter_dir],
            max_loras_per_batch=1,
            lora_backend="triton",
            lora_target_modules=["lm_head"],
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.80,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self._adapter_dir] * len(prompts),
            )

        torch.cuda.empty_cache()

        # Run HuggingFace with LoRA (via PEFT)
        with HFRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self._adapter_dir] * len(prompts),
            )

        # Compare prefill logprobs
        for i in range(len(prompts)):
            srt_logprobs = torch.tensor(srt_outputs.top_input_logprobs[i])
            hf_logprobs = torch.tensor(hf_outputs.top_input_logprobs[i])
            max_diff = torch.max(torch.abs(srt_logprobs - hf_logprobs)).item()
            print(f"Prompt {i} prefill logprob max_diff (SGLang vs HF): {max_diff:.6e}")
            self.assertLess(
                max_diff,
                LOGPROB_THRESHOLD,
                f"Prompt {i}: prefill logprob diff {max_diff:.6e} "
                f"exceeds threshold {LOGPROB_THRESHOLD:.0e}",
            )

        # Compare decode logprobs
        for i in range(len(prompts)):
            srt_logprobs = torch.tensor(srt_outputs.top_output_logprobs[i])
            hf_logprobs = torch.tensor(hf_outputs.top_output_logprobs[i])
            max_diff = torch.max(torch.abs(srt_logprobs - hf_logprobs)).item()
            print(f"Prompt {i} decode logprob max_diff (SGLang vs HF): {max_diff:.6e}")
            self.assertLess(
                max_diff,
                LOGPROB_THRESHOLD,
                f"Prompt {i}: decode logprob diff {max_diff:.6e} "
                f"exceeds threshold {LOGPROB_THRESHOLD:.0e}",
            )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
