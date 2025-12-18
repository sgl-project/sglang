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
Test loading LoRA adapter from tensors directly.

This test verifies that LoRA adapters can be loaded from serialized tensors
and configuration dictionaries, without requiring a local directory path.

Usage:
    python test_lora_from_tensors.py
    or
    python -m unittest test_lora_from_tensors
"""

import json
import multiprocessing as mp
import os
import unittest

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import LoadLoRAAdapterFromTensorsReqInput
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)

from sglang.test.test_utils import CustomTestCase

# Test configuration
MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
TEST_PROMPT = "Hello, my name is"
EXPECTED_OUTPUT = (
    " Alice, and I am a software engineer. I am excited to share my journey"
)
MAX_NEW_TOKENS = 16


class TestLoRAFromTensors(CustomTestCase):
    """Test case for loading LoRA adapters from tensors."""

    def test_load_lora_from_tensors(self):
        """Test loading LoRA from tensors using the Engine API."""
        # Download LoRA adapter to HuggingFace cache directory
        lora_path = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )

        # Load tensors and config from downloaded adapter
        tensors = load_file(os.path.join(lora_path, "adapter_model.safetensors"))
        with open(os.path.join(lora_path, "adapter_config.json"), "r") as f:
            config_dict = json.load(f)

        # Verify loaded data
        self.assertGreater(len(tensors), 0, "LoRA tensors should not be empty")
        self.assertIn("r", config_dict, "Config should contain 'r' field")
        self.assertIn(
            "lora_alpha", config_dict, "Config should contain 'lora_alpha' field"
        )

        # Initialize engine with LoRA enabled
        engine = Engine(
            model_path=MODEL_PATH,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            mem_fraction_static=0.6,
            log_level="error",
        )

        try:

            # Test generation without LoRA
            output_before = engine.generate(
                prompt=[TEST_PROMPT],
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
            )

            # Serialize tensors and create request
            serialized_tensors = MultiprocessingSerializer.serialize(
                tensors, output_str=True
            )

            req = LoadLoRAAdapterFromTensorsReqInput(
                lora_name="test_lora_from_tensors",
                config_dict=config_dict,
                serialized_tensors=serialized_tensors,
            )

            # Load LoRA adapter from tensors
            result = engine.loop.run_until_complete(
                engine.tokenizer_manager.load_lora_adapter_from_tensors(req, None)
            )
            self.assertTrue(
                result.success,
                f"Failed to load LoRA from tensors: {result.error_message}",
            )

            # Test generation with loaded LoRA
            output_after = engine.generate(
                prompt=[TEST_PROMPT],
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
                lora_path=["test_lora_from_tensors"],
            )

            print(f"\n[Before LoRA] {output_before[0]}")
            print(f"[After LoRA]  {output_after[0]}")

            # Verify the outputs
            self.assertNotEqual(
                output_before[0]["text"][: len(EXPECTED_OUTPUT)],
                EXPECTED_OUTPUT,
                "Output before applying LoRA should not match expected result",
            )

            self.assertEqual(
                output_after[0]["text"][: len(EXPECTED_OUTPUT)],
                EXPECTED_OUTPUT,
                "Output after applying LoRA does not match expected result",
            )

        finally:
            # Cleanup
            if hasattr(engine, "shutdown"):
                engine.shutdown()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
