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
from typing import Dict, List, Tuple

import torch

from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Compose a SQL query that uses the following table: users, and returns the user_id and name of all users whose name that does not have a duplicate in the table.
    ### Response:
    SELECT user_id, name FROM users WHERE name LIKE 'A%';
    """,
]

ADAPTERS = [
    "faridlazuarda/valadapt-llama-3.1-8B-it-chinese",  # target_modules = q, v
    "philschmid/code-llama-3-1-8b-text-to-sql-lora",  # target_modules = q, k, v, o, gate, up, down
]

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class TestLoRAEviction(CustomTestCase):
    def test_lora_eviction_with_different_target_modules(self):
        """
        Test LoRA eviction with different target modules.

        This test runs inference against two LoRA adapters in different orders to force eviction behavior, and ensures
        that the outputs of the same (adapter, prompt) pair are consistent across runs.
        """
        output_history = {}
        self._run_test(ADAPTERS, output_history, reverse=False)
        self._run_test(ADAPTERS, output_history, reverse=True)

    def _run_test(
        self,
        lora_paths: List[str],
        output_history: Dict[Tuple[str, str], str],
        reverse: bool,
        repeat: int = 2,
    ):
        max_new_tokens = 256
        backend = "triton"
        torch_dtype = torch.float16
        base_path = BASE_MODEL
        assert len(lora_paths) >= 2

        # Initialize runners
        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=lora_paths,
            max_loras_per_batch=1,
            lora_backend=backend,
            disable_radix_cache=True,
        ) as srt_runner:
            adapter_sequence = lora_paths if not reverse else lora_paths[::-1]

            for i in range(repeat):
                for j, adapter in enumerate(adapter_sequence):
                    print(
                        f"\n========== Testing LoRA eviction with adapter '{adapter}' (#{j+1}/{len(adapter_sequence)}), reversed: {reverse}, repeat: {i+1}/{repeat} ---"
                    )
                    for prompt in PROMPTS:
                        print("\nprompt:\n", prompt)
                        srt_outputs = srt_runner.forward(
                            [prompt],
                            max_new_tokens=max_new_tokens,
                            lora_paths=[adapter],
                        )
                        output = srt_outputs.output_strs[0].strip()
                        print("\noutput:\n", output)

                        prev_output = output_history.get((adapter, prompt))
                        if prev_output is not None:
                            self.assertEqual(
                                prev_output,
                                output,
                                f"Output mismatch for adapter {adapter} and prompt '{prompt}' on repeat {j + 1}, previous: '{prev_output}', current: '{output}'.",
                            )
                        else:
                            output_history[(adapter, prompt)] = output


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
