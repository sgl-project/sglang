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

import contextlib
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


@contextlib.contextmanager
def dynamically_loaded_adapter(runner, lora_path: str, lora_name: str):
    """A context manager to load and automatically unload a LoRA adapter."""
    try:
        runner.load_lora_adapter(lora_name=lora_name, lora_path=lora_path)
        yield
    finally:
        runner.unload_lora_adapter(lora_name=lora_name)


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

    def test_lora_eviction_with_reused_lora_name(self):
        """
        Test LoRA eviction with reused LoRA names.

        This test runs inference against two LoRA adapters with the same name to ensure that the eviction behavior
        works correctly when reusing LoRA names.
        """
        output_history = {}
        self._run_test(ADAPTERS, output_history, reuse_lora_name=True, repeat=1)
        self._run_test(ADAPTERS, output_history, reuse_lora_name=False, repeat=1)

    def _run_test(
        self,
        lora_paths: List[str],
        output_history: Dict[Tuple[str, str], str],
        reverse: bool = False,
        repeat: int = 2,
        reuse_lora_name: bool = False,
    ):
        REUSED_LORA_NAME = "lora"
        max_new_tokens = 256
        torch_dtype = torch.float16
        base_path = BASE_MODEL
        assert len(lora_paths) >= 2

        initial_lora_paths = lora_paths if not reuse_lora_name else None
        # Initialize runners
        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=initial_lora_paths,
            max_loras_per_batch=1,
            enable_lora=True,
            max_lora_rank=256,
            # Need to list all lora modules, or "all" might include lora modules without assigning lora weights
            # lora_target_modules=["all"],
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ) as srt_runner:
            adapter_sequence = lora_paths if not reverse else lora_paths[::-1]

            for i in range(repeat):
                for j, lora_path in enumerate(adapter_sequence):
                    print(
                        f"\n========== Testing LoRA eviction with adapter '{lora_path}' (#{j + 1}/{len(adapter_sequence)}), reuse_lora_name: {reuse_lora_name}, reversed: {reverse}, repeat: {i + 1}/{repeat} ---"
                    )

                    lora_name = REUSED_LORA_NAME if reuse_lora_name else lora_path
                    context = (
                        dynamically_loaded_adapter(srt_runner, lora_path, lora_name)
                        if reuse_lora_name
                        else contextlib.nullcontext()
                    )
                    with context:
                        for prompt in PROMPTS:
                            print("\nprompt:\n", prompt)
                            srt_outputs = srt_runner.forward(
                                [prompt],
                                max_new_tokens=max_new_tokens,
                                lora_paths=[lora_name],
                            )
                            output = srt_outputs.output_strs[0].strip()
                            print("\noutput:\n", output)

                            prev_output = output_history.get((lora_path, prompt))
                            if prev_output is not None:
                                self.assertEqual(
                                    prev_output,
                                    output,
                                    f"Output mismatch for adapter {lora_path} and prompt '{prompt}' on repeat {j + 1}, previous: '{prev_output}', current: '{output}'.",
                                )
                            else:
                                output_history[(lora_path, prompt)] = output


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
