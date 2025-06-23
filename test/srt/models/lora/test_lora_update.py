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
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import torch

from sglang.test.runners import SRTRunner
from sglang.test.test_utils import CustomTestCase

PROMPTS = [
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    NOOP = "noop"
    FORWARD = "forward"


@dataclass
class Operation:
    type: OperationType
    data: Optional[str]


@dataclass
class TestCase:
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    initial_adapters: List[str]
    op_sequence: List[Operation]
    max_new_tokens: int = 32


def create_batch_data(adapters: Union[str, list]) -> dict:
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]


TEST_CASES = [
    # basic test, no eviction
    TestCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        all_adapters=[
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            "pbevan11/llama-3.1-8b-ocr-correction",
        ],
        initial_adapters=["philschmid/code-llama-3-1-8b-text-to-sql-lora"],
        op_sequence=[
            Operation(
                type=OperationType.LOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.LOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                    ]
                ),
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                    ]
                ),
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
            ),
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                    ]
                ),
            ),
        ],
    ),
    # Eviction
    TestCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=1,
        all_adapters=[
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            "pbevan11/llama-3.1-8b-ocr-correction",
        ],
        initial_adapters=["philschmid/code-llama-3-1-8b-text-to-sql-lora"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ),
            Operation(
                type=OperationType.LOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
            ),
            Operation(
                type=OperationType.LOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
            ),
        ],
    ),
]


class TestLoRADynamicUpdate(CustomTestCase):
    """
    This test case verifies that the SRT runner can dynamically load and unload LoRA adapters
    during a sequence of operations, and that the outputs of forward passes with dynamically loaded
    adapters match the outputs of forward passes with statically loaded adapters.
    """

    def _repeat_each(lst, n):
        return [x for x in lst for _ in range(n)]

    def _run_operation_sequence(
        self,
        base: str,
        max_loras_per_batch: int,
        initial_adapters: List[str],
        op_sequence: List[Operation],
        max_new_tokens: int = 32,
    ) -> List[tuple]:
        """
        Runs a sequence of operations on the SRT runner, including loading and unloading LoRA adapters,
        and performing forward passes with the current set of loaded adapters.
        """

        forward_outputs = []
        with SRTRunner(
            model_path=base,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=initial_adapters,
            max_loras_per_batch=max_loras_per_batch,
            lora_backend="triton",
            disable_cuda_graph=False,
            disable_radix_cache=True,
        ) as srt_runner:
            expected_adapters = set(initial_adapters)
            actual_adapters = set(initial_adapters)
            for op in op_sequence:
                op_type = op.type
                data = op.data
                print(
                    f"\n---  Running operation: {op_type} --- data: {data} --- current adapters: {actual_adapters}"
                )
                if op_type == OperationType.LOAD:
                    expected_adapters.add(data)
                    result = srt_runner.load_lora_adapter(
                        lora_name=data,
                        lora_path=data,
                    )
                    actual_adapters = set(result.loaded_adapters)
                    assert result.success
                    assert actual_adapters == expected_adapters
                elif op_type == OperationType.UNLOAD:
                    expected_adapters.remove(data)
                    result = srt_runner.unload_lora_adapter(
                        lora_name=data,
                    )
                    actual_adapters = set(result.loaded_adapters)
                    assert result.success
                    assert actual_adapters == expected_adapters
                elif op_type == OperationType.FORWARD:
                    prompts, adapters = zip(*data)
                    result = srt_runner.batch_forward(
                        prompts=list(prompts),
                        lora_paths=list(adapters),
                        max_new_tokens=max_new_tokens,
                    )
                    forward_outputs.append(result)
                print(f"\n--- Operation {op_type} result: {result}")

            return forward_outputs

    def test_dynamic_adapter_updates(self):
        for test_case in TEST_CASES:
            # Test dynamic loading of adapters
            # TODO (lifuhuang): currently at least one LoRA path is required during initialization to enable lora,
            # we should fix this in the future https://github.com/sgl-project/sglang/issues/7463.
            dynamic_output = self._run_operation_sequence(
                initial_adapters=test_case.initial_adapters,
                base=test_case.base,
                max_loras_per_batch=test_case.max_loras_per_batch,
                op_sequence=test_case.op_sequence,
                max_new_tokens=test_case.max_new_tokens,
            )

            # static loading
            forward_ops = [
                x for x in test_case.op_sequence if x.type == OperationType.FORWARD
            ]
            static_output = self._run_operation_sequence(
                initial_adapters=test_case.all_adapters,
                base=test_case.base,
                max_loras_per_batch=test_case.max_loras_per_batch,
                op_sequence=forward_ops,
                max_new_tokens=test_case.max_new_tokens,
            )

            assert len(dynamic_output) == len(
                static_output
            ), f"Dynamic output length {len(dynamic_output)} does not match static output length {len(static_output)}"
            for i, (dynamic, static) in enumerate(zip(dynamic_output, static_output)):
                assert len(dynamic.output_strs) == len(static.output_strs), (
                    f"Output length mismatch at batch {i}: "
                    f"Dynamic: {len(dynamic.output_strs)}, Static: {len(static.output_strs)}"
                )
                for j, (d_out, s_out) in enumerate(
                    zip(dynamic.output_strs, static.output_strs)
                ):
                    d_out = d_out.strip()
                    s_out = s_out.strip()
                    assert d_out == s_out, (
                        f"Output mismatch at batch {i}, prompt {j}: "
                        f"Dynamic: '{d_out}', Static: '{s_out}'"
                    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
