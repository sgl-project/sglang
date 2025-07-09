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
from typing import Any, List, Optional, Union

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]

MEM_FRACTION_STATIC = 0.8


class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    FORWARD = "forward"
    EXPECT_ERROR = "expect_error"


@dataclass
class Operation:
    # Operation type, can be LOAD, UNLOAD, FORWARD, or EXPECT_ERROR
    type: OperationType
    # Data associated with the operation. Exact type varies depending on the operation
    data: Optional[Any]


@dataclass
class TestCase:
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    initial_adapters: List[str]
    op_sequence: List[Operation]
    max_new_tokens: int = 32


def create_batch_data(adapters: Union[str, list]) -> List[tuple[str, str]]:
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
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ),
            Operation(
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data(
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                    ),
                    "not loaded",
                ),
            ),
            Operation(
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                    "not loaded",
                ),
            ),
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
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                    "not loaded",
                ),
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
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data(
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                    ),
                    "not loaded",
                ),
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
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data(
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                    ),
                    "not loaded",
                ),
            ),
            Operation(
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                    "not loaded",
                ),
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
                type=OperationType.EXPECT_ERROR,
                data=(
                    create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                    "not loaded",
                ),
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


class LoRAUpdateTestSessionMode(Enum):
    ENGINE = "engine"
    SERVER = "server"


class LoRAUpdateTestSessionBase:
    """
    Base context manager for testing LoRA adapters.
    """

    def __init__(
        self,
        *,
        testcase: Optional[TestCase],
        model_path: str,
        lora_paths: list[str],
        max_loras_per_batch: int = 1,
        lora_backend: str = "triton",
        disable_cuda_graph: bool = False,
        cuda_graph_max_bs: int = 4,
    ):
        self.testcase = testcase
        self.model_path = model_path
        self.lora_paths = lora_paths
        self.max_loras_per_batch = max_loras_per_batch
        self.lora_backend = lora_backend
        self.disable_cuda_graph = disable_cuda_graph
        self.cuda_graph_max_bs = cuda_graph_max_bs

        self.expected_adapters = set(lora_paths)
        self.handle = None  # Will be set in __enter__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't suppress exceptions by default
        return False

    def load_lora_adapter(self, lora_name: str, lora_path: Optional[str] = None):
        """
        Load a LoRA adapter by name and path.
        """
        raise NotImplementedError("Subclasses must implement load_lora_adapter")

    def unload_lora_adapter(self, lora_name: str):
        """
        Unload a LoRA adapter by name.
        """
        raise NotImplementedError("Subclasses must implement unload_lora_adapter")

    def forward(
        self,
        prompts: List[str],
        lora_paths: List[str],
        max_new_tokens: int = 32,
    ):
        """
        Perform a batch forward pass with the current set of loaded LoRA adapters.
        """
        raise NotImplementedError("Subclasses must implement forward")


class LoRAUpdateEngineTestSession(LoRAUpdateTestSessionBase):
    """
    Context manager for testing LoRA adapters with in-process engine.
    """

    def __enter__(self):
        # in-process runner
        self.handle = SRTRunner(
            model_path=self.model_path,
            model_type="generation",
            lora_paths=self.lora_paths,
            lora_backend=self.lora_backend,
            torch_dtype=torch.float16,
            mem_fraction_static=MEM_FRACTION_STATIC,
            max_loras_per_batch=self.max_loras_per_batch,
            disable_cuda_graph=self.disable_cuda_graph,
            cuda_graph_max_bs=self.cuda_graph_max_bs,
            disable_radix_cache=True,
        )
        self.handle.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            # delegate cleanup to SRTRunner
            return self.handle.__exit__(exc_type, exc_val, exc_tb)
        # don't suppress exceptions
        return False

    def load_lora_adapter(self, lora_name: str, lora_path: Optional[str] = None):
        """
        Load a LoRA adapter by name and path.
        """
        if lora_path is None:
            lora_path = lora_name

        self.expected_adapters.add(lora_name)

        response = self.handle.load_lora_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
        )
        self.testcase.assertTrue(response.success)
        loaded_adapters = set(response.loaded_adapters)

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(loaded_adapters, self.expected_adapters)

    def unload_lora_adapter(self, lora_name: str):
        """
        Unload a LoRA adapter by name.
        """
        self.expected_adapters.remove(lora_name)

        response = self.handle.unload_lora_adapter(
            lora_name=lora_name,
        )
        self.testcase.assertTrue(response.success)
        loaded_adapters = set(response.loaded_adapters)

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(loaded_adapters, self.expected_adapters)

    def forward(
        self,
        prompts: List[str],
        lora_paths: List[str],
        max_new_tokens: int = 32,
        expected_error: str = None,
    ):
        """
        Perform a batch forward pass with the current set of loaded LoRA adapters.
        """
        try:
            response = self.handle.batch_forward(
                prompts=prompts,
                lora_paths=lora_paths,
                max_new_tokens=max_new_tokens,
            )
        except ValueError as e:
            if expected_error:
                error_message = str(e)
                self.testcase.assertIn(expected_error, error_message)
                print(f"Received error as expected: {error_message}")
                return error_message

            raise e

        self.testcase.assertEqual(len(response.output_strs), len(prompts))
        output = response.output_strs
        print(f"output_strs: {output}")

        return output


class LoRAUpdateServerTestSession(LoRAUpdateTestSessionBase):
    """
    Context manager for testing LoRA adapters with standalone server.
    """

    def __enter__(self):
        other_args = [
            "--cuda-graph-max-bs",
            str(self.cuda_graph_max_bs),
            "--lora-paths",
            *self.lora_paths,
            "--max-loras-per-batch",
            str(self.max_loras_per_batch),
            "--lora-backend",
            self.lora_backend,
            "--disable-radix-cache",
            "--random-seed",
            "42",
            "--max-running-request",
            "1",
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
        ]
        if self.disable_cuda_graph:
            other_args.append("--disable-cuda-graph")

        # launch external server
        self.handle = popen_launch_server(
            self.model_path,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            kill_process_tree(self.handle.pid)
        # don't suppress exceptions
        return False

    def load_lora_adapter(self, lora_name: str, lora_path: Optional[str] = None):
        """
        Load a LoRA adapter by name and path.
        """
        if lora_path is None:
            lora_path = lora_name

        self.expected_adapters.add(lora_name)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
        )
        self.testcase.assertTrue(response.ok)
        loaded_adapters = set(response.json()["loaded_adapters"])

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(loaded_adapters, self.expected_adapters)

    def unload_lora_adapter(self, lora_name: str):
        """
        Unload a LoRA adapter by name.
        """
        self.expected_adapters.remove(lora_name)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": lora_name},
        )
        self.testcase.assertTrue(response.ok)
        loaded_adapters = set(response.json()["loaded_adapters"])

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(loaded_adapters, self.expected_adapters)

    def forward(
        self,
        prompts: List[str],
        lora_paths: List[str],
        max_new_tokens: int = 32,
        expected_error: str = None,
    ):
        """
        Perform a batch forward pass with the current set of loaded LoRA adapters.
        """
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompts,
                "lora_path": lora_paths,
                "sampling_params": {
                    "temperature": 0,
                    "top_k": 1,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        if expected_error:
            self.testcase.assertEqual(response.status_code, 400)
            self.testcase.assertIn(expected_error, response.text)
            output = response.text
            print(f"Received error as expected: {response.text}")
            return output
        else:
            self.testcase.assertTrue(response.ok)
            output = [r["text"] for r in response.json()]
            self.testcase.assertEqual(len(output), len(prompts))
            print(f"output_strs: {output}")
            return output


# Factory function to create the appropriate LoRA test session based on mode
def LoRAUpdateTestSession(
    *,
    testcase: Optional[TestCase],
    mode: LoRAUpdateTestSessionMode,
    model_path: str,
    lora_paths: list[str],
    max_loras_per_batch: int = 1,
    lora_backend: str = "triton",
    disable_cuda_graph: bool = False,
    cuda_graph_max_bs: int = 4,
):
    common_kwargs = {
        "testcase": testcase,
        "model_path": model_path,
        "lora_paths": lora_paths,
        "max_loras_per_batch": max_loras_per_batch,
        "lora_backend": lora_backend,
        "disable_cuda_graph": disable_cuda_graph,
        "cuda_graph_max_bs": cuda_graph_max_bs,
    }

    if mode == LoRAUpdateTestSessionMode.ENGINE:
        return LoRAUpdateEngineTestSession(**common_kwargs)
    elif mode == LoRAUpdateTestSessionMode.SERVER:
        return LoRAUpdateServerTestSession(**common_kwargs)
    else:
        raise ValueError(f"Unrecognized mode: {mode!r}")


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
        mode: LoRAUpdateTestSessionMode,
        base: str,
        initial_adapters: List[str],
        max_loras_per_batch: int,
        op_sequence: List[Operation],
        max_new_tokens: int = 32,
    ) -> List[tuple]:
        """
        Runs a sequence of operations on the SRT runner, including loading and unloading LoRA adapters,
        and performing forward passes with the current set of loaded adapters.
        """

        forward_outputs = []
        with LoRAUpdateTestSession(
            testcase=self,
            mode=mode,
            model_path=base,
            lora_paths=initial_adapters,
            max_loras_per_batch=max_loras_per_batch,
        ) as session:
            for op in op_sequence:
                op_type = op.type
                data = op.data
                print("-" * 100)
                print(
                    f"Running operation: {op_type} --- data: {data} --- mode: {mode} ---"
                )
                if op_type == OperationType.LOAD:
                    result = session.load_lora_adapter(
                        lora_name=data,
                        lora_path=data,
                    )
                elif op_type == OperationType.UNLOAD:
                    result = session.unload_lora_adapter(
                        lora_name=data,
                    )
                elif op_type == OperationType.FORWARD:
                    prompts, adapters = zip(*data)
                    result = session.forward(
                        prompts=list(prompts),
                        lora_paths=list(adapters),
                        max_new_tokens=max_new_tokens,
                    )
                    forward_outputs.append(result)
                elif op_type == OperationType.EXPECT_ERROR:
                    input_data, expected_error = data
                    prompts, adapters = zip(*input_data)
                    result = session.forward(
                        prompts=list(prompts),
                        lora_paths=list(adapters),
                        max_new_tokens=max_new_tokens,
                        expected_error=expected_error,
                    )

            return forward_outputs

    def test_dynamic_adapter_updates(self):
        for case_idx, test_case in enumerate(TEST_CASES, start=1):
            for mode in [
                LoRAUpdateTestSessionMode.ENGINE,
                LoRAUpdateTestSessionMode.SERVER,
            ]:
                print("=" * 100)
                print(f"Starting test case {case_idx} in {mode.value} mode.")
                print("=" * 100)

                print(
                    f"--- Running dynamic update pass with {len(test_case.op_sequence)} operations ---"
                )
                # Test dynamic loading of adapters
                # TODO (lifuhuang): currently at least one LoRA path is required during initialization to enable lora,
                # we should fix this in the future https://github.com/sgl-project/sglang/issues/7463.
                dynamic_output = self._run_operation_sequence(
                    mode=mode,
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

                print("=" * 100)
                print(
                    f"\n--- Running static pass with {len(forward_ops)} operations ---"
                )
                static_output = self._run_operation_sequence(
                    mode=mode,
                    initial_adapters=test_case.all_adapters,
                    base=test_case.base,
                    max_loras_per_batch=test_case.max_loras_per_batch,
                    op_sequence=forward_ops,
                    max_new_tokens=test_case.max_new_tokens,
                )

                print(f"Dynamic output: {dynamic_output}")
                print(f"Static output: {static_output}")
                print("=" * 100)
                self.assertEqual(
                    len(dynamic_output),
                    len(static_output),
                    f"Dynamic output length {len(dynamic_output)} does not match static output length {len(static_output)}",
                )
                for i, (dynamic, static) in enumerate(
                    zip(dynamic_output, static_output), start=1
                ):
                    self.assertEqual(
                        len(dynamic),
                        len(static),
                        f"Output length mismatch at batch {i}:\n- Dynamic={len(dynamic)}\n- Static={len(static)}",
                    )
                    for j, (d_out, s_out) in enumerate(zip(dynamic, static), start=1):
                        d_out = d_out.strip()
                        s_out = s_out.strip()
                        self.assertEqual(
                            d_out,
                            s_out,
                            f"Output mismatch at batch {i}, prompt {j}:\n- Dynamic: '{d_out}'\n- Static: '{s_out}'",
                        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
