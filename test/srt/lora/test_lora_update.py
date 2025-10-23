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

import json
import multiprocessing as mp
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, List, Optional, Union

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
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


@dataclass
class Operation:
    # Operation type, can be LOAD, UNLOAD, FORWARD
    type: OperationType
    # Data associated with the operation. Exact type varies depending on the operation
    data: Optional[Any]
    # If the operation is expected to fail, this is the error message to expect
    expected_error: Optional[str] = None


@dataclass
class TestCase:
    description: str
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    op_sequence: List[Operation]
    initial_adapters: Optional[List[str]] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[List] = None
    max_new_tokens: int = 32
    max_loaded_loras: Optional[int] = None


def create_batch_data(adapters: Union[str, list]) -> List[tuple[str, str]]:
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]


BASIC_TESTS = [
    TestCase(
        description="dynamic lora update with initial lora_paths",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        all_adapters=[
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            "pbevan11/llama-3.1-8b-ocr-correction",
        ],
        initial_adapters=[
            # Testing 3 supported lora-path formats.
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16=Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            {
                "lora_name": "pbevan11/llama-3.1-8b-ocr-correction",
                "lora_path": "pbevan11/llama-3.1-8b-ocr-correction",
                "pinned": False,
            },
        ],
        op_sequence=[
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
                expected_error="already loaded",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
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
                type=OperationType.UNLOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                expected_error="not loaded",
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
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                expected_error="not loaded",
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
                type=OperationType.UNLOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    None,
                ),
            ),
        ],
    ),
    TestCase(
        description="dynamic lora update without initial lora_paths",
        base="meta-llama/Llama-3.1-8B-Instruct",
        enable_lora=True,
        max_lora_rank=256,
        lora_target_modules=["all"],
        max_loras_per_batch=4,
        all_adapters=[
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            "pbevan11/llama-3.1-8b-ocr-correction",
        ],
        op_sequence=[
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
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
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
                expected_error="already loaded",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
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
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                        None,
                    ]
                ),
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        None,
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                        None,
                    ]
                ),
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(None),
            ),
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
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
                        None,
                    ]
                ),
            ),
        ],
    ),
]
TARGET_MODULE_TESTS = [
    TestCase(
        description="Test explicitly specified lora-target-modules.",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        max_lora_rank=64,
        all_adapters=[
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",  # target_modules = q, k, v, o, gate, up, down
            "algoprog/fact-generation-llama-3.1-8b-instruct-lora",  # target_modules = q, k, v, o, gate
        ],
        initial_adapters=["algoprog/fact-generation-llama-3.1-8b-instruct-lora"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.LOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        None,
                    ]
                ),
            ),
        ],
    ),
    TestCase(
        description="Test inferred lora-target-modules - start with larger adapter",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        max_lora_rank=64,
        all_adapters=[
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",  # target_modules = q, k, v, o, gate, up, down
            "algoprog/fact-generation-llama-3.1-8b-instruct-lora",  # target_modules = q, k, v, o, gate
        ],
        initial_adapters=["Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.LOAD,
                data="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        None,
                    ]
                ),
            ),
        ],
    ),
    TestCase(
        description="Test inferred lora-target-modules - start with smaller adapter",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        max_lora_rank=64,
        all_adapters=[
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",  # target_modules = q, k, v, o, gate, up, down
            "algoprog/fact-generation-llama-3.1-8b-instruct-lora",  # target_modules = q, k, v, o, gate
        ],
        initial_adapters=["algoprog/fact-generation-llama-3.1-8b-instruct-lora"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "algoprog/fact-generation-llama-3.1-8b-instruct-lora"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.LOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                expected_error="incompatible",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                        None,
                    ]
                ),
            ),
        ],
    ),
]
MAX_LORA_RANK_TESTS = [
    TestCase(
        description="Test explicitly specified max-lora-rank.",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        max_lora_rank=32,
        all_adapters=[
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",  # r = 4
            "pbevan11/llama-3.1-8b-ocr-correction",  # r = 32
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",  # r = 256
        ],
        initial_adapters=["Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16"
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.LOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "pbevan11/llama-3.1-8b-ocr-correction",
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        None,
                    ]
                ),
            ),
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
                expected_error="incompatible",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                ),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "pbevan11/llama-3.1-8b-ocr-correction",
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        None,
                    ]
                ),
            ),
        ],
    ),
    TestCase(
        description="test implicitly inferred max-lora-rank",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=3,
        all_adapters=[
            "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",  # r = 4
            "pbevan11/llama-3.1-8b-ocr-correction",  # r = 32
            "philschmid/code-llama-3-1-8b-text-to-sql-lora",  # r = 256
        ],
        initial_adapters=["pbevan11/llama-3.1-8b-ocr-correction"],
        op_sequence=[
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("pbevan11/llama-3.1-8b-ocr-correction"),
            ),
            Operation(
                type=OperationType.LOAD,
                data="philschmid/code-llama-3-1-8b-text-to-sql-lora",
                expected_error="incompatible",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data("philschmid/code-llama-3-1-8b-text-to-sql-lora"),
                expected_error="not loaded",
            ),
            Operation(
                type=OperationType.LOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                        "pbevan11/llama-3.1-8b-ocr-correction",
                        None,
                    ]
                ),
            ),
        ],
    ),
]
MAX_LOADED_LORAS_TESTS = [
    TestCase(
        description="Test max_loaded_loras limit",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=2,
        max_loaded_loras=2,
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
                expected_error="Maximum number of loaded LoRA adapters",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            ),
            Operation(
                type=OperationType.LOAD,
                data="pbevan11/llama-3.1-8b-ocr-correction",
            ),
        ],
    ),
]
EVICTION_TESTS = [
    TestCase(
        description="dynamic lora update with evictions",
        base="meta-llama/Llama-3.1-8B-Instruct",
        max_loras_per_batch=2,
        all_adapters=[
            "lora1=philschmid/code-llama-3-1-8b-text-to-sql-lora",
            "lora2=Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
            "lora3=pbevan11/llama-3.1-8b-ocr-correction",
        ],
        enable_lora=True,
        max_lora_rank=256,
        lora_target_modules=["all"],
        op_sequence=[
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora1",
                    "lora_path": "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                    "pinned": True,
                },
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora2",
                    "lora_path": "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                    "pinned": True,
                },
                expected_error="starvation",
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora2",
                    "lora_path": "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
                    "pinned": False,
                },
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora3",
                    "lora_path": "pbevan11/llama-3.1-8b-ocr-correction",
                    "pinned": False,
                },
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="lora1",
            ),
            Operation(
                type=OperationType.UNLOAD,
                data="lora3",
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora3",
                    "lora_path": "pbevan11/llama-3.1-8b-ocr-correction",
                    "pinned": True,
                },
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora1",
                    "lora_path": "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                    "pinned": True,
                },
                expected_error="starvation",
            ),
            Operation(
                type=OperationType.LOAD,
                data={
                    "lora_name": "lora1",
                    "lora_path": "philschmid/code-llama-3-1-8b-text-to-sql-lora",
                    "pinned": False,
                },
            ),
            # pinned: lora3
            # unpinned: lora1, lora2
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "lora1",
                        "lora2",
                    ]
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "lora1",
                        "lora3",
                    ]
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "lora1",
                        "lora2",
                    ]
                ),
            ),
            Operation(
                type=OperationType.FORWARD,
                data=create_batch_data(
                    [
                        "lora1",
                        "lora2",
                        None,
                    ]
                ),
            ),
        ],
    ),
]

ALL_TESTS = (
    BASIC_TESTS
    + TARGET_MODULE_TESTS
    + MAX_LORA_RANK_TESTS
    + MAX_LOADED_LORAS_TESTS
    + EVICTION_TESTS
)


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
        lora_paths: List[Union[str, dict]],
        max_loras_per_batch: int,
        max_loaded_loras: Optional[int] = None,
        max_lora_rank: Optional[int],
        enable_lora: Optional[bool] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_backend: str = "triton",
        disable_cuda_graph: bool = False,
        cuda_graph_max_bs: int = 4,
    ):
        self.testcase = testcase
        self.model_path = model_path
        self.lora_paths = lora_paths
        self.max_lora_rank = max_lora_rank
        self.lora_target_modules = lora_target_modules
        self.max_loras_per_batch = max_loras_per_batch
        self.max_loaded_loras = max_loaded_loras
        self.lora_backend = lora_backend
        self.disable_cuda_graph = disable_cuda_graph
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_lora = enable_lora

        self.expected_adapters = set()
        if self.lora_paths:
            for adapter in self.lora_paths:
                if isinstance(adapter, dict):
                    lora_name = adapter["lora_name"]
                elif "=" in adapter:
                    lora_name = adapter.split("=")[0]
                else:
                    lora_name = adapter
                self.expected_adapters.add(lora_name)

        self.handle = None  # Will be set in __enter__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't suppress exceptions by default
        return False

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: Optional[str] = None,
        expected_error: Optional[str] = None,
    ):
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
        expected_error: Optional[str] = None,
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
            max_lora_rank=self.max_lora_rank,
            lora_target_modules=self.lora_target_modules,
            lora_backend=self.lora_backend,
            torch_dtype=torch.float16,
            mem_fraction_static=MEM_FRACTION_STATIC,
            max_loras_per_batch=self.max_loras_per_batch,
            max_loaded_loras=self.max_loaded_loras,
            disable_cuda_graph=self.disable_cuda_graph,
            cuda_graph_max_bs=self.cuda_graph_max_bs,
            enable_lora=self.enable_lora,
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

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: Optional[str] = None,
        expected_error: Optional[str] = None,
        pinned: bool = False,
    ):
        """
        Load a LoRA adapter by name and path.
        """
        if lora_path is None:
            lora_path = lora_name

        response = self.handle.load_lora_adapter(
            lora_name=lora_name,
            lora_path=lora_path,
            pinned=pinned,
        )
        if expected_error:
            self.testcase.assertFalse(
                response.success, f"Expected failure for {lora_name}, but got success."
            )
            self.testcase.assertIn(
                expected_error,
                response.error_message,
                f"Expected error message to contain '{expected_error}', but got '{response.error_message}'",
            )
            print(f"Received error as expected: {response.error_message}")
        else:
            self.expected_adapters.add(lora_name)
            self.testcase.assertTrue(
                response.success,
                f"Failed to load LoRA adapter {lora_name}: {response.error_message}",
            )
            loaded_adapters = set(response.loaded_adapters)
            print(f"loaded_adapters: {loaded_adapters}")
            self.testcase.assertEqual(
                loaded_adapters,
                self.expected_adapters,
                f"Expected loaded adapters to be {self.expected_adapters}, but got {loaded_adapters}",
            )

    def unload_lora_adapter(self, lora_name: str):
        """
        Unload a LoRA adapter by name.
        """
        self.expected_adapters.remove(lora_name)

        response = self.handle.unload_lora_adapter(
            lora_name=lora_name,
        )
        self.testcase.assertTrue(
            response.success,
            f"Failed to unload LoRA adapter {lora_name}: {response.error_message}",
        )
        loaded_adapters = set(response.loaded_adapters)

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(
            loaded_adapters,
            self.expected_adapters,
            f"Expected loaded adapters to be {self.expected_adapters}, but got {loaded_adapters}",
        )

    def forward(
        self,
        prompts: List[str],
        lora_paths: List[str],
        max_new_tokens: int = 32,
        expected_error: Optional[str] = None,
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
                self.testcase.assertIn(
                    expected_error,
                    error_message,
                    f"Expected error message to contain '{expected_error}', but got '{error_message}'",
                )
                print(f"Received error as expected: {error_message}")
                return error_message

            raise e

        self.testcase.assertEqual(
            len(response.output_strs),
            len(prompts),
            f"Expected {len(prompts)} outputs, but got {len(response.output_strs)}",
        )
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
            "--max-loras-per-batch",
            str(self.max_loras_per_batch),
            "--lora-backend",
            self.lora_backend,
            "--random-seed",
            "42",
            "--max-running-request",
            "1",
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--disable-radix-cache",
        ]
        if self.enable_lora:
            other_args.append("--enable-lora")
        if self.lora_paths:
            other_args.append("--lora-paths")
            for lora_path in self.lora_paths:
                if isinstance(lora_path, dict):
                    lora_path = json.dumps(lora_path)
                other_args.append(lora_path)
        if self.disable_cuda_graph:
            other_args.append("--disable-cuda-graph")
        if self.max_lora_rank is not None:
            other_args.extend(["--max-lora-rank", str(self.max_lora_rank)])
        if self.lora_target_modules is not None:
            other_args.extend(["--lora-target-modules"] + self.lora_target_modules)
        if self.max_loaded_loras is not None:
            other_args.extend(["--max-loaded-loras", str(self.max_loaded_loras)])

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

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: Optional[str] = None,
        expected_error: Optional[str] = None,
        pinned: bool = False,
    ):
        """
        Load a LoRA adapter by name and path.
        """
        if lora_path is None:
            lora_path = lora_name

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path, "pinned": pinned},
        )
        if expected_error:
            self.testcase.assertEqual(
                response.status_code,
                400,
                f"Expected error for {lora_name}, but got success.",
            )
            self.testcase.assertIn(
                expected_error,
                response.text,
                f"Expected error message to contain '{expected_error}', but got '{response.text}'",
            )
            print(f"Received error as expected: {response.text}")
        else:
            self.expected_adapters.add(lora_name)
            self.testcase.assertTrue(
                response.ok, f"Failed to load LoRA adapter {lora_name}: {response.text}"
            )
            loaded_adapters = set(response.json()["loaded_adapters"])
            print(f"loaded_adapters: {loaded_adapters}")
            self.testcase.assertEqual(
                loaded_adapters,
                self.expected_adapters,
                f"Expected loaded adapters to be {self.expected_adapters}, but got {loaded_adapters}",
            )

    def unload_lora_adapter(self, lora_name: str):
        """
        Unload a LoRA adapter by name.
        """
        self.expected_adapters.remove(lora_name)

        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": lora_name},
        )
        self.testcase.assertTrue(
            response.ok, f"Failed to unload LoRA adapter {lora_name}: {response.text}"
        )
        loaded_adapters = set(response.json()["loaded_adapters"])

        print(f"loaded_adapters: {loaded_adapters}")
        self.testcase.assertEqual(
            loaded_adapters,
            self.expected_adapters,
            f"Expected loaded adapters to be {self.expected_adapters}, but got {loaded_adapters}",
        )

    def forward(
        self,
        prompts: List[str],
        lora_paths: List[str],
        max_new_tokens: int = 32,
        expected_error: Optional[str] = None,
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
            self.testcase.assertEqual(
                response.status_code,
                400,
                f"Expected error for forward pass, but got success: {response.text}",
            )
            self.testcase.assertIn(
                expected_error,
                response.text,
                f"Expected error message to contain '{expected_error}', but got '{response.text}'",
            )
            output = response.text
            print(f"Received error as expected: {response.text}")
            return output
        else:
            self.testcase.assertTrue(
                response.ok, f"Failed to generate text: {response.text}"
            )
            output = [r["text"] for r in response.json()]
            self.testcase.assertEqual(
                len(output),
                len(prompts),
                f"Expected {len(prompts)} outputs, but got {len(output)}",
            )
            print(f"output_strs: {output}")
            return output


# Factory function to create the appropriate LoRA test session based on mode
def LoRAUpdateTestSession(
    testcase: Optional[TestCase],
    mode: LoRAUpdateTestSessionMode,
    **kwargs: Any,
):
    if mode == LoRAUpdateTestSessionMode.ENGINE:
        return LoRAUpdateEngineTestSession(testcase=testcase, **kwargs)
    elif mode == LoRAUpdateTestSessionMode.SERVER:
        return LoRAUpdateServerTestSession(testcase=testcase, **kwargs)
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
        initial_adapters: List[Union[str, dict]],
        op_sequence: List[Operation],
        max_loras_per_batch: int,
        max_loaded_loras: Optional[int] = None,
        enable_lora: Optional[bool] = None,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
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
            max_loaded_loras=max_loaded_loras,
            max_lora_rank=max_lora_rank,
            lora_target_modules=lora_target_modules,
            enable_lora=enable_lora,
        ) as session:
            for op in op_sequence:
                op_type = op.type
                data = op.data
                expected_error = op.expected_error
                print("-" * 100)
                print(
                    f"Running operation: {op_type} --- data: {data} --- mode: {mode} ---"
                )
                if op_type == OperationType.LOAD:
                    if isinstance(data, str):
                        adapter_info = {
                            "lora_name": data,
                            "lora_path": data,
                            "pinned": False,
                        }
                    else:
                        adapter_info = data

                    result = session.load_lora_adapter(
                        expected_error=expected_error,
                        **adapter_info,
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
                        expected_error=expected_error,
                    )
                    if not expected_error:
                        forward_outputs.append(result)

            return forward_outputs

    def _run_dynamic_adapter_updates(
        self, mode: LoRAUpdateTestSessionMode, test_cases: Iterable[TestCase]
    ):
        for case_idx, test_case in enumerate(test_cases, start=1):
            print("=" * 100)
            print(
                f"Starting test case {case_idx} in {mode.value} mode. Test description: {test_case.description}"
            )
            print("=" * 100)

            print(
                f"--- Running dynamic update pass with {len(test_case.op_sequence)} operations ---"
            )
            # Test dynamic loading of adapters
            dynamic_output = self._run_operation_sequence(
                mode=mode,
                initial_adapters=test_case.initial_adapters,
                enable_lora=test_case.enable_lora,
                base=test_case.base,
                max_loras_per_batch=test_case.max_loras_per_batch,
                max_loaded_loras=test_case.max_loaded_loras,
                op_sequence=test_case.op_sequence,
                max_new_tokens=test_case.max_new_tokens,
                max_lora_rank=test_case.max_lora_rank,
                lora_target_modules=test_case.lora_target_modules,
            )

            # static loading
            forward_ops = [
                x
                for x in test_case.op_sequence
                if x.type == OperationType.FORWARD and x.expected_error is None
            ]

            if not forward_ops:
                print(
                    f"No forward operations found in test case {case_idx}. Skipping static pass."
                )
                continue

            print("=" * 100)
            print(f"\n--- Running static pass with {len(forward_ops)} operations ---")
            static_output = self._run_operation_sequence(
                mode=mode,
                initial_adapters=test_case.all_adapters,
                enable_lora=test_case.enable_lora,
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

    def test_dynamic_lora_update_engine(self):
        """
        Test dynamic LoRA updates in engine mode.
        """
        test_cases = BASIC_TESTS if is_in_ci() else ALL_TESTS
        self._run_dynamic_adapter_updates(
            mode=LoRAUpdateTestSessionMode.ENGINE,
            test_cases=test_cases,
        )

    def test_dynamic_lora_update_server(self):
        """
        Test dynamic LoRA updates in server mode.
        """
        test_cases = BASIC_TESTS if is_in_ci() else ALL_TESTS
        self._run_dynamic_adapter_updates(
            mode=LoRAUpdateTestSessionMode.SERVER, test_cases=test_cases
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
