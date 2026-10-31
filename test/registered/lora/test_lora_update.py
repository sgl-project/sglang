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
"""Dynamic adapter updates must match an eviction-free startup reference
(ROUGE-L >= 0.9 per prompt). Reusing an all-module slot for a q,v-only adapter
must clear stale weights; reusing a name must serve its current adapter.
/v1/models must list exactly what is loaded.
"""

import json
import multiprocessing as mp
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    calculate_rouge_l,
    popen_launch_server,
)

register_cuda_ci(
    est_time=250,
    stage="base-b",
    runner_config="1-gpu-large",
)
register_amd_ci(
    est_time=380,
    suite="stage-b-test-1-gpu-large-amd",
)

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]

MEM_FRACTION_STATIC = 0.8
ROUGE_L_TOL = 0.9

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# target_modules = q, k, v, o, gate, up, down
SQL_ADAPTER = "philschmid/code-llama-3-1-8b-text-to-sql-lora"
TOPIC_ADAPTER = "nvidia/llama-3.1-nemoguard-8b-topic-control"
OCR_ADAPTER = "pbevan11/llama-3.1-8b-ocr-correction"
# target_modules = q, v: loaded into a slot an all-module adapter used, the
# untouched k/o/gate/up/down planes must be zeroed, not inherited.
QV_ADAPTER = "faridlazuarda/valadapt-llama-3.1-8B-it-chinese"
SHARED_NAME = "shared-name"


class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    FORWARD = "forward"
    LIST_MODELS = "list_models"


@dataclass
class Operation:
    type: OperationType
    # LOAD: adapter path or {"lora_name", "lora_path", "pinned"}; UNLOAD: name;
    # FORWARD: [(prompt, adapter name or None)].
    data: Optional[Any] = None
    # If the operation is expected to fail, the error message to expect.
    expected_error: Optional[str] = None
    # FORWARD only: index (among the recorded forwards) of an earlier forward
    # whose outputs this one must reproduce; checked in the dynamic pass and
    # excluded from the static comparison.
    same_output_as: Optional[int] = None


@dataclass
class TestCase:
    description: str
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    op_sequence: List[Operation]
    initial_adapters: Optional[List[Union[str, dict]]] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[List] = None
    max_new_tokens: int = 32


def create_batch_data(adapters: Union[str, list, None]) -> List[tuple]:
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]


def _load(name_or_dict, expected_error=None):
    return Operation(OperationType.LOAD, name_or_dict, expected_error)


def _unload(name):
    return Operation(OperationType.UNLOAD, name)


def _forward(adapters, same_output_as=None):
    return Operation(
        OperationType.FORWARD,
        create_batch_data(adapters),
        same_output_as=same_output_as,
    )


LIST_MODELS = Operation(OperationType.LIST_MODELS)

# Recorded forwards (the ones compared with the static pass) are numbered in
# the comments; `same_output_as` refers to those numbers.
DYNAMIC_UPDATE_CASE = TestCase(
    description="dynamic load/unload/reload, slot reuse under pressure, name reuse",
    base=BASE_MODEL,
    enable_lora=True,
    max_lora_rank=256,
    lora_target_modules=["all"],
    # Four adapters plus base rows over three slots: eviction by construction.
    max_loras_per_batch=3,
    all_adapters=[SQL_ADAPTER, TOPIC_ADAPTER, OCR_ADAPTER, QV_ADAPTER],
    initial_adapters=[
        # The three supported lora-path formats.
        SQL_ADAPTER,
        f"{TOPIC_ADAPTER}={TOPIC_ADAPTER}",
        {"lora_name": OCR_ADAPTER, "lora_path": OCR_ADAPTER, "pinned": False},
    ],
    op_sequence=[
        _load(SQL_ADAPTER, expected_error="already loaded"),
        _unload(SQL_ADAPTER),
        _load(SQL_ADAPTER),
        LIST_MODELS,
        _forward([SQL_ADAPTER, TOPIC_ADAPTER, OCR_ADAPTER]),  # 0
        # A fourth adapter: its first forward evicts one of the three and the
        # q,v-only weights land in a slot that held an all-module adapter.
        _load(QV_ADAPTER),
        _forward(QV_ADAPTER),  # 1
        _forward(SQL_ADAPTER),  # 2: the evicted all-module adapter comes back
        _forward([None, TOPIC_ADAPTER, OCR_ADAPTER, None]),  # 3
        _unload(TOPIC_ADAPTER),
        _unload(OCR_ADAPTER),
        LIST_MODELS,
        _forward(TOPIC_ADAPTER),  # 4: implicit reload by path
        _forward(OCR_ADAPTER),  # 5
        _forward(None),  # 6: base only after unloads (freed slots replayed)
        _load(TOPIC_ADAPTER, expected_error="already loaded"),
        # Name reuse: the same name must serve whatever adapter is behind it.
        _unload(QV_ADAPTER),
        _load({"lora_name": SHARED_NAME, "lora_path": QV_ADAPTER, "pinned": False}),
        _forward(SHARED_NAME, same_output_as=1),
        _unload(SHARED_NAME),
        _unload(SQL_ADAPTER),
        _load({"lora_name": SHARED_NAME, "lora_path": SQL_ADAPTER, "pinned": False}),
        _forward(SHARED_NAME, same_output_as=2),
        _unload(SHARED_NAME),
        _load(SQL_ADAPTER),
        _forward([SQL_ADAPTER, TOPIC_ADAPTER, OCR_ADAPTER, None]),  # 7
    ],
)


class LoRAUpdateServerTestSession:
    """A standalone server with dynamic LoRA loading; tracks what is loaded."""

    def __init__(
        self,
        *,
        testcase: CustomTestCase,
        model_path: str,
        lora_paths: Optional[List[Union[str, dict]]],
        max_loras_per_batch: int,
        max_lora_rank: Optional[int],
        enable_lora: Optional[bool] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_backend: str = "csgmv",
        cuda_graph_max_bs_decode: int = 4,
    ):
        self.testcase = testcase
        self.model_path = model_path
        self.lora_paths = lora_paths
        self.max_lora_rank = max_lora_rank
        self.lora_target_modules = lora_target_modules
        self.max_loras_per_batch = max_loras_per_batch
        self.lora_backend = lora_backend
        self.cuda_graph_max_bs_decode = cuda_graph_max_bs_decode
        self.enable_lora = enable_lora
        # name -> path of every adapter the server should currently list
        self.expected_adapters: Dict[str, str] = {}
        for adapter in self.lora_paths or []:
            if isinstance(adapter, dict):
                self.expected_adapters[adapter["lora_name"]] = adapter["lora_path"]
            elif "=" in adapter:
                name, path = adapter.split("=", 1)
                self.expected_adapters[name] = path
            else:
                self.expected_adapters[adapter] = adapter
        self.handle = None

    def __enter__(self):
        other_args = [
            "--cuda-graph-max-bs-decode",
            str(self.cuda_graph_max_bs_decode),
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
        if self.max_lora_rank is not None:
            other_args.extend(["--max-lora-rank", str(self.max_lora_rank)])
        if self.lora_target_modules is not None:
            other_args.extend(["--lora-target-modules"] + self.lora_target_modules)

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
        return False

    def _assert_loaded(self, response):
        self.testcase.assertTrue(response.ok, response.text)
        loaded_adapters = set(response.json()["loaded_adapters"])
        self.testcase.assertEqual(
            loaded_adapters,
            set(self.expected_adapters),
            f"Expected loaded adapters {set(self.expected_adapters)}, got {loaded_adapters}",
        )

    def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: Optional[str] = None,
        expected_error: Optional[str] = None,
        pinned: bool = False,
    ):
        if lora_path is None:
            lora_path = lora_name
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path, "pinned": pinned},
        )
        if expected_error:
            self.testcase.assertEqual(response.status_code, 400, response.text)
            self.testcase.assertIn(expected_error, response.text)
            print(f"Received error as expected: {response.text}")
            return
        self.expected_adapters[lora_name] = lora_path
        self._assert_loaded(response)

    def unload_lora_adapter(self, lora_name: str):
        del self.expected_adapters[lora_name]
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
            json={"lora_name": lora_name},
        )
        self._assert_loaded(response)

    def forward(
        self, prompts: List[str], lora_paths: List[Optional[str]], max_new_tokens: int
    ):
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
        self.testcase.assertTrue(
            response.ok, f"Failed to generate text: {response.text}"
        )
        output = [r["text"] for r in response.json()]
        self.testcase.assertEqual(len(output), len(prompts))
        print(f"output_strs: {output}")
        # A forward on an unloaded adapter path loads it implicitly.
        for lora_path in lora_paths:
            if lora_path is not None:
                self.expected_adapters.setdefault(lora_path, lora_path)
        return output

    def assert_models_listing(self):
        response = requests.get(DEFAULT_URL_FOR_TEST + "/v1/models")
        self.testcase.assertTrue(response.ok, response.text)
        models = response.json()
        self.testcase.assertEqual(models["object"], "list")
        base = [m for m in models["data"] if not m.get("parent")]
        adapters = {m["id"]: m for m in models["data"] if m.get("parent")}
        self.testcase.assertEqual(len(base), 1, models)
        self.testcase.assertIn("meta-llama", base[0]["id"].lower())
        self.testcase.assertEqual(set(adapters), set(self.expected_adapters), models)
        for name, path in self.expected_adapters.items():
            self.testcase.assertEqual(adapters[name]["root"], path)
            self.testcase.assertEqual(adapters[name]["parent"], base[0]["id"])


class TestLoRADynamicUpdate(CustomTestCase):
    def _run_operation_sequence(
        self,
        test_case: TestCase,
        initial_adapters,
        op_sequence: List[Operation],
        max_loras_per_batch: int,
    ) -> List[List[str]]:
        """Run the operations on one server; return the recorded forward outputs."""
        forward_outputs = []
        with LoRAUpdateServerTestSession(
            testcase=self,
            model_path=test_case.base,
            lora_paths=initial_adapters,
            max_loras_per_batch=max_loras_per_batch,
            max_lora_rank=test_case.max_lora_rank,
            lora_target_modules=test_case.lora_target_modules,
            enable_lora=test_case.enable_lora,
        ) as session:
            for op in op_sequence:
                print("-" * 100)
                print(f"Running operation: {op.type} --- data: {op.data}")
                if op.type == OperationType.LOAD:
                    adapter_info = (
                        op.data
                        if isinstance(op.data, dict)
                        else {
                            "lora_name": op.data,
                            "lora_path": op.data,
                            "pinned": False,
                        }
                    )
                    session.load_lora_adapter(
                        expected_error=op.expected_error, **adapter_info
                    )
                elif op.type == OperationType.UNLOAD:
                    session.unload_lora_adapter(lora_name=op.data)
                elif op.type == OperationType.LIST_MODELS:
                    session.assert_models_listing()
                elif op.type == OperationType.FORWARD:
                    prompts, adapters = zip(*op.data)
                    output = session.forward(
                        prompts=list(prompts),
                        lora_paths=list(adapters),
                        max_new_tokens=test_case.max_new_tokens,
                    )
                    if op.same_output_as is None:
                        forward_outputs.append(output)
                    else:
                        self._assert_outputs_match(
                            forward_outputs[op.same_output_as],
                            output,
                            f"forward {op.same_output_as} vs the reused name",
                        )
        return forward_outputs

    def _assert_outputs_match(self, expected: List[str], actual: List[str], label: str):
        self.assertEqual(len(expected), len(actual), label)
        for j, (e_out, a_out) in enumerate(zip(expected, actual), start=1):
            rouge_score = calculate_rouge_l([e_out.strip()], [a_out.strip()])[0]
            self.assertGreaterEqual(
                rouge_score,
                ROUGE_L_TOL,
                f"ROUGE-L {rouge_score} below {ROUGE_L_TOL} ({label}, prompt {j}):"
                f"\n- expected: '{e_out}'\n- actual:   '{a_out}'",
            )

    def test_dynamic_lora_update_server(self):
        test_case = DYNAMIC_UPDATE_CASE
        recorded = [
            op
            for op in test_case.op_sequence
            if op.type == OperationType.FORWARD and op.same_output_as is None
        ]
        # Eviction pressure by construction: more distinct adapters (plus the
        # base slot) are served than the pool has slots.
        served = {adapter for op in recorded for _, adapter in op.data} | {None}
        self.assertGreater(len(served), test_case.max_loras_per_batch)

        print("=" * 100)
        print(f"Dynamic pass: {test_case.description}")
        dynamic_output = self._run_operation_sequence(
            test_case,
            test_case.initial_adapters,
            test_case.op_sequence,
            max_loras_per_batch=test_case.max_loras_per_batch,
        )
        print("=" * 100)
        print(f"Static pass: {len(recorded)} forwards with every adapter preloaded")
        # The reference never evicts: every adapter and the base slot stay
        # resident, so weights leaking through a reused slot in the dynamic
        # pass cannot be reproduced here and show up as a mismatch.
        static_output = self._run_operation_sequence(
            test_case,
            test_case.all_adapters,
            recorded,
            max_loras_per_batch=len(test_case.all_adapters) + 1,
        )

        self.assertEqual(len(dynamic_output), len(static_output))
        for i, (dynamic, static) in enumerate(zip(dynamic_output, static_output)):
            self._assert_outputs_match(static, dynamic, f"recorded forward {i}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
