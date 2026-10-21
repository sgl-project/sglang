"""Static compressed-tensors loading and decoding on one Ascend A3."""

import json
import math
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.static_int8_model_fixture import (
    MAX_LOGPROB_ERROR,
    NEW_TOKENS,
    prepare_static_int8_qwen,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNPUStaticCompressedTensorsInt8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="sglang-static-int8-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model_path = Path(cls.directory.name) / "checkpoint"
        base_model = os.environ.get(
            "SGLANG_NPU_STATIC_INT8_BASE_MODEL", QWEN3_0_6B_WEIGHTS_PATH
        )
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            cls.prompts = prepare_static_int8_qwen(base_model, cls.model_path)
        finally:
            torch.set_num_threads(threads)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.references = cls._collect_reference()

    @staticmethod
    def _server_args(eager):
        args = [
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--attention-backend",
            "ascend",
            "--sampling-backend",
            "pytorch",
            "--context-length",
            "512",
            "--max-total-tokens",
            "2048",
            "--mem-fraction-static",
            "0.45",
            "--chunked-prefill-size",
            "-1",
            "--disable-overlap-schedule",
            "--disable-radix-cache",
            "--cuda-graph-max-bs-decode",
            "4",
            "--max-running-requests",
            "4",
        ]
        if eager:
            args.append("--disable-cuda-graph")
        return args

    @classmethod
    def _generate(cls, input_ids):
        response = requests.post(
            cls.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    @classmethod
    def _collect_reference(cls):
        config = Path(cls.directory.name) / "reference-config.json"
        receipt = Path(cls.directory.name) / "reference-receipt.json"
        config.write_text(
            json.dumps(
                {
                    "checkpoint": str(cls.model_path),
                    "first_input_ids": cls.prompts[0]["input_ids"],
                    "receipt": str(receipt),
                }
            )
        )
        process = popen_launch_server(
            str(cls.model_path),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._server_args(eager=True),
            env={
                "SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang.test.ascend.static_int8_reference",
                "SGLANG_STATIC_INT8_REFERENCE_CONFIG": str(config),
            },
        )
        try:
            single = [cls._generate(row["input_ids"]) for row in cls.prompts]
            batch = cls._generate([row["input_ids"] for row in cls.prompts])
            references = {}
            for kind, responses in (("single", single), ("batch", batch)):
                if len(responses) != len(cls.prompts):
                    raise ValueError("Missing full-model integer reference output")
                references[kind] = []
                for prompt, response in zip(cls.prompts, responses):
                    logprobs = [
                        float(row[0])
                        for row in response["meta_info"]["output_token_logprobs"]
                    ]
                    if (
                        len(response["output_ids"]) != NEW_TOKENS
                        or len(logprobs) != NEW_TOKENS
                        or not all(math.isfinite(x) for x in logprobs)
                    ):
                        raise ValueError("Incomplete/nonfinite integer reference")
                    references[kind].append(
                        {
                            **prompt,
                            "output_ids": response["output_ids"],
                            "logprobs": logprobs,
                        }
                    )
            provenance = json.loads(receipt.read_text())
            counts = Counter(
                row["module"] for row in provenance["cpu_integer_linear_calls"]
            )
            token_counts = Counter()
            for row in provenance["cpu_integer_linear_calls"]:
                token_counts[row["module"]] += math.prod(row["input_shape"][:-1])
            expected_tokens = 2 * sum(
                len(row["input_ids"]) for row in cls.prompts
            ) + 4 * (NEW_TOKENS - 1)
            if (
                len(provenance["loader_checks"]) != 675
                or not all(row["passed"] for row in provenance["loader_checks"])
                or provenance["checkpoint_logical_linears"] != 196
                or provenance["fused_modules"] != 112
                or len(counts) != 112
                or not all(24 <= count <= 32 for count in counts.values())
                or set(token_counts.values()) != {expected_tokens}
            ):
                raise ValueError("Incomplete serialized-checkpoint integer reference")
            print(
                json.dumps(
                    {
                        "type": "static_int8_reference",
                        "loader_checks": 675,
                        "cpu_integer_calls": sum(counts.values()),
                        "checkpoint_weight_sha256": provenance[
                            "checkpoint_weight_sha256"
                        ],
                    }
                ),
                flush=True,
            )
            return references
        finally:
            kill_process_tree(process.pid)

    def _check_response(self, response, expected, request_kind):
        self.assertEqual(response["output_ids"], expected["output_ids"])
        actual = [
            float(item[0]) for item in response["meta_info"]["output_token_logprobs"]
        ]
        self.assertEqual(len(actual), NEW_TOKENS)
        self.assertEqual(len(expected["logprobs"]), NEW_TOKENS)
        self.assertTrue(
            all(math.isfinite(value) for value in actual + expected["logprobs"])
        )
        max_error = max(
            abs(left - right) for left, right in zip(actual, expected["logprobs"])
        )
        self.assertLessEqual(max_error, MAX_LOGPROB_ERROR)
        print(
            json.dumps(
                {
                    "type": "static_int8_e2e",
                    "execution": self.execution_mode,
                    "request_kind": request_kind,
                    "input_ids": expected["input_ids"],
                    "output_ids": response["output_ids"],
                    "reference_output_ids": expected["output_ids"],
                    "logprobs": actual,
                    "reference_logprobs": expected["logprobs"],
                    "max_abs_logprob_error": max_error,
                }
            ),
            flush=True,
        )

    def test_eager_same_checkpoint(self):
        self._run_same_checkpoint(eager=True)

    def test_graph_same_checkpoint(self):
        self._run_same_checkpoint(eager=False)

    def _run_same_checkpoint(self, eager):
        self.execution_mode = "eager" if eager else "graph"
        process = popen_launch_server(
            str(self.model_path),
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self._server_args(eager),
            env={
                "SGLANG_EXTERNAL_MODEL_PACKAGE": "",
                "SGLANG_STATIC_INT8_REFERENCE_CONFIG": "",
            },
        )
        try:
            if not eager:
                info = requests.get(self.base_url + "/get_server_info", timeout=60)
                info.raise_for_status()
                states = info.json()["internal_states"]
                if isinstance(states, dict):
                    states = [states]
                self.assertTrue(
                    any(
                        float(
                            state.get("memory_usage", {})
                            .get("graph", {})
                            .get("decode", 0)
                        )
                        > 0
                        for state in states
                    )
                )
            for expected in self.references["single"]:
                self._check_response(
                    self._generate(expected["input_ids"]), expected, "single"
                )
            batch = self._generate(
                [row["input_ids"] for row in self.references["batch"]]
            )
            self.assertEqual(len(batch), len(self.references["batch"]))
            for actual, expected in zip(batch, self.references["batch"]):
                self._check_response(actual, expected, "batch")
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
