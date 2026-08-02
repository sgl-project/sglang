"""TP4/DCP4 parity and shared harness for symmetric-memory DCP E2E tests."""

import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-c", runner_config="4-gpu-h100")

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
LOGPROB_ABS_TOLERANCE = 0.1
ACTIONABLE_FP8_FALLBACK = "--dcp-comm-backend a2a"

# The first sequence is deliberately shorter than every tested DCP world size.
# The ragged batch also covers ordinary non-empty layouts and a graph batch of 4.
INPUT_IDS = [
    [1],
    [1, 100],
    [1, 100, 101],
    [1, 100, 101, 102, 103, 104, 105, 106],
]


class ServerRequestError(RuntimeError):
    """The server explicitly rejected a generate request."""


@dataclass(frozen=True)
class CaseResult:
    outputs: Optional[list[dict]] = None
    unsupported_reason: Optional[str] = None


def _has_required_cuda_gpus(required_gpus: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= required_gpus


class SymmA2ATestBase(CustomTestCase):
    """Reusable sequential server fixture with strict failure classification."""

    base_url = DEFAULT_URL_FOR_TEST
    required_gpus = 1
    _active_process = None

    @classmethod
    def setUpClass(cls):
        if not _has_required_cuda_gpus(cls.required_gpus):
            raise unittest.SkipTest(
                f"symm_a2a E2E correctness requires {cls.required_gpus} CUDA GPUs"
            )

    @classmethod
    def _stop_active_process(cls):
        process = getattr(cls, "_active_process", None)
        cls._active_process = None
        if process:
            kill_process_tree(process.pid, wait_timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls._stop_active_process()

    @staticmethod
    def _server_args(
        *,
        tp_size: int,
        backend: str,
        disable_cuda_graph: bool,
        kv_cache_dtype: str,
    ) -> list[str]:
        args = [
            "--tp-size",
            str(tp_size),
            "--dcp-size",
            str(tp_size),
            "--dcp-comm-backend",
            backend,
            "--attention-backend",
            "flashinfer",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--random-seed",
            "0",
            "--trust-remote-code",
            "--disable-radix-cache",
            "--max-running-requests",
            "8",
            "--mem-fraction-static",
            "0.70",
        ]
        if disable_cuda_graph:
            args.append("--disable-cuda-graph")
        else:
            args.extend(["--cuda-graph-max-bs-decode", str(len(INPUT_IDS))])
        return args

    @staticmethod
    def _request(base_url: str) -> list[dict]:
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": INPUT_IDS,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 1,
                "logprob_start_len": 0,
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise ServerRequestError(
                f"generate failed with status {response.status_code}: "
                f"{response.text[:4000]}"
            )
        outputs = response.json()
        if not isinstance(outputs, list) or len(outputs) != len(INPUT_IDS):
            raise AssertionError(
                f"expected {len(INPUT_IDS)} batched outputs, got {outputs!r}"
            )
        return outputs

    @staticmethod
    def _log_tail(stdout_path: Path, stderr_path: Path) -> str:
        chunks = []
        for path in (stdout_path, stderr_path):
            if path.exists():
                chunks.append(path.read_text(errors="replace")[-8000:])
        return "\n".join(chunks)

    @staticmethod
    def _is_actionable_fp8_failure(message: str) -> bool:
        lowered = message.lower()
        names_fp8 = "fp8" in lowered or "float8" in lowered
        names_output_dtype = (
            "attention output" in lowered
            or "only supports fp16 and bf16" in lowered
            or "only supports fp16/bf16" in lowered
        )
        return (
            "symm_a2a" in lowered
            and names_fp8
            and names_output_dtype
            and ACTIONABLE_FP8_FALLBACK in lowered
        )

    def _run_case(
        self,
        *,
        tp_size: int,
        backend: str,
        disable_cuda_graph: bool,
        kv_cache_dtype: str = "auto",
        allow_actionable_fp8_failure: bool = False,
    ) -> CaseResult:
        with tempfile.TemporaryDirectory(prefix="sglang-symm-a2a-") as temp_dir:
            stdout_path = Path(temp_dir) / "server.stdout"
            stderr_path = Path(temp_dir) / "server.stderr"
            with stdout_path.open("w+") as stdout, stderr_path.open("w+") as stderr:
                try:
                    self.__class__._active_process = popen_launch_server(
                        MODEL,
                        self.base_url,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
                        other_args=self._server_args(
                            tp_size=tp_size,
                            backend=backend,
                            disable_cuda_graph=disable_cuda_graph,
                            kv_cache_dtype=kv_cache_dtype,
                        ),
                        return_stdout_stderr=(stdout, stderr),
                    )
                    # The first request warms the server and graph; compare the
                    # second request so the enabled case necessarily replays it.
                    self._request(self.base_url)
                    return CaseResult(outputs=self._request(self.base_url))
                except (TimeoutError, requests.Timeout):
                    raise
                except Exception as exc:
                    stdout.flush()
                    stderr.flush()
                    details = f"{exc}\n{self._log_tail(stdout_path, stderr_path)}"
                    is_explicit_rejection = isinstance(
                        exc, ServerRequestError
                    ) or "server unexpectedly exited" in str(exc).lower()
                    if (
                        allow_actionable_fp8_failure
                        and is_explicit_rejection
                        and self._is_actionable_fp8_failure(details)
                    ):
                        return CaseResult(unsupported_reason=details)
                    raise AssertionError(
                        f"server case failed (tp={tp_size}, backend={backend}, "
                        f"disable_cuda_graph={disable_cuda_graph}, "
                        f"kv_cache_dtype={kv_cache_dtype}):\n{details}"
                    ) from exc
                finally:
                    self.__class__._stop_active_process()

    def _assert_finite_logprobs(self, outputs: list[dict]) -> None:
        for request_index, output in enumerate(outputs):
            logprobs = output["meta_info"]["output_token_logprobs"]
            self.assertGreater(len(logprobs), 0)
            for token_index, item in enumerate(logprobs):
                self.assertTrue(
                    math.isfinite(float(item[0])),
                    f"request {request_index}, token {token_index}: non-finite logprob",
                )

    def _assert_backend_parity(
        self, baseline: list[dict], actual: list[dict]
    ) -> None:
        self._assert_finite_logprobs(baseline)
        self._assert_finite_logprobs(actual)
        self.assertEqual(len(baseline), len(actual))
        for request_index, (expected, observed) in enumerate(zip(baseline, actual)):
            self.assertEqual(
                expected["text"],
                observed["text"],
                f"request {request_index}: generated text differs",
            )
            expected_logprobs = expected["meta_info"]["output_token_logprobs"]
            observed_logprobs = observed["meta_info"]["output_token_logprobs"]
            self.assertEqual(
                len(expected_logprobs),
                len(observed_logprobs),
                f"request {request_index}: output token count differs",
            )
            for token_index, (expected_item, observed_item) in enumerate(
                zip(expected_logprobs, observed_logprobs)
            ):
                self.assertEqual(
                    expected_item[1],
                    observed_item[1],
                    f"request {request_index}, token {token_index}: token id differs",
                )
                self.assertAlmostEqual(
                    float(expected_item[0]),
                    float(observed_item[0]),
                    delta=LOGPROB_ABS_TOLERANCE,
                    msg=(
                        f"request {request_index}, token {token_index}: "
                        "output logprob differs"
                    ),
                )


class TestDCPSymmA2ATP4(SymmA2ATestBase):
    required_gpus = 4

    def test_bf16_matches_ag_rs_with_graph_and_eager(self):
        for disable_cuda_graph in (False, True):
            with self.subTest(disable_cuda_graph=disable_cuda_graph):
                baseline = self._run_case(
                    tp_size=4,
                    backend="ag_rs",
                    disable_cuda_graph=disable_cuda_graph,
                )
                actual = self._run_case(
                    tp_size=4,
                    backend="symm_a2a",
                    disable_cuda_graph=disable_cuda_graph,
                )
                self.assertIsNotNone(baseline.outputs)
                self.assertIsNotNone(actual.outputs)
                self._assert_backend_parity(baseline.outputs, actual.outputs)


if __name__ == "__main__":
    unittest.main()
