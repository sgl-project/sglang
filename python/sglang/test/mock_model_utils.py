"""Default args for launching a mock-model + canary test server.

Mock-model mode is testing-only; the default-filling logic lives here so the
main code (server_args) does not have to know about it.
"""

from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    get_benchmark_args,
    popen_launch_server,
)

MOCK_MODEL_PATH = "Qwen/Qwen3-0.6B"

_MOCK_MODEL_SERVER_ARGS: list[str] = [
    "--load-format",
    "dummy",
    "--json-model-override-args",
    json.dumps({"num_hidden_layers": 1}),
    "--sampling-backend",
    "token_oracle",
    "--kv-canary",
    "raise",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class MockModelBenchResult:
    result: dict[str, Any]
    stdout: str
    stderr: str
    server_return_code: int | None

    @property
    def log_text(self) -> str:
        return self.stdout + self.stderr

    def log_tail(self, length: int = 2000) -> str:
        return self.log_text[-length:]


def mock_model_server_args(*extra_args: str) -> list[str]:
    return [*_MOCK_MODEL_SERVER_ARGS, *extra_args]


def mock_model_server_env(*, input_check_enabled: bool = True) -> dict[str, str]:
    """Return env overrides for popen_launch_server in mock-model + canary mode."""
    return {
        "SGLANG_KV_CANARY_INPUT_CHECK": "1" if input_check_enabled else "0",
        "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
    }


def run_mock_model_bench_serving(
    *,
    extra_server_args: Sequence[str],
    input_check_enabled: bool = True,
    num_prompts: int = 8,
    random_input_len: int = 6144,
    random_output_len: int = 1024,
) -> MockModelBenchResult:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    process: subprocess.Popen[Any] | None = None
    try:
        process = popen_launch_server(
            MOCK_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=mock_model_server_args(*extra_server_args),
            env=mock_model_server_env(input_check_enabled=input_check_enabled),
            return_stdout_stderr=(stdout_buf, stderr_buf),
        )

        args = get_benchmark_args(
            base_url=DEFAULT_URL_FOR_TEST,
            dataset_name="random",
            tokenizer=MOCK_MODEL_PATH,
            num_prompts=num_prompts,
            random_input_len=random_input_len,
            random_output_len=random_output_len,
            request_rate=float("inf"),
            max_concurrency=num_prompts,
        )
        args.random_range_ratio = 1.0
        args.warmup_requests = 0
        args.disable_tqdm = True

        result = run_benchmark(args)
        server_return_code = process.poll()

        return MockModelBenchResult(
            result=result,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            server_return_code=server_return_code,
        )
    finally:
        if process is not None:
            kill_process_tree(process.pid)
