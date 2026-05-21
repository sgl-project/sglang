"""Default kwargs for spinning up an Engine in mock-model + canary mode.

Mock-model mode is testing-only; the default-filling logic lives here so the
main code (server_args) does not have to know about it.
"""

from __future__ import annotations

import io
import json
import os
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


def mock_model_engine_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return Engine() kwargs that wire up mock-model + canary together.

    Defaults:
        load_format = "dummy"            (no real weights loaded)
        json_model_override_args = '{"num_hidden_layers": 1}'
        sampling_backend = "token_oracle" (gate for install_token_oracle_from_env)
        kv_canary = "raise"              (mock-model without canary is mostly pointless)

    Also sets ``SGLANG_KV_CANARY_INPUT_CHECK=1`` in the current process env so
    the canary's input-id verification path turns on when the engine starts.
    This is a side effect because input-check is mock-model-only and is no
    longer a server arg; the env var is the only injection path.

    ``SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1`` is also set so server_args
    accepts the test-only ``token_oracle`` sampling backend (the choice is
    env-gated to keep it out of production ``--sampling-backend --help``).

    When ``speculative_algorithm`` is set, ``SGLANG_KV_CANARY_INPUT_CHECK`` is
    forced off — the oracle can't predict draft-position tokens, so
    input-check would always fire.

    Caller-supplied overrides win; for json_model_override_args, the override
    dict is merged on top of the default so callers can add extra keys without
    losing num_hidden_layers=1.
    """
    is_spec = "speculative_algorithm" in overrides
    os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "0" if is_spec else "1"
    os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1"

    defaults: dict[str, Any] = {
        "load_format": "dummy",
        "json_model_override_args": json.dumps({"num_hidden_layers": 1}),
        "sampling_backend": "token_oracle",
        "kv_canary": "raise",
    }
    if "json_model_override_args" in overrides:
        user_dict = json.loads(overrides.pop("json_model_override_args"))
        merged = {"num_hidden_layers": 1, **user_dict}
        defaults["json_model_override_args"] = json.dumps(merged)
    defaults.update(overrides)
    return defaults


def mock_model_server_args(*extra_args: str) -> list[str]:
    return [*_MOCK_MODEL_SERVER_ARGS, *extra_args]


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
    env = {
        "SGLANG_KV_CANARY_INPUT_CHECK": "1" if input_check_enabled else "0",
        "SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1",
    }

    process: subprocess.Popen[Any] | None = None
    try:
        process = popen_launch_server(
            MOCK_MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=mock_model_server_args(*extra_server_args),
            env=env,
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
