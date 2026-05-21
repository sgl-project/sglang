from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModeConfig:
    """Mode-specific server launch config so per-mode test classes only set
    `mode = "mha"` / `"swa"`, not individual flags. All flags collected here.

    Fields:
        model_path: HF model id used by popen_launch_server.
        json_model_override_args: JSON string passed to --json-model-override-args (typically
            shrinks num_hidden_layers so the canary e2e fits a 1-gpu CI budget).
        cuda_graph_max_bs: --cuda-graph-max-bs value.
        max_running_requests: --max-running-requests value.
        context_length: --context-length value.
        max_total_tokens: --max-total-tokens value.
    """

    model_path: str
    json_model_override_args: str
    cuda_graph_max_bs: int
    max_running_requests: int
    context_length: int
    max_total_tokens: int


_MODE_CONFIGS: dict[str, _ModeConfig] = {
    "mha": _ModeConfig(
        model_path="Qwen/Qwen3-0.6B",
        json_model_override_args=json.dumps({"num_hidden_layers": 1}),
        cuda_graph_max_bs=8,
        max_running_requests=32,
        context_length=2048,
        max_total_tokens=16384,
    ),
    "swa": _ModeConfig(
        model_path="google/gemma-3-1b-it",
        json_model_override_args=json.dumps({"num_hidden_layers": 6}),
        cuda_graph_max_bs=8,
        max_running_requests=32,
        context_length=2048,
        max_total_tokens=16384,
    ),
}


class CanaryE2EBase(CustomTestCase):
    """Base for canary e2e tests. Subclasses set ``mode`` and optionally
    ``kv_canary_mode``, ``perturb_env``, ``sweep_interval``, ``use_unique_prompts``.

    ``setUpClass`` launches the server with mode-specific args + canary env;
    ``tearDownClass`` kills the server and cleans env vars set in setUpClass.

    NOTE: assert_violation_logged / assert_no_violation are placeholders this
    commit (raise NotImplementedError); the follow-up commit stabilizes the
    violation log format so they can be implemented for real.
    """

    mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[Literal["log", "raise"]] = "log"
    perturb_env: ClassVar[dict[str, str]] = {}
    sweep_interval: ClassVar[int] = 0
    use_unique_prompts: ClassVar[bool] = False

    process: ClassVar[Optional[object]] = None
    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    _stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _stderr_buf: ClassVar[Optional[io.StringIO]] = None
    _env_keys_set: ClassVar[list[str]] = []
    _cfg: ClassVar[Optional[_ModeConfig]] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._cfg = _MODE_CONFIGS[cls.mode]
        cls._env_keys_set = []
        for k, v in cls.perturb_env.items():
            os.environ[k] = v
            cls._env_keys_set.append(k)
        os.environ["SGLANG_KV_CANARY_SWEEP_INTERVAL"] = str(cls.sweep_interval)
        cls._env_keys_set.append("SGLANG_KV_CANARY_SWEEP_INTERVAL")

        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()

        server_args = [
            "--json-model-override-args",
            cls._cfg.json_model_override_args,
            "--kv-canary",
            cls.kv_canary_mode,
            "--cuda-graph-max-bs",
            str(cls._cfg.cuda_graph_max_bs),
            "--max-running-requests",
            str(cls._cfg.max_running_requests),
            "--context-length",
            str(cls._cfg.context_length),
            "--max-total-tokens",
            str(cls._cfg.max_total_tokens),
        ]
        cls.process = popen_launch_server(
            cls._cfg.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=os.environ.copy(),
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        for k in cls._env_keys_set:
            os.environ.pop(k, None)

    def make_prompts(self, n: int) -> list[str]:
        if self.use_unique_prompts:
            return _make_unique_prompts(n)
        return ["The capital of France is"] * n

    def send_parallel_requests(
        self,
        n: int,
        *,
        max_new_tokens: int = 16,
        timeout: float = 60.0,
    ) -> list[dict]:
        """Fan out n parallel /generate requests; return list of response dicts."""
        prompts = self.make_prompts(n)

        def _send(prompt: str) -> dict:
            try:
                resp = requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": 0.0,
                        },
                    },
                    timeout=timeout,
                )
                return {"status_code": resp.status_code, "body": resp.text}
            except requests.RequestException as exc:
                return {"status_code": -1, "error": repr(exc)}

        with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
            return list(pool.map(_send, prompts))

    def assert_violation_logged(
        self,
        *,
        launch_tag_pattern: str,
        fail_reason: str,
        flush_wait_seconds: float = 2.0,
    ) -> None:
        """Scan server log for a violation line whose launch_tag matches launch_tag_pattern
        (fnmatch) and whose fail_reason set contains fail_reason.

        Placeholder this commit: requires the next commit to stabilize the violation log
        format. Concrete implementation lands then.
        """
        raise NotImplementedError(
            "assert_violation_logged: requires the follow-up commit that stabilizes the "
            "violation log format"
        )

    def assert_no_violation(self, *, wait_seconds: float = 2.0) -> None:
        """No violation should be logged within wait_seconds.

        Placeholder this commit: see assert_violation_logged.
        """
        raise NotImplementedError(
            "assert_no_violation: requires the follow-up commit that stabilizes the "
            "violation log format"
        )


def _make_unique_prompts(n: int) -> list[str]:
    """Each prompt has a unique high-entropy prefix so no two share a radix prefix path.
    Used by perturb_real_kv_unused_cache tests so orphan slots actually stay orphan
    (no future request will hit the corrupted KV)."""
    return [
        f"<{hex(i * 0x9E3779B1 & 0xFFFFFFFF)[2:]}> Tell me a fact." for i in range(n)
    ]
