from __future__ import annotations

import fnmatch
import io
import json
import os
import re
import time
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
        context_length=8192,
        max_total_tokens=65536,
    ),
    "swa": _ModeConfig(
        model_path="google/gemma-3-1b-it",
        # Gemma 3 1B-it's HF config carries layer-typed rope params; SGLang's
        # parser also needs an explicit rope_type / factor on full_attention,
        # otherwise the swa-mode server fails to launch. Passing these via
        # --json-model-override-args avoids touching the model source.
        json_model_override_args=json.dumps(
            {
                "num_hidden_layers": 6,
                "rope_parameters": {
                    "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000,
                    },
                    "full_attention": {
                        "rope_type": "default",
                        "rope_theta": 1000000,
                        "factor": 8.0,
                    },
                },
            }
        ),
        cuda_graph_max_bs=8,
        max_running_requests=32,
        context_length=8192,
        max_total_tokens=65536,
    ),
}


# Long prompt body shared by all canary e2e tests. The repetition count is chosen
# so the tokenised prompt comfortably exceeds the SWA sliding window of swa-mode
# fixtures (gemma-3-1b sliding_window = 512); short prompts would never exercise
# the SWA-windowed verify path. Token count is roughly 6k after BPE.
_LONG_PROMPT_BODY = ("The quick brown fox jumps over the lazy dog. " * 700).strip()


class CanaryE2EBase(CustomTestCase):
    """Base for canary e2e tests. Subclasses set ``mode`` and optionally
    ``kv_canary_mode``, ``perturb_env``, ``sweep_interval``, ``use_unique_prompts``.

    ``setUpClass`` launches the server with mode-specific args + canary env;
    ``tearDownClass`` kills the server and cleans env vars set in setUpClass.

    Violation log assertions parse the stable one-line summary emitted by
    ViolationReporter (see python/sglang/srt/kv_canary/runner/violation.py):
        ``kv_canary violation: launch_tag=<TAG> fail_reason=<NAME[+NAME...]> ...``
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
        return [_LONG_PROMPT_BODY] * n

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
        (fnmatch) and whose fail_reason set contains fail_reason exactly.

        Looks for lines of the form
            ``kv_canary violation: launch_tag=<TAG> fail_reason=<NAME[+NAME...]> ...``
        emitted by ViolationReporter. Raises AssertionError if no matching line found.
        """
        time.sleep(flush_wait_seconds)
        log_text = self._captured_log_text()
        line_re = re.compile(r"kv_canary violation: launch_tag=(\S+) fail_reason=(\S+)")
        for match in line_re.finditer(log_text):
            tag = match.group(1)
            reason_field = match.group(2)
            if not fnmatch.fnmatchcase(tag, launch_tag_pattern):
                continue
            if fail_reason in reason_field.split("+"):
                return
        raise AssertionError(
            f"No canary violation matching launch_tag={launch_tag_pattern!r} "
            f"fail_reason={fail_reason!r} found in server log. Log tail:\n"
            f"{log_text[-2000:]}"
        )

    def assert_no_violation(self, *, wait_seconds: float = 2.0) -> None:
        """Assert no ``kv_canary violation:`` line appears in the captured server log within
        wait_seconds."""
        time.sleep(wait_seconds)
        log_text = self._captured_log_text()
        if "kv_canary violation:" in log_text:
            raise AssertionError(
                f"Unexpected canary violation found. Log tail:\n{log_text[-2000:]}"
            )

    def _captured_log_text(self) -> str:
        stdout_text = (
            self._stdout_buf.getvalue() if self._stdout_buf is not None else ""
        )
        stderr_text = (
            self._stderr_buf.getvalue() if self._stderr_buf is not None else ""
        )
        return stdout_text + stderr_text


def _make_unique_prompts(n: int) -> list[str]:
    """Each prompt has a unique high-entropy prefix so no two share a radix prefix path.
    Used by perturb_real_kv_unused_cache tests so orphan slots actually stay orphan
    (no future request will hit the corrupted KV). The body is the shared
    _LONG_PROMPT_BODY so the prompt still exceeds the SWA sliding window."""
    return [
        f"<{hex(i * 0x9E3779B1 & 0xFFFFFFFF)[2:]}> {_LONG_PROMPT_BODY}"
        for i in range(n)
    ]
