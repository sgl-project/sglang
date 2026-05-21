from __future__ import annotations

import fnmatch
import io
import os
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Literal, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kv_canary.mode_config import _MODE_CONFIGS, _ModeConfig
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# Long prompt body shared by all canary e2e tests. The repetition count is chosen
# so the tokenised prompt comfortably exceeds the SWA sliding window of swa-mode
# fixtures (gemma-3-1b sliding_window = 512); short prompts would never exercise
# the SWA-windowed verify path. Token count is roughly 6k after BPE.
_LONG_PROMPT_BODY = ("The quick brown fox jumps over the lazy dog. " * 700).strip()
_UNIQUE_PROMPT_FIRST_CHARS = string.ascii_letters + string.digits


class CanaryE2EBase(CustomTestCase):
    """Base for canary e2e tests. Subclasses set ``model_mode``, ``kv_canary_mode``,
    ``extra_env``, ``extra_server_args``, ``use_unique_prompts``.

    ``setUpClass`` launches the server with mode-specific args + canary env;
    ``tearDownClass`` kills the server.

    Violation log assertions parse the stable one-line summary emitted by
    ViolationReporter (see python/sglang/srt/kv_canary/runner/violation_reporter.py):
        ``kv_canary violation: launch_tag=<TAG> fail_reason=<NAME[+NAME...]> ...``
    """

    model_mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[Literal["off", "log", "raise"]]
    extra_env: ClassVar[dict[str, str]] = {}
    extra_server_args: ClassVar[tuple[str, ...]] = ()
    use_unique_prompts: ClassVar[bool] = False

    process: ClassVar[Optional[object]] = None
    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    _stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _stderr_buf: ClassVar[Optional[io.StringIO]] = None
    _cfg: ClassVar[Optional[_ModeConfig]] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._cfg = _MODE_CONFIGS[cls.model_mode]
        server_env = os.environ.copy()
        server_env.update(cls.extra_env)

        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()

        server_args = [
            "--kv-canary",
            cls.kv_canary_mode,
            "--context-length",
            "8192",
            *cls.extra_server_args,
        ]
        if cls._cfg.json_model_override_args is not None:
            server_args.extend(
                [
                    "--json-model-override-args",
                    cls._cfg.json_model_override_args,
                ]
            )
        cls.process = popen_launch_server(
            cls._cfg.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=server_env,
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def make_prompts(self, n: int) -> list[str]:
        if self.use_unique_prompts:
            return _make_unique_prompts(n)
        return [_LONG_PROMPT_BODY] * n

    def send_parallel_requests(
        self,
        n: int,
        *,
        assert_all_successs: bool = True,
        max_new_tokens: int = 200,
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
            results = list(pool.map(_send, prompts))

        if assert_all_successs:
            for result in results:
                self.assertEqual(result.get("status_code"), 200, result)

        return results

    def assert_per_forward_violation_reported(
        self,
        *,
        fail_reason: str,
        target_group: Optional[Literal["full", "swa"]] = None,
        flush_wait_seconds: float = 2.0,
    ) -> None:
        suffix = "" if target_group is None else f"_{target_group.upper()}"
        self.assert_violation_logged_any(
            launch_tag_patterns=(f"HEAD_*{suffix}", f"TAIL_*{suffix}"),
            fail_reason=fail_reason,
            flush_wait_seconds=flush_wait_seconds,
        )

    def assert_sweep_violation_reported(
        self,
        *,
        fail_reason: str,
        target_group: Literal["full", "swa"],
        flush_wait_seconds: float = 2.0,
    ) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=(f"SWEEP_*_{target_group.upper()}",),
            fail_reason=fail_reason,
            flush_wait_seconds=flush_wait_seconds,
        )

    def assert_violation_logged_any(
        self,
        *,
        launch_tag_patterns: tuple[str, ...],
        fail_reason: str,
        flush_wait_seconds: float = 2.0,
    ) -> None:
        """Scan server log for a violation line whose launch_tag matches any pattern
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
            if fail_reason not in reason_field.split("+"):
                continue
            if any(
                fnmatch.fnmatchcase(tag, pattern) for pattern in launch_tag_patterns
            ):
                return
        raise AssertionError(
            f"No canary violation matching launch_tag_patterns={launch_tag_patterns!r} "
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
    if n > len(_UNIQUE_PROMPT_FIRST_CHARS):
        raise ValueError(
            f"unique prompt count {n} exceeds supported count "
            f"{len(_UNIQUE_PROMPT_FIRST_CHARS)}"
        )

    return [
        (
            f"{_UNIQUE_PROMPT_FIRST_CHARS[i]}"
            f"{hex(i * 0x9E3779B1 & 0xFFFFFFFF)[2:]} "
            f"{_LONG_PROMPT_BODY}"
        )
        for i in range(n)
    ]
