from __future__ import annotations

import io
import os
import string
from typing import ClassVar, Literal, Optional

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.utils import kill_process_tree
from sglang.test.kv_canary.mode_config import _MODE_CONFIGS, _ModeConfig
from sglang.test.kv_canary.utils import build_canary_server_args, post_parallel_generate
from sglang.test.kv_canary.violation_assert_mixin import CanaryViolationAssertMixin
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


class CanaryE2EBase(CanaryViolationAssertMixin, CustomTestCase):
    """Base for canary e2e tests. Subclasses set ``model_mode``, ``kv_canary_mode``,
    ``extra_env``, ``extra_server_args``, ``use_unique_prompts``.

    ``setUpClass`` launches the server with mode-specific args + canary env;
    ``tearDownClass`` kills the server.

    Violation log assertions parse the stable one-line summary emitted by
    ViolationReporter (see python/sglang/srt/kv_canary/runner/violation_reporter.py):
        ``kv_canary violation: launch_tag=<TAG> fail_reason=<NAME[+NAME...]> ...``
    """

    model_mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[CanaryMode]
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

        server_args = build_canary_server_args(
            kv_canary_mode=cls.kv_canary_mode,
            mode_cfg=cls._cfg,
            extra_server_args=cls.extra_server_args,
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
        for buf in (cls._stdout_buf, cls._stderr_buf):
            if buf is not None:
                buf.close()
        cls._stdout_buf = None
        cls._stderr_buf = None

    def make_prompts(self, n: int) -> list[str]:
        if self.use_unique_prompts:
            return _make_unique_prompts(n)
        return [_LONG_PROMPT_BODY] * n

    def send_parallel_requests(
        self,
        n: int,
        *,
        assert_all_success: bool = True,
        max_new_tokens: int = 200,
        timeout: float = 60.0,
    ) -> list[dict]:
        """Fan out n parallel /generate requests; return list of response dicts."""
        results = post_parallel_generate(
            url=self.base_url + "/generate",
            prompts=self.make_prompts(n),
            max_new_tokens=max_new_tokens,
            timeout=timeout,
        )
        if assert_all_success:
            for result in results:
                self.assertEqual(result.get("status_code"), 200, result)
        return results

    def _captured_log_text(
        self, side: Optional[Literal["prefill", "decode"]] = None
    ) -> str:
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
