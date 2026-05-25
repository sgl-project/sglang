from __future__ import annotations

import io
import os
import string
import time
from typing import ClassVar, Literal, Optional

from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.runner.swa_divergence import SwaDivergenceLog
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
# fixtures (gemma-4-E2B); short prompts would never exercise the SWA-windowed
# verify path. Token count is roughly 7k after BPE.
_LONG_PROMPT_BODY = ("The quick brown fox jumps over the lazy dog. " * 700).strip()
_UNIQUE_PROMPT_FIRST_CHARS = string.ascii_letters + string.digits


class CapturedServerE2EBase(CanaryViolationAssertMixin, CustomTestCase):
    process: ClassVar[Optional[object]] = None
    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    _stdout_buf: ClassVar[Optional[io.StringIO]] = None
    _stderr_buf: ClassVar[Optional[io.StringIO]] = None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        for buf in (cls._stdout_buf, cls._stderr_buf):
            if buf is not None:
                buf.close()
        cls._stdout_buf = None
        cls._stderr_buf = None

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

    def assert_log_contains(self, substring: str) -> None:
        log_text = self._captured_log_text()
        if substring not in log_text:
            raise AssertionError(
                f"Expected substring {substring!r} not found in captured log. "
                f"Log tail:\n{log_text[-2000:]}"
            )


class CanaryE2EBase(CapturedServerE2EBase):
    model_mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[CanaryMode]
    extra_env: ClassVar[dict[str, str]] = {}
    extra_server_args: ClassVar[tuple[str, ...]] = ()
    use_unique_prompts: ClassVar[bool] = False
    # SWA divergence assertions need slot recycling across batches; setting > 1 makes the
    # test methods send N sequential batches so the SWA allocator's full→swa index mapping
    # diverges from identity. Default 1 keeps MHA tests fast.
    workload_n_batches: ClassVar[int] = 1

    _cfg: ClassVar[Optional[_ModeConfig]] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._cfg = _MODE_CONFIGS[cls.model_mode]
        server_env = os.environ.copy()
        server_env.update(cls.extra_env)
        if cls.model_mode == "swa":
            server_env.setdefault(
                "SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL", "20"
            )
            # SWA mode uses google/gemma-4-E2B-it, whose forward does a
            # ``positions += 1`` in-place. canary's WRITE/VERIFY require
            # forward_batch.positions to stay 0-indexed, so flip the gemma
            # path to out-of-place shift for these tests.
            server_env.setdefault("SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION", "1")

        cls._stdout_buf = io.StringIO()
        cls._stderr_buf = io.StringIO()

        server_args = build_canary_server_args(
            kv_canary_mode=cls.kv_canary_mode,
            mode_cfg=cls._cfg,
            extra_server_args=("--max-total-tokens", "65536", *cls.extra_server_args),
        )
        cls.process = popen_launch_server(
            cls._cfg.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=server_env,
            return_stdout_stderr=(cls._stdout_buf, cls._stderr_buf),
        )

    def make_prompts(self, n: int) -> list[str]:
        if self.use_unique_prompts:
            return _make_unique_prompts(n)
        return [_LONG_PROMPT_BODY] * n

    def send_parallel_requests(
        self,
        n: int = 8,
        *,
        assert_all_success: bool = True,
        max_new_tokens: int = 2048,
        timeout: float = 240.0,
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

    def maybe_assert_swa_divergence_observed(self) -> None:
        if self.model_mode == "swa":
            self.assert_swa_divergence_observed()

    def assert_swa_divergence_observed(
        self,
        *,
        min_swa_full_idx_divergence: int = 0,
        require_verify_lag: bool = True,
        flush_wait_seconds: float = 3.0,
        max_retries: int = 10,
    ) -> None:
        """Assert that the SWA path was genuinely exercised.

        The `verify_swa < verify_full` check is the durable signal — gemma's 1024-token SWA
        window naturally produces fewer SWA verify entries than FULL entries as soon as any
        prompt exceeds the window. `swa_full_idx_divergence >= 1` would additionally require
        non-identity SWA-pool remapping (i.e. actual eviction), but triggering that in a
        non-flaky way also drives the canary into a known FULL-kernel position false-positive
        path; default min is 0 to keep the broader signal as the gate.
        """
        last_parsed = None
        last_line: str = ""
        for _ in range(max_retries):
            time.sleep(flush_wait_seconds)
            log_text = self._captured_log_text()
            found = SwaDivergenceLog.find_last(log_text)
            if found is not None:
                last_parsed, last_line = found
                break

        if last_parsed is None:
            raise AssertionError(
                "No kv_canary swa_divergence line found in server log after "
                f"{max_retries} retries (wait={flush_wait_seconds}s each). "
                f"Log tail:\n{self._captured_log_text()[-2000:]}"
            )

        if last_parsed.swa_full_idx_divergence < min_swa_full_idx_divergence:
            raise AssertionError(
                f"SWA divergence not observed: swa_full_idx_divergence={last_parsed.swa_full_idx_divergence} "
                f"< min={min_swa_full_idx_divergence}. Line: {last_line}"
            )
        if require_verify_lag and not (
            last_parsed.verify_swa < last_parsed.verify_full
        ):
            raise AssertionError(
                f"SWA divergence not observed: verify_swa={last_parsed.verify_swa} "
                f"not strictly less than verify_full={last_parsed.verify_full}. "
                f"Line: {last_line}"
            )


def _make_unique_prompts(n: int) -> list[str]:
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
