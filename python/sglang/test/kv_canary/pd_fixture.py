from __future__ import annotations

import fnmatch
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Literal, Optional

import requests

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.kv_canary.mode_config import _MODE_CONFIGS, _ModeConfig
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

_SHORT_PROMPT_BODY = ("The quick brown fox jumps over the lazy dog. " * 8).strip()


class CanaryPDFixture(PDDisaggregationServerBase):
    """Base for PD disagg canary self-tests.

    Launches P + D + LB with ``--kv-canary <mode>`` and a per-mode model
    (MHA: Qwen3-0.6B, SWA: gemma-3-1b-it). Subclasses set ``model_mode`` and
    optionally ``extra_prefill_env`` / ``extra_decode_env`` to drive P-only or
    D-only perturbation. Log-assertion helpers parse the ViolationReporter line
    format on either side independently.
    """

    model_mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[Literal["off", "log", "raise"]] = "log"
    extra_server_args: ClassVar[tuple[str, ...]] = ("--kv-canary-real-data", "partial")

    _cfg: ClassVar[Optional[_ModeConfig]] = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._cfg = _MODE_CONFIGS[cls.model_mode]
        cls.model = cls._cfg.model_path

        canary_args = [
            "--kv-canary",
            cls.kv_canary_mode,
            "--context-length",
            "8192",
            *cls.extra_server_args,
        ]
        if cls._cfg.json_model_override_args is not None:
            canary_args.extend(
                [
                    "--json-model-override-args",
                    cls._cfg.json_model_override_args,
                ]
            )
        cls.extra_prefill_args = list(canary_args)
        cls.extra_decode_args = list(canary_args)
        cls.launch_all()

    def send_parallel_short_requests(
        self,
        n: int,
        *,
        assert_all_success: bool = True,
        max_new_tokens: int = 1,
        timeout: float = 60.0,
    ) -> list[dict]:
        """Fan out n parallel /generate requests with short prompts so each request fits
        in a single prefill forward (no chunked-prefill on the P side)."""
        prompts = [_SHORT_PROMPT_BODY] * n

        def _send(prompt: str) -> dict:
            try:
                resp = requests.post(
                    self.lb_url + "/generate",
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

        if assert_all_success:
            for result in results:
                self.assertEqual(result.get("status_code"), 200, result)

        return results

    def assert_d_per_forward_violation_reported(
        self,
        *,
        fail_reason: str,
        target_group: TargetGroupKind,
        flush_wait_seconds: float = 4.0,
    ) -> None:
        """Scan the decode-side log for a HEAD_*/TAIL_* canary violation matching the
        target group. The longer default flush_wait_seconds accounts for D-side D2H
        pump latency plus PD transfer overhead."""
        suffix = f"_{target_group.name}"
        self._assert_violation_logged_any(
            side="decode",
            launch_tag_patterns=(f"HEAD_*{suffix}", f"TAIL_*{suffix}"),
            fail_reason=fail_reason,
            flush_wait_seconds=flush_wait_seconds,
        )

    def assert_no_violation_on(
        self,
        side: Literal["prefill", "decode"],
        *,
        wait_seconds: float = 2.0,
    ) -> None:
        time.sleep(wait_seconds)
        log_text = self._captured_log_text(side)
        if "kv_canary violation:" in log_text:
            raise AssertionError(
                f"Unexpected canary violation on side={side}. Log tail:\n"
                f"{log_text[-2000:]}"
            )

    def _assert_violation_logged_any(
        self,
        *,
        side: Literal["prefill", "decode"],
        launch_tag_patterns: tuple[str, ...],
        fail_reason: str,
        flush_wait_seconds: float,
    ) -> None:
        time.sleep(flush_wait_seconds)
        log_text = self._captured_log_text(side)
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
            f"fail_reason={fail_reason!r} on side={side}. Log tail:\n"
            f"{log_text[-2000:]}"
        )

    def _captured_log_text(self, side: Literal["prefill", "decode"]) -> str:
        if side == "prefill":
            stdout_buf = type(self)._prefill_stdout_buf
            stderr_buf = type(self)._prefill_stderr_buf
        elif side == "decode":
            stdout_buf = type(self)._decode_stdout_buf
            stderr_buf = type(self)._decode_stderr_buf
        else:
            raise ValueError(f"Unsupported side={side!r}")
        stdout_text = stdout_buf.getvalue() if stdout_buf is not None else ""
        stderr_text = stderr_buf.getvalue() if stderr_buf is not None else ""
        return stdout_text + stderr_text
