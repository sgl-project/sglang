from __future__ import annotations

import uuid
from typing import ClassVar, Literal, Optional

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.kv_canary.mode_config import _MODE_CONFIGS, _ModeConfig
from sglang.test.kv_canary.utils import build_canary_server_args, post_parallel_generate
from sglang.test.kv_canary.violation_assert_mixin import CanaryViolationAssertMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

_SHORT_PROMPT_BODY = ("The quick brown fox jumps over the lazy dog. " * 8).strip()


class CanaryPDFixture(CanaryViolationAssertMixin, PDDisaggregationServerBase):
    capture_per_side_logs = True

    model_mode: ClassVar[Literal["mha", "swa", "dsv4"]]
    kv_canary_mode: ClassVar[CanaryMode] = CanaryMode.LOG
    extra_server_args: ClassVar[tuple[str, ...]] = (
        "--kv-canary-real-data",
        "partial",
        "--skip-server-warmup",
    )

    _cfg: ClassVar[Optional[_ModeConfig]] = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._cfg = _MODE_CONFIGS[cls.model_mode]
        cls.model = cls._cfg.model_path

        canary_args = build_canary_server_args(
            kv_canary_mode=cls.kv_canary_mode,
            mode_cfg=cls._cfg,
            extra_server_args=cls.extra_server_args,
        )
        cls.extra_prefill_args = list(canary_args)
        cls.extra_decode_args = list(canary_args)
        if cls.model_mode == "swa":
            # SWA mode uses google/gemma-4-E2B-it, whose forward does a
            # ``positions += 1`` in-place. canary's WRITE/VERIFY require
            # forward_batch.positions to stay 0-indexed, so flip the gemma
            # path to out-of-place shift for these tests.
            cls.extra_prefill_env = {
                **cls.extra_prefill_env,
                "SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION": "1",
            }
            cls.extra_decode_env = {
                **cls.extra_decode_env,
                "SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION": "1",
            }
        cls.launch_all()

    def send_parallel_short_requests(
        self,
        n: int,
        *,
        assert_all_success: bool = True,
        max_new_tokens: int = 100,
        timeout: float = 60.0,
        distinct_prompts: bool = False,
    ) -> list[dict]:
        if distinct_prompts:
            # Diverge at the very first token (request index before the per-call
            # nonce) so requests share no radix-dedupable prefix beyond a
            # tokenizer-added BOS, and retries never hit earlier attempts' cache.
            nonce = uuid.uuid4().hex[:8]
            prompts = [f"{i} {nonce} {_SHORT_PROMPT_BODY}" for i in range(n)]
        else:
            prompts = [_SHORT_PROMPT_BODY] * n
        results = post_parallel_generate(
            url=self.lb_url + "/generate",
            prompts=prompts,
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
