from __future__ import annotations

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
    """Base for PD disagg canary self-tests."""

    capture_per_side_logs = True

    model_mode: ClassVar[Literal["mha", "swa"]]
    kv_canary_mode: ClassVar[CanaryMode] = CanaryMode.LOG
    # --skip-server-warmup: the sglang HTTP warmup uses FAKE_BOOTSTRAP_HOST to bypass real
    # mooncake transfer, which leaves decode-side canary buffers uninitialised and makes
    # HEAD_K_FULL verify fire false-positive chain_hash violations during boot. Until the
    # canary runner learns to mask fake-transfer batches, skip the HTTP warmup on both sides.
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
        cls.launch_all()

    def send_parallel_short_requests(
        self,
        n: int,
        *,
        assert_all_success: bool = True,
        max_new_tokens: int = 100,
        timeout: float = 60.0,
    ) -> list[dict]:
        """Fan out n parallel /generate requests with short prompts.

        ``max_new_tokens`` defaults to 100 so D-side actually runs decode forwards rather
        than only returning the prefill bonus token. At max_new_tokens=1, the prefill
        bonus-token forward is the only generation step and D has no opportunity to run
        canary verify on the transferred prefix slots; tests asserting D-side detection
        would silently never exercise that code path.
        """
        results = post_parallel_generate(
            url=self.lb_url + "/generate",
            prompts=[_SHORT_PROMPT_BODY] * n,
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
