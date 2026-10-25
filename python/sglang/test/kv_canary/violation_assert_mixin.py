from __future__ import annotations

import time
from typing import Literal, Optional

from sglang.srt.kv_canary.perturb.config import TargetGroupKind
from sglang.test.kv_canary.violation_log_utils import (
    assert_no_violation_in_log,
    find_violation_in_log,
)

_Side = Optional[Literal["prefill", "decode"]]


class CanaryViolationAssertMixin:
    def _captured_log_text(self, side: _Side = None) -> str:
        raise NotImplementedError

    def assert_per_forward_violation_reported(
        self,
        *,
        fail_reason: str,
        target_group: Optional[TargetGroupKind] = None,
        side: _Side = None,
        flush_wait_seconds: float = 2.0,
    ) -> None:
        suffix = "" if target_group is None else f"_{target_group.name}"
        self.assert_violation_logged_any(
            launch_tag_patterns=(f"HEAD_*{suffix}", f"TAIL_*{suffix}"),
            fail_reason=fail_reason,
            side=side,
            flush_wait_seconds=flush_wait_seconds,
        )

    def assert_sweep_violation_reported(
        self,
        *,
        fail_reason: str,
        target_group: TargetGroupKind,
        side: _Side = None,
        flush_wait_seconds: float = 2.0,
        max_retries: int = 4,
    ) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=(f"SWEEP_*_{target_group.name}",),
            fail_reason=fail_reason,
            side=side,
            flush_wait_seconds=flush_wait_seconds,
            max_retries=max_retries,
        )

    def assert_any_launch_tag_violation_reported(
        self,
        *,
        fail_reason: str,
        side: _Side = None,
        flush_wait_seconds: float = 3.0,
        max_retries: int = 10,
    ) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=("*",),
            fail_reason=fail_reason,
            side=side,
            flush_wait_seconds=flush_wait_seconds,
            max_retries=max_retries,
        )

    def assert_any_launch_tag_violation_absent(
        self, *, fail_reason: str, side: _Side = None
    ) -> None:
        self.assert_no_violation_matching(
            launch_tag_patterns=("*",), fail_reason=fail_reason, side=side
        )

    def assert_violation_logged_any(
        self,
        *,
        launch_tag_patterns: tuple[str, ...],
        fail_reason: str,
        side: _Side = None,
        flush_wait_seconds: float = 2.0,
        max_retries: int = 1,
    ) -> None:
        log_text = ""
        for _ in range(max_retries):
            time.sleep(flush_wait_seconds)
            log_text = self._captured_log_text(side)
            if find_violation_in_log(
                log_text,
                launch_tag_patterns=launch_tag_patterns,
                fail_reason=fail_reason,
            ):
                return
        side_label = "" if side is None else f" on side={side}"
        other_side_diag = ""
        if side in ("prefill", "decode"):
            other = "decode" if side == "prefill" else "prefill"
            try:
                other_text = self._captured_log_text(other)
                other_match = find_violation_in_log(
                    other_text,
                    launch_tag_patterns=launch_tag_patterns,
                    fail_reason=fail_reason,
                )
                other_side_diag = (
                    f"\n[diag] other side ({other}) buf len={len(other_text)} "
                    f"contains_match={other_match}"
                )
            except (NotImplementedError, ValueError):
                pass
        raise AssertionError(
            f"No canary violation matching launch_tag_patterns={launch_tag_patterns!r} "
            f"fail_reason={fail_reason!r}{side_label} after max_retries={max_retries} "
            f"(wait={flush_wait_seconds}s each). "
            f"log_text len={len(log_text)}.{other_side_diag} Log tail:\n"
            f"{log_text[-2000:]}"
        )

    def assert_no_violation_matching(
        self,
        *,
        launch_tag_patterns: tuple[str, ...],
        fail_reason: str,
        side: _Side = None,
    ) -> None:
        log_text = self._captured_log_text(side)
        if find_violation_in_log(
            log_text,
            launch_tag_patterns=launch_tag_patterns,
            fail_reason=fail_reason,
        ):
            raise AssertionError(
                f"Unexpected canary violation matching "
                f"launch_tag_patterns={launch_tag_patterns!r} "
                f"fail_reason={fail_reason!r}. Log tail:\n{log_text[-2000:]}"
            )

    def assert_no_violation(
        self,
        *,
        side: _Side = None,
        wait_seconds: float = 2.0,
    ) -> None:
        time.sleep(wait_seconds)
        assert_no_violation_in_log(self._captured_log_text(side))
