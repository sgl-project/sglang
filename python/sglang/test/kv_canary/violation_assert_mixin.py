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
    """Canary-violation log assertions shared by single-server and PD test bases.

    Subclasses implement ``_captured_log_text(side)``. For single-server bases ``side``
    is always None; for PD bases it selects between prefill and decode captured logs.
    """

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
    ) -> None:
        self.assert_violation_logged_any(
            launch_tag_patterns=(f"SWEEP_*_{target_group.name}",),
            fail_reason=fail_reason,
            side=side,
            flush_wait_seconds=flush_wait_seconds,
        )

    def assert_violation_logged_any(
        self,
        *,
        launch_tag_patterns: tuple[str, ...],
        fail_reason: str,
        side: _Side = None,
        flush_wait_seconds: float = 2.0,
    ) -> None:
        time.sleep(flush_wait_seconds)
        log_text = self._captured_log_text(side)
        if find_violation_in_log(
            log_text,
            launch_tag_patterns=launch_tag_patterns,
            fail_reason=fail_reason,
        ):
            return
        side_label = "" if side is None else f" on side={side}"
        # DEBUG: dump both sides' buffer sizes and violation-line counts so we can tell whether
        # the per-side capture is correctly routing canary violations.
        debug_summary = ""
        for dbg_side in ("prefill", "decode"):
            try:
                dbg_text = self._captured_log_text(dbg_side)
                v_count = dbg_text.count("kv_canary violation:")
                debug_summary += (
                    f"  [debug] side={dbg_side} len={len(dbg_text)} "
                    f"violation_lines={v_count}\n"
                )
            except Exception as exc:  # noqa: BLE001
                debug_summary += f"  [debug] side={dbg_side} unavailable: {exc!r}\n"
        raise AssertionError(
            f"No canary violation matching launch_tag_patterns={launch_tag_patterns!r} "
            f"fail_reason={fail_reason!r}{side_label}.\n"
            f"{debug_summary}"
            f"Log tail:\n{log_text[-2000:]}"
        )

    def assert_no_violation(
        self,
        *,
        side: _Side = None,
        wait_seconds: float = 2.0,
    ) -> None:
        time.sleep(wait_seconds)
        assert_no_violation_in_log(self._captured_log_text(side))
