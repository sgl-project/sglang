"""Host-side decoder for one canary violation row.

The canary kernel writes violation rows into a fixed-width int64 buffer.
This module turns one row (as a plain Python list, the form
``CanaryRunner._pull_first_violation`` returns) into a structured
:class:`CanaryViolationView` for test-side assertion + pretty-print.

Single source of truth for the row layout lives in
``sglang/jit_kernel/kv_cache_canary.py:_VIOLATION_FIELD_*`` — this
module re-imports those constants rather than redefining offsets, so
any future row-layout change there breaks here at import time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from sglang.jit_kernel.kv_cache_canary import (
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    FailReason,
    _VIOLATION_FIELD_ACTUAL_HASH,
    _VIOLATION_FIELD_EXPECTED_HASH,
    _VIOLATION_FIELD_EXPECTED_POSITION,
    _VIOLATION_FIELD_EXPECTED_REQ_ID,
    _VIOLATION_FIELD_FAIL_REASON,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
    _VIOLATION_FIELD_REQ_ID,
    _VIOLATION_FIELD_SLOT_IDX,
    _VIOLATION_FIELD_TOKEN_ID,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryViolationView:
    """One canary violation, decoded for test-side inspection."""

    shadow_kind: str
    kernel_kind: str
    fail_reason: str
    fail_reason_int: int
    slot_idx: int
    req_id: int
    token_id: int
    position: int
    expected_hash: int
    actual_hash: int
    expected_req_id: int
    expected_position: int
    write_index: int

    @classmethod
    def from_row(
        cls,
        *,
        row: List[int],
        shadow_kind: str,
        write_index: int,
    ) -> "CanaryViolationView":
        """Build a view from one ``_pull_first_violation`` row.

        ``shadow_kind`` must be one of the four ``VIOLATION_KINDS``
        (head_k / head_v / tail_k / tail_v); the caller already knows
        which slot it pulled from.
        """
        fail_reason_int = int(row[_VIOLATION_FIELD_FAIL_REASON])
        try:
            fail_reason = FailReason(fail_reason_int).name
        except ValueError:
            fail_reason = f"unknown({fail_reason_int})"

        kernel_kind_int = int(row[_VIOLATION_FIELD_KERNEL_KIND])
        kernel_kind = {
            KERNEL_KIND_HEAD: "HEAD",
            KERNEL_KIND_TAIL: "TAIL",
        }.get(kernel_kind_int, str(kernel_kind_int))

        return cls(
            shadow_kind=shadow_kind,
            kernel_kind=kernel_kind,
            fail_reason=fail_reason,
            fail_reason_int=fail_reason_int,
            slot_idx=int(row[_VIOLATION_FIELD_SLOT_IDX]),
            req_id=int(row[_VIOLATION_FIELD_REQ_ID]),
            token_id=int(row[_VIOLATION_FIELD_TOKEN_ID]),
            position=int(row[_VIOLATION_FIELD_POSITION]),
            expected_hash=int(row[_VIOLATION_FIELD_EXPECTED_HASH]),
            actual_hash=int(row[_VIOLATION_FIELD_ACTUAL_HASH]),
            expected_req_id=int(row[_VIOLATION_FIELD_EXPECTED_REQ_ID]),
            expected_position=int(row[_VIOLATION_FIELD_EXPECTED_POSITION]),
            write_index=int(write_index),
        )

    def is_real(self) -> bool:
        """A row is "real" iff its fail_reason is not NONE."""
        return self.fail_reason_int != int(FailReason.NONE)

    def __str__(self) -> str:
        return (
            f"CanaryViolation(shadow={self.shadow_kind} "
            f"kernel={self.kernel_kind} reason={self.fail_reason} "
            f"slot={self.slot_idx} req_id={self.req_id} "
            f"pos={self.position} write_index={self.write_index})"
        )
