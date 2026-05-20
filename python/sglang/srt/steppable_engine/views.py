from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class SteppableReqHandle:
    rid: str
    prompt_len: int
    max_new_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocatorStats:
    free: int
    used: int
    held: int
    total: int
    held_slots: FrozenSet[int]


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryViolationView:
    fail_reason_name: str
    fail_reason_bits: int
    req_pool_idx: int
    position: int
    expected: int
    actual: int
    runner_kind: str
    step_id: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryWritePlanView:
    num_write: int
    num_verify: int
    write_slot_indices: List[int]
    verify_slot_indices: List[int]
    expected_input_tokens: Optional[List[int]]
    expected_input_positions: Optional[List[int]]
