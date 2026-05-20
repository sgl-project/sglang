from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from sglang.srt.managers.io_struct import BaseReq


@dataclass(slots=True, kw_only=True)
class EnterSteppingModeReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class StepReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class CanaryViolationsReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class AllocatorStatsReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class BlockTableReq(BaseReq):
    rid: str


@dataclass(slots=True, kw_only=True)
class OutputHistoryReq(BaseReq):
    rid: str


@dataclass(slots=True, kw_only=True)
class IsActiveReq(BaseReq):
    rid: str


@dataclass(slots=True, kw_only=True)
class ActiveReqsReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class LastWritePlanReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class CanaryOverheadPctReq(BaseReq):
    pass


@dataclass(slots=True, kw_only=True)
class InjectPerturbationReq(BaseReq):
    channel: str
    kind: str
    rank: Optional[int] = None


@dataclass(slots=True, kw_only=True)
class CanaryViolationsResp(BaseReq):
    violations_pickled: bytes


@dataclass(slots=True, kw_only=True)
class AllocatorStatsResp(BaseReq):
    free: int
    used: int
    held: int
    total: int
    held_slots_pickled: bytes


@dataclass(slots=True, kw_only=True)
class BlockTableResp(BaseReq):
    slot_indices: List[int]


@dataclass(slots=True, kw_only=True)
class OutputHistoryResp(BaseReq):
    tokens: List[int]


@dataclass(slots=True, kw_only=True)
class IsActiveResp(BaseReq):
    active: bool


@dataclass(slots=True, kw_only=True)
class ActiveReqsResp(BaseReq):
    handles_pickled: bytes


@dataclass(slots=True, kw_only=True)
class LastWritePlanResp(BaseReq):
    plan_pickled: Optional[bytes]


@dataclass(slots=True, kw_only=True)
class CanaryOverheadPctResp(BaseReq):
    pct: float


_OBSERVATION_TYPES: Tuple[type, ...] = (
    CanaryViolationsReq,
    AllocatorStatsReq,
    BlockTableReq,
    OutputHistoryReq,
    IsActiveReq,
    ActiveReqsReq,
    LastWritePlanReq,
    CanaryOverheadPctReq,
)
