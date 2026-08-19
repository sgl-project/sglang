# SPDX-License-Identifier: Apache-2.0
"""Control-request protocol between the HTTP process and scheduler workers.

These types are cross-process IPC contracts, not utilities: the HTTP side
constructs them (scheduler_client treats them as ``_CONTROL_REQ_TYPES`` and
fans them out to every replica) and each scheduler dispatches them through
``Scheduler.request_handlers``. Keep this module import-light -- both
processes import it, and the HTTP process must not drag in torch-heavy
worker modules through it.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import msgspec


@dataclass
class SetLoraReq:
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    merge_mode: Optional[str] = None
    lora_alpha: Optional[Union[int, List[Optional[int]]]] = None


@dataclass
class MergeLoraWeightsReq:
    target: str = "all"
    strength: float = 1.0


@dataclass
class UnmergeLoraWeightsReq:
    target: str = "all"


@dataclass
class ListLorasReq:
    pass


@dataclass
class ShutdownReq:
    pass


@dataclass
class ReleaseRealtimeSessionReq:
    session_id: str


@dataclass
class GetDisaggStatsReq:
    """Request to get disagg pipeline metrics from the scheduler."""

    pass


class AutoResidencyReq(msgspec.Struct, frozen=True):
    """Apply or roll back the warmup-calibrated residency promotion.

    Sent by the server warmup orchestrator after the synthetic warmup
    requests finish and before the server reports ready.
    """

    action: str = "apply"  # "apply" | "rollback"
