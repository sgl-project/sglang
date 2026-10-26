# SPDX-License-Identifier: Apache-2.0
"""Control-request protocol between the HTTP process and scheduler workers.

These types are cross-process IPC contracts, not utilities: the HTTP side
constructs them (scheduler_client treats them as ``_CONTROL_REQ_TYPES`` and
fans them out to every replica) and each scheduler dispatches them through
``Scheduler.request_handlers``. Keep this module import-light -- both
processes import it, and the HTTP process must not drag in torch-heavy
worker modules through it.
"""

from typing import List, Optional, Union

import msgspec


class SetLoraReq(msgspec.Struct):
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    merge_mode: Optional[str] = None
    lora_alpha: Optional[Union[int, List[Optional[int]]]] = None


class MergeLoraWeightsReq(msgspec.Struct):
    target: str = "all"
    strength: float = 1.0


class UnmergeLoraWeightsReq(msgspec.Struct):
    target: str = "all"


class ListLorasReq(msgspec.Struct):
    pass


class ShutdownReq(msgspec.Struct):
    pass


class ReleaseRealtimeSessionReq(msgspec.Struct):
    session_id: str


class GetDisaggStatsReq(msgspec.Struct):
    """Request to get disagg pipeline metrics from the scheduler."""

    pass


class AutoResidencyReq(msgspec.Struct, frozen=True):
    """Plan, apply, or roll back automatic component residency.

    Sent by the server warmup orchestrator after the synthetic warmup
    requests finish and before the server reports ready.
    """

    action: str = "apply"  # "apply" | "apply_static" | "validate" | "rollback"
