"""The owners of the scheduler's modes: a live object shared with the
components that read them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DisaggDecode
    from sglang.srt.disaggregation.prefill import DisaggPrefill
    from sglang.srt.dllm.manager import DllmManager
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator


@dataclass(kw_only=True, slots=True)
class ModeOwners:
    """The owner of each mode that is on, None for the others. A PD role
    switch replaces the PD owners here, in place, so a component holding this
    never reads a torn-down owner."""

    disagg_prefill: Optional[DisaggPrefill] = None
    disagg_decode: Optional[DisaggDecode] = None
    dllm: Optional[DllmManager] = None
    hisparse_coordinator: Optional[HiSparseCoordinator] = None
